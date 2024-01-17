import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.attention import Attention
import math
from models.pos_enc import WordPosEnc

# import numpy as np
# from counting_utils import gen_counting_label


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):  # position_embedding = PositionEmbeddingSine(256, normalize=True)
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(
        self, x, mask
    ):  # pos = position_embedding(cnn_features_trans, images_mask[:,0,:,:])
        y_embed = mask.cumsum(1, dtype=torch.float32)  # 返回给定axis上的累计和
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(
            3
        )  # []
        pos = torch.cat((pos_y, pos_x), dim=3).permute(
            0, 3, 1, 2
        )  # [b, h", w", c] --> [b, c, h", w"]     channel = 256 + 256 --> 512
        return pos


class AttDecoder(nn.Module):
    def __init__(self, params):
        super(AttDecoder, self).__init__()
        self.params = params
        self.input_size = params["decoder"]["input_size"]  # 256
        self.hidden_size = params["decoder"]["hidden_size"]  # 256
        self.out_channel = params["encoder"]["out_channel"]  # 684
        self.attention_dim = params["attention"]["attention_dim"]  # 512
        self.dropout_prob = params["dropout"]  # True
        self.device = params["device"]
        self.word_num = params["word_num"]  # len(words)
        self.counting_num = params["counting_decoder"]["out_channel"]  # 111
        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params["densenet"]["ratio"]  # 16
        # init hidden state
        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)  # 684 -> 256
        # word embedding
        self.embedding = nn.Embedding(
            self.word_num, self.input_size
        )  # len(words) -> 256
        # word gru
        self.word_input_gru = nn.GRUCell(
            self.input_size, self.hidden_size
        )  # 256 -> 256
        self.word_output_gru = nn.GRUCell(
            self.out_channel, self.hidden_size
        )  # 256 -> 256
        # attention
        self.word_attention = Attention(params)
        self.encoder_feature_conv = nn.Conv2d(
            self.out_channel,
            self.attention_dim,  # 684 -> 512
            kernel_size=params["attention"]["word_conv_kernel"],  # 1
            padding=params["attention"]["word_conv_kernel"] // 2,
        )
        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size)
        self.word_embedding_weight = nn.Linear(self.input_size, self.hidden_size)
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.counting_context_weight = nn.Linear(self.counting_num, self.hidden_size)
        self.word_convert = nn.Linear(self.hidden_size, self.word_num)
        self.relu = nn.ReLU()
        if params["dropout"]:
            self.dropout = nn.Dropout(params["dropout_ratio"])

    def forward(
        self,
        cnn_features,
        labels,
        counting_preds,
        images_mask,
        labels_mask,
        is_train=True,
    ):
        batch_size, num_steps = labels.shape  # [b, l']
        height, width = cnn_features.shape[2:]  # [h", w"]
        word_probs = torch.zeros((batch_size, num_steps, self.word_num)).to(
            device=self.device
        )  # [b, l', len(words) = 111]  初始化为全0

        images_mask = images_mask[
            :, :, :: self.ratio, :: self.ratio
        ]  # [b, c, h", w"]  c = 1
        word_alpha_sum = torch.zeros((batch_size, 1, height, width)).to(
            device=self.device
        )  # [b, 1, h", w"]
        word_alphas = torch.zeros((batch_size, num_steps, height, width)).to(
            device=self.device
        )  # [b, l', h", w"]
        hidden = self.init_hidden(
            cnn_features, images_mask, self.init_weight
        )  # [b, 256]  初始化hidden
        counting_context_weighted = self.counting_context_weight(
            counting_preds
        )  # C：[b, 111]  ==>  [b, 256]

        cnn_features_trans = self.encoder_feature_conv(
            cnn_features
        )  # [b, 684, h", w"] ==> [b, 512, h", w"]    T
        position_embedding = PositionEmbeddingSine(256, normalize=True)  # 512/2
        pos = position_embedding(
            cnn_features_trans, images_mask[:, 0, :, :]
        )  # pos_embedding   P
        cnn_features_trans = cnn_features_trans + pos  # T'

        if is_train:
            for i in range(num_steps):  # num_steps = l'
                word_embedding = (
                    self.embedding(labels[:, i - 1])
                    if i
                    else self.embedding(torch.ones([batch_size]).long().to(self.device))
                )
                hidden = self.word_input_gru(
                    word_embedding, hidden
                )  # ht    yt-1 --> embedding --> GRU --> ht
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(
                    cnn_features,
                    cnn_features_trans,
                    hidden,
                    word_alpha_sum,
                    images_mask,
                )
                print("word_context_vec.shape: ", word_context_vec.shape)
                hidden = self.word_output_gru(word_context_vec, hidden)
                current_state = self.word_state_weight(hidden)  # ht： current/hidden
                word_weighted_embedding = self.word_embedding_weight(
                    word_embedding
                )  # E(yt-1) word embeding -> 256
                word_context_weighted = self.word_context_weight(
                    word_context_vec
                )  # V: context vec -> 256

                if self.params["dropout"]:
                    word_out_state = self.dropout(
                        current_state
                        + word_weighted_embedding
                        + word_context_weighted
                        + counting_context_weighted
                    )
                else:
                    word_out_state = (
                        current_state
                        + word_weighted_embedding
                        + word_context_weighted
                        + counting_context_weighted
                    )

                word_prob = self.word_convert(word_out_state)  # [b, 256] --> [b, 111]
                word_probs[:, i] = word_prob  # 填上该步的预测的结果  word_probs[b, l', 111]
                word_alphas[:, i] = word_alpha
            return word_probs, word_alphas, word_alpha_sum
        else:
            word_embedding = self.embedding(
                torch.ones([batch_size]).long().to(device=self.device)
            )
            for i in range(num_steps):
                hidden = self.word_input_gru(word_embedding, hidden)
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(
                    cnn_features,
                    cnn_features_trans,
                    hidden,
                    word_alpha_sum,
                    images_mask,
                )
                hidden = self.word_output_gru(word_context_vec, hidden)
                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)

                if self.params["dropout"]:
                    word_out_state = self.dropout(
                        current_state
                        + word_weighted_embedding
                        + word_context_weighted
                        + counting_context_weighted
                    )
                else:
                    word_out_state = (
                        current_state
                        + word_weighted_embedding
                        + word_context_weighted
                        + counting_context_weighted
                    )

                word_prob = self.word_convert(word_out_state)
                _, word = word_prob.max(1)
                word_embedding = self.embedding(word)
                word_probs[:, i] = word_prob
                word_alphas[:, i] = word_alpha
            return word_probs, word_alphas, word_alpha_sum

    # (cnn_features, images_mask)， # [b, 684, h", w"]    [b, 1, h", w"]
    def init_hidden(self, features, feature_mask, linear):
        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(
            -1
        )  # [b, 684]
        # nn.Linear(self.out_channel, self.hidden_size)  # 684 -> 256  [b, 256]
        average = linear(average)
        return torch.tanh(average)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


# S-MSA
class MS_MSA(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)  # [b, h, w, c] -> [b, hw, c]
        q_inp = self.to_q(x)  # q_inp:[b, hw, c] -> [b, hw, heads*dim_head]
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        # step1: q, k, v:[b, hw, heads*dim_head] -> [b, heads, hw, dim_head]
        # rearrange中h=self.num_heads表示将h维度分为self.num_heads份(N个头)
        # d = (heads*dim_head) // self.num_heads = (heads*dim_head) // heads = dim_head
        # step2: 形状变换后的q_inp,k_inp,v_inp进行map操作，将它们分别映射到 q,k,v 变量中
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q_inp, k_inp, v_inp),
        )
        v = v
        # [b, heads, hw, dim_head] -> [b, heads, dim_head, hw]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1, p=2)  # L2 Normalization
        k = F.normalize(k, dim=-1, p=2)
        # [b, heads, dim_head, hw] @ [b, heads, hw, dim_head] -> [b, heads, dim_head, dim_head]
        attn = k @ q.transpose(-2, -1)  # A = K^T*Q
        # 注意力重新缩放，rescale形状为[heads, 1, 1]，最后两个维度都是1，在计算时会自动进行广播操作
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # [b, heads, dim_head, hw]
        # 先将注意力头的数量和空间位置的数量进行合并，然后再展平，方便全连接处理
        x = x.permute(
            0, 3, 1, 2
        )  # [b, heads, dim_head, hw] -> [b, hw, heads, dim_head]
        x = x.reshape(
            b, h * w, self.num_heads * self.dim_head
        )  # [b, hw, heads*dim_head]

        # [b, hw, heads*dim_head] -> [b, hw, dim]  -> [b, h, w, c]
        out_c = self.proj(x).view(b, h, w, c)
        # 对v_inp进行reshape然后计算位置编码
        # [b, hw, heads*dim_head] -> [b, h, w, c] -> [b, c, h, w] ->pos_emb-> [b, h, w, c]
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )
        out = out_c + out_p

        return out


# LayerNorm + fn(自定义传入函数)
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    # *args表示任意数量的非关键字参数，**kwargs表示任意数量的关键字参数
    # 可以接受任意数量的输入参数，然后将这些参数传递给块操作函数self.fn
    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)  # 将块操作函数的输出作为最终的前归一化模块的输出


# FFN
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        # conv1*1 -> GELU -> conv3*3 -> GELU -> conv1*1
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


# SAB
class MSAB(nn.module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)  # [b,c,h,w] -> [b,h,w,c]
        # x -> MS_MSA -> 残差连接 -> LayerNorm -> FFN -> 残差连接
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


# MST_Decoder
class MST(nn.module):
    def __init__(
        self, params, dim=128, stage=2, num_blocks=[2, 4, 4]
    ):  # num_blocks=[4, 7, 5]
        super().__init__()
        self.dim = dim
        self.stage = stage
        dim_stage = dim

        self.device = params["device"]
        self.word_num = params["word_num"]  # len(words)
        self.enc_out_channel = params["encoder"]["out_channel"]  # 684
        self.word_num = params["word_num"]  # len(words)

        self.word_embed = nn.Sequential(
            nn.Embedding(self.word_num, d_model), nn.LayerNorm(d_model)
        )
        self.pos_enc = WordPosEnc(d_model=d_model)

        # Input proj
        self.embedding = nn.Conv2d(
            self.enc_out_channel, self.dim, kernel_size=1, padding=0, bias=False
        )

        # Bottleneck layers
        self.blttleneck = MSAB(
            dim=dim_stage,
            dim_head=dim,
            heads=dim_stage // dim,
            num_blocks=num_blocks[-1],
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        # (deConv2*2 -> conv1*1 -> SAB) * 3
        for i in range(stage):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            dim_stage,
                            dim_stage // 2,
                            stride=2,
                            kernel_size=2,
                            padding=0,
                            output_padding=0,
                        ),
                        nn.Conv2d(dim_stage // 2, dim_stage // 2, 1, 1, bias=False),
                        # nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                        MSAB(
                            dim=dim_stage // 2,
                            num_blocks=num_blocks[stage - 1 - i],
                            dim_head=dim,
                            heads=(dim_stage // 2) // dim,
                        ),
                    ]
                )
            )
            dim_stage //= 2  # 每次上采样后，通道数减半

        # Output proj
        self.out_proj = nn.Linear(128, self.word_num)

        # Activation function

    def build_attention_mask(self, length):
        mask = torch.full((length, length), fill_value=1, dtype=torch.bool, device=self.device)
        mask.triu_(1)  # 上三角矩阵
        return mask

    def forward(
        self,
        cnn_features,
        labels,
        counting_preds,
        images_mask,
        labels_mask,
        is_train=True,
    ):
        """
        x: [b,c,h,w]
        return out:
        """
        _, length = labels.shape  # [b, l']
        tgt_mask = self.build_attention_mask(length)
        tgt_pad_mask = labels == 0  # 用于忽略填充词的影响，只计算非填充词位置的损失

        tgt = self.word_embed(tgt)  # [b, l, d_model]
        tgt = self.pos_enc(tgt)  # [b, l, d_model]
        tgt = tgt.permute(0, 2, 1)  # [b, d_model, l]

        # memory_key_padding_mask = images_mask

        src = self.embedding(src)
        src = self.blttleneck(src)
        print("out_encoder shape:", src.shape, "\n----------")

        # q:tgt, k:feature, v:feature
        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            src = FeaUpSample(src)
            src = Fution(src)
            src = LeWinBlcok(src, tgt, tgt_mask, tgt_pad_mask)
        print("out_decoder shape:", src.shape, "\n----------")

        # Mapping
        src = src.sum(-1).sum(-1)  # [b, c, h, w] -> [b, c]
        print("sum shape:", src.shape, "\n----------")
        out = self.out_proj(src)
