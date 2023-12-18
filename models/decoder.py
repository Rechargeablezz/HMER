import torch
import torch.nn as nn
from models.attention import Attention
import math
# import numpy as np
# from counting_utils import gen_counting_label


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):  # position_embedding = PositionEmbeddingSine(256, normalize=True)
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):        # pos = position_embedding(cnn_features_trans, images_mask[:,0,:,:])
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
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # []
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [b, h", w", c] --> [b, c, h", w"]     channel = 256 + 256 --> 512
        return pos


class AttDecoder(nn.Module):
    def __init__(self, params):
        super(AttDecoder, self).__init__()
        self.params = params
        self.input_size = params['decoder']['input_size']  # 256
        self.hidden_size = params['decoder']['hidden_size']  # 256
        self.out_channel = params['encoder']['out_channel']  # 684
        self.attention_dim = params['attention']['attention_dim']  # 512
        self.dropout_prob = params['dropout']  # True
        self.device = params['device']
        self.word_num = params['word_num']  # len(words)
        self.counting_num = params['counting_decoder']['out_channel']  # 111

        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']  # 16

        # init hidden state
        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)  # 684 -> 256
        self.init_weight2 = nn.Linear(self.out_channel, self.hidden_size)  # 684 -> 256

        # word embedding
        self.embedding = nn.Embedding(self.word_num, self.input_size)  # len(words) -> 256
        self.embedding2 = nn.Embedding(self.word_num, self.input_size)  # len(words) -> 256

        # word gru
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)  # 256 -> 256
        self.word_input_gru2 = nn.GRUCell(self.input_size, self.hidden_size)  # 256 -> 256
        self.word_output_gru = nn.GRUCell(self.out_channel, self.hidden_size)  # 256 -> 256
        self.word_output_gru2 = nn.GRUCell(self.out_channel, self.hidden_size)  # 256 -> 256

        # attention
        self.word_attention = Attention(params)
        self.word_attention2 = Attention(params)
        self.encoder_feature_conv = nn.Conv2d(self.out_channel, self.attention_dim,     # 684 -> 512
                                              kernel_size=params['attention']['word_conv_kernel'],  # 1
                                              padding=params['attention']['word_conv_kernel'] // 2)

        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size)
        self.word_state_weight2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.word_embedding_weight = nn.Linear(self.input_size, self.hidden_size)
        self.word_embedding_weight2 = nn.Linear(self.input_size, self.hidden_size)

        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.word_context_weight2 = nn.Linear(self.out_channel, self.hidden_size)

        self.counting_context_weight = nn.Linear(self.counting_num, self.hidden_size)

        self.word_convert = nn.Linear(self.hidden_size, self.word_num)
        self.word_convert2 = nn.Linear(self.hidden_size, self.word_num)

        self.relu = nn.ReLU()
        if params['dropout']:
            self.dropout = nn.Dropout(params['dropout_ratio'])

    def forward(self, cnn_features, labels, labels2, counting_preds, images_mask, labels_mask, labels_mask2, is_train=True):
        batch_size, num_steps = labels.shape  # [b, l']
        height, width = cnn_features.shape[2:]  # [h", w"]
        word_probs = torch.zeros((batch_size, num_steps, self.word_num)).to(device=self.device)  # [b, l', len(words) = 111]  初始化为全0
        word_probs2 = torch.ones((batch_size, num_steps, self.word_num)).to(device=self.device)

        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio]  # [b, c, h", w"]  c = 1

        word_alpha_sum = torch.zeros((batch_size, 1, height, width)).to(device=self.device)  # [b, 1, h", w"]
        word_alpha_sum2 = torch.zeros((batch_size, 1, height, width)).to(device=self.device)

        word_alphas = torch.zeros((batch_size, num_steps, height, width)).to(device=self.device)  # [b, l', h", w"]
        word_alphas2 = torch.zeros((batch_size, num_steps, height, width)).to(device=self.device)

        hidden = self.init_hidden(cnn_features, images_mask, self.init_weight)   # [b, 256]  初始化hidden
        hidden2 = self.init_hidden(cnn_features, images_mask, self.init_weight2)

        counting_context_weighted = self.counting_context_weight(counting_preds)        # C：[b, 111]  ==>  [b, 256]
        counting_context2_weighted = self.counting_context_weight(counting_preds)

        cnn_features_trans = self.encoder_feature_conv(cnn_features)     # [b, 684, h", w"] ==> [b, 512, h", w"]    T
        position_embedding = PositionEmbeddingSine(256, normalize=True)     # 512/2
        pos = position_embedding(cnn_features_trans, images_mask[:, 0, :, :])  # pos_embedding   P
        cnn_features_trans = cnn_features_trans + pos   # T'

        if is_train:
            for i in range(num_steps):  # num_steps = l'
                word_embedding = self.embedding(labels[:, i - 1]) if i else self.embedding(torch.ones([batch_size]).long().to(self.device))
                word_embedding2 = self.embedding2(labels2[:, i - 1]) if i else self.embedding2(torch.zeros([batch_size]).long().to(self.device))
                # word_embedding2 = self.embedding2(labels2[:, i]) if i < num_steps - 1 else self.embedding(torch.zeros([batch_size]).long().to(self.device))
                hidden = self.word_input_gru(word_embedding, hidden)  # ht    yt-1 --> embedding --> GRU --> ht
                hidden2 = self.word_input_gru2(word_embedding2, hidden2)

                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden,
                                                                                   word_alpha_sum, images_mask)
                word_context_vec2, word_alpha2, word_alpha_sum2 = self.word_attention2(cnn_features, cnn_features_trans, hidden2,
                                                                                       word_alpha_sum2, images_mask)

                hidden = self.word_output_gru(word_context_vec, hidden)
                hidden2 = self.word_output_gru2(word_context_vec2, hidden2)

                current_state = self.word_state_weight(hidden)                         # ht： current/hidden
                current_state2 = self.word_state_weight2(hidden2)

                word_weighted_embedding = self.word_embedding_weight(word_embedding)   # E(yt-1) word embeding -> 256
                word_weighted_embedding2 = self.word_embedding_weight2(word_embedding2)

                word_context_weighted = self.word_context_weight(word_context_vec)     # V: context vec -> 256
                word_context_weighted2 = self.word_context_weight2(word_context_vec2)

                if self.params['dropout']:
                    word_out_state = self.dropout(current_state + word_weighted_embedding +
                                                  word_context_weighted + counting_context_weighted)
                    word_out_state2 = self.dropout(current_state2 + word_weighted_embedding2 +
                                                   word_context_weighted2 + counting_context2_weighted)
                else:
                    word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted
                    word_out_state2 = current_state2 + word_weighted_embedding2 + word_context_weighted2 + counting_context2_weighted

                word_prob = self.word_convert(word_out_state)   # [b, 256] --> [b, 111]
                word_prob2 = self.word_convert2(word_out_state2)

                word_probs[:, i] = word_prob  # 填上该步的预测的结果  word_probs[b, l', 111]
                word_probs2[:, i] = word_prob2

                word_alphas[:, i] = word_alpha
                word_alphas2[:, i] = word_alpha2

                # probs = torch.softmax(word_probs, dim=-1)
                # probs_sum = torch.sum(probs, dim=1)
                # counting_preds = self.relu(counting_preds - probs_sum)
                # counting_context_weighted = self.counting_context_weight(counting_preds)

            return word_probs, word_alphas, word_alpha_sum, word_probs2, word_alphas2, word_alpha_sum2
        else:
            word_embedding = self.embedding(torch.ones([batch_size]).long().to(device=self.device))
            for i in range(num_steps):
                hidden = self.word_input_gru(word_embedding, hidden)
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden,
                                                                                   word_alpha_sum, images_mask)
                hidden = self.word_output_gru(word_context_vec, hidden)
                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)

                if self.params['dropout']:
                    word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted)
                else:
                    word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted

                word_prob = self.word_convert(word_out_state)
                _, word = word_prob.max(1)
                word_embedding = self.embedding(word)
                word_probs[:, i] = word_prob
                word_alphas[:, i] = word_alpha

                # probs = torch.softmax(word_probs, dim=-1)
                # probs_sum = torch.sum(probs, dim=1)
                # counting_preds = self.relu(counting_preds - probs_sum)
                # counting_context_weighted = self.counting_context_weight(counting_preds)

            return word_probs, word_alphas, word_alpha_sum, None, None, None

    def init_hidden(self, features, feature_mask, linear):  # (cnn_features, images_mask)， # [b, 684, h", w"]    [b, 1, h", w"]
        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)  # [b, 684]
        average = linear(average)   # nn.Linear(self.out_channel, self.hidden_size)  # 684 -> 256  [b, 256]
        return torch.tanh(average)
