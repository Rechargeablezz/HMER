import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.params = params
        self.hidden = params['decoder']['hidden_size']  # 256
        self.attention_dim = params['attention']['attention_dim']  # 512
        self.hidden_weight = nn.Linear(self.hidden, self.attention_dim)

        self.count_weight = nn.Linear(self.hidden, self.attention_dim)

        self.attention_conv = nn.Conv2d(1, 512, kernel_size=11, padding=5, bias=False)
        # self.attention_conv2 = nn.Conv2d(1, 512, kernel_size=5, padding=2, bias=False)
        self.attention_weight = nn.Linear(512, self.attention_dim, bias=False)
        # self.attention_weight2 = nn.Linear(512, self.attention_dim, bias=False)

        self.alpha_convert = nn.Linear(self.attention_dim, 1)

    def forward(self, cnn_features, cnn_features_trans, hidden, alpha_sum, counting_dyna, image_mask=None):  # F、T'(pos_T)、ht、A(传入时全0)
        query = self.hidden_weight(hidden)  # [b, 256,  ==> [b, 512,
        counting_dyna = self.count_weight(counting_dyna)  # [b, 256]  ==> [b, 512]
        alpha_sum_trans = self.attention_conv(alpha_sum)  # [b, 1, h", w"] ==> [b, 512, h", w"]
        # alpha_sum_trans2 = self.attention_conv2(alpha_sum)  # [b, 1, h", w"] ==> [b, 512, h", w"]

        coverage_alpha = self.attention_weight(alpha_sum_trans.permute(0, 2, 3, 1))  # 维度交换[b, h", w", 512]
        # coverage_alpha2 = self.attention_weight2(alpha_sum_trans2.permute(0, 2, 3, 1))  # 维度交换[b, h", w", 512]
        # alpha_score = torch.tanh(query[:, None, None, :] + coverage_alpha + coverage_alpha2 + cnn_features_trans.permute(0, 2, 3, 1))  # tanh(ht + A + T)
        alpha_score = torch.tanh(query[:, None, None, :] + coverage_alpha + cnn_features_trans.permute(0, 2, 3, 1) + counting_dyna[:, None, None, :])  # tanh(ht + A + T)

        energy = self.alpha_convert(alpha_score)    # bhw1
        energy = energy - energy.max()  # 归一化处理,避免指数运算可能导致的数值溢出
        energy_exp = torch.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:, None, None] + 1e-10)
        alpha_sum = alpha[:, None, :, :] + alpha_sum
        context_vector = (alpha[:, None, :, :] * cnn_features).sum(-1).sum(-1)
        return context_vector, alpha, alpha_sum
