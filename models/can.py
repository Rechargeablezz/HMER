import torch
import models

import torch.nn as nn
import torch.nn.functional as F

from models.densenet import DenseNet
from models.counting import CountingDecoder as counting_decoder
from counting_utils import gen_counting_label


class CAN(nn.Module):
    def __init__(self, params=None):
        super(CAN, self).__init__()
        self.params = params
        self.use_label_mask = params['use_label_mask']   # 是否使用label mask，默认为False
        self.encoder = DenseNet(params=self.params)   # Dense做主干网络
        self.in_channel = params['counting_decoder']['in_channel']             # counting_decoder:  in_channel: 684   out_channel: 111
        self.out_channel = params['counting_decoder']['out_channel']
        self.counting_decoder1 = counting_decoder(self.in_channel, self.out_channel, 3)   # counting decoder有两路
        self.counting_decoder2 = counting_decoder(self.in_channel, self.out_channel, 5)
        self.decoder = getattr(models, params['decoder']['net'])(params=self.params)   # AttDecoder
        self.cross = nn.CrossEntropyLoss(reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()  # 使用label mask则reduction为None，否则为mean
        self.counting_loss = nn.SmoothL1Loss(reduction='mean')
        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']     # ratio: 16

    def forward(self, images, images_mask, labels, labels_mask, is_train=True):
        cnn_features = self.encoder(images)
        print('cnn_features.shape: ', cnn_features.shape)
        counting_mask = images_mask[:, :, ::self.ratio, ::self.ratio]     # [b, c, h, w]   按比例缩小, c=684
        counting_labels = gen_counting_label(labels, self.out_channel, True)
        counting_preds1, _ = self.counting_decoder1(cnn_features, counting_mask)  # counting_preds1, counting_maps1
        counting_preds2, _ = self.counting_decoder2(cnn_features, counting_mask)  # counting_preds2, counting_maps2
        counting_preds = (counting_preds1 + counting_preds2) / 2
        print('counting_preds.shape: ', counting_preds.shape)
        counting_loss = self.counting_loss(counting_preds1, counting_labels) + self.counting_loss(counting_preds2, counting_labels) + \
            self.counting_loss(counting_preds, counting_labels)  # 三项和
        # cnn_feature [b, 684, h, w]
        word_probs, word_alphas, word_alpha_sum = self.decoder(cnn_features, labels, counting_preds, images_mask, labels_mask, is_train=is_train)
        word_loss = self.cross(word_probs.contiguous().view(-1, word_probs.shape[-1]), labels.view(-1))
        word_average_loss = (word_loss * labels_mask.view(-1)).sum() / (labels_mask.sum() + 1e-10) if self.use_label_mask else word_loss

        return word_probs, counting_preds, word_average_loss, counting_loss
