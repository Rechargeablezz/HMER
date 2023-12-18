import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# import os


def gen_counting_label(labels, channel, tag):      # tag = True   labels[b, l']   channel: num of dic
    b, t = labels.size()
    device = labels.device
    counting_labels = torch.zeros((b, channel))
    if tag:
        ignore = [0, 1, 107, 108, 109, 110]   # eos  sos  ^  _  {  }
    else:
        ignore = []
    for i in range(b):
        for j in range(t):
            k = labels[i][j]
            if k in ignore:   # 检查该数字对应的原始label是否需要忽略
                continue
            else:
                counting_labels[i][k] += 1   # 类似于公式中每个符号的计数器，碰到一个相应数字，count+1
    return counting_labels.to(device)
