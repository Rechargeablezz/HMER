import os
import cv2
import yaml
import math
import torch
import numpy as np
from difflib import SequenceMatcher


def load_config(yaml_path):  # 加载config.yaml
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except Exception:
        print('尝试UTF-8编码....')
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    if not params['experiment']:
        print('实验名不能为空!')
        exit(-1)
    if not params['train_image_path']:
        print('训练图片路径不能为空！')
        exit(-1)
    if not params['train_label_path']:
        print('训练label路径不能为空！')
        exit(-1)
    if not params['word_path']:
        print('word dict路径不能为空！')
        exit(-1)
    if 'train_parts' not in params:
        params['train_parts'] = 1
    if 'valid_parts' not in params:
        params['valid_parts'] = 1
    if 'valid_start' not in params:
        params['valid_start'] = 0
    if 'word_conv_kernel' not in params['attention']:
        params['attention']['word_conv_kernel'] = 1
    return params

    # if not 'lr_decay' in params or params['lr_decay'] == 'cosine'，
    # update_lr(optimizer, epoch, batch_idx, len(train_loader), params['epochs'], params['lr'])


def update_lr(optimizer, current_epoch, current_step, steps, epochs, initial_lr):
    if current_epoch < 1:
        new_lr = initial_lr / steps * (current_step + 1)
    # elif 1 <= current_epoch <= 200:
    #     new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (200 * steps))) * initial_lr
    else:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (epochs * steps))) * initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def save_checkpoint(model, optimizer, word_score, ExpRate_score, epoch, optimizer_save=False, path='checkpoints', multi_gpu=False, local_rank=0):
    filename = f'{os.path.join(path, model.name)}/{model.name}_WordRate-{word_score:.4f}_ExpRate-{ExpRate_score:.4f}_{epoch}.pth'
    if optimizer_save:
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    else:
        state = {
            'model': model.state_dict()  # 返回当前module所有状态数据并保存
        }
    torch.save(state, filename)
    print(f'Save checkpoint: {filename}\n')
    return filename


def load_checkpoint(model, optimizer, path):
    # 保存模型的state_dict()，只是保存模型的参数
    # 加载时需要先创建一个模型的实例model，之后通过torch.load()将保存的模型参数加载进来，得到dict，再通过model.load_state_dict(dict)将模型的参数更新
    # map_location参数是用于重定向，比如此前模型使用cuda:0训练并存储，但是想加载到cpu：map_location='cpu'
    # 在不同cuda之间的转换：map_location={'cuda:0':'cuda:1'}
    state = torch.load(path, map_location='cpu')
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    else:
        print('No optimizer in the pretrained model')
    model.load_state_dict(state['model'])


class Meter:  # loss_meter = Meter()
    def __init__(self, alpha=0.9):
        self.nums = []
        self.exp_mean = 0
        self.alpha = alpha

    @property
    def mean(self):
        return np.mean(self.nums)  # 求取平均值，必须对数组操作

    def add(self, num):
        if len(self.nums) == 0:
            self.exp_mean = num
        self.nums.append(num)
        self.exp_mean = self.alpha * self.exp_mean + (1 - self.alpha) * num


def cal_score(word_probs, word_label, mask):
    line_right = 0
    if word_probs is not None:
        _, word_pred = word_probs.max(2)  # 第三个维度求最大值
    # SequenceMatcher(None, s1', s2', autojunk=False).ratio()  相似性度量     (label和pred间)
    # ((len(s1') + len(s2')) / (len(s1') / 2
    # .detach() 阻断反向传播，返回值：Tensor，且经过detach()方法后，变量仍然在GPU上
    # .cpu() 移至cpu 返回值是cpu上的Tensor
    # .numpy()  返回值为numpy.array()
    # zip()列表配对生成字典[(label, pred, mask), (), ...]
    word_scores = [SequenceMatcher(None, s1[:int(np.sum(s3))], s2[:int(np.sum(s3))], autojunk=False).ratio() *
                   (len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) / len(s1[: int(np.sum(s3))]) / 2
                   for s1, s2, s3 in zip(word_label.cpu().detach().numpy(), word_pred.cpu().detach().numpy(), mask.cpu().detach().numpy())]

    batch_size = len(word_scores)
    for i in range(batch_size):
        if word_scores[i] == 1:
            line_right += 1

    ExpRate = line_right / batch_size
    word_scores = np.mean(word_scores)
    return word_scores, ExpRate


def draw_attention_map(image, attention):
    h, w = image.shape
    attention = cv2.resize(attention, (w, h))
    # uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255。如果原数据是大于255的，那么在直接使用np.uint8()后，比第八位更大的数据都被截断了
    attention_heatmap = ((attention - np.min(attention)) / (np.max(attention) - np.min(attention))*255).astype(np.uint8)
    attention_heatmap = cv2.applyColorMap(attention_heatmap, cv2.COLORMAP_JET)  # 应用色度图来伪彩色化（这里是JET）
    image_new = np.stack((image, image, image), axis=-1).astype(np.uint8)  # numpy.stack(arrays, axis=0)，axis参数指定新轴在结果尺寸中的索引
    attention_map = cv2.addWeighted(attention_heatmap, 0.4, image_new, 0.6, 0.)  # 加权和 cv2.addWeighted(img1, alpha, img2, beta, gamma)，gamma对图片做一个亮度调整
    return attention_map


def draw_counting_map(image, counting_attention):
    h, w = image.shape
    counting_attention = torch.clamp(counting_attention, 0.0, 1.0).numpy()  # .clamp()将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量
    counting_attention = cv2.resize(counting_attention, (w, h))
    counting_attention_heatmap = (counting_attention * 255).astype(np.uint8)
    counting_attention_heatmap = cv2.applyColorMap(counting_attention_heatmap, cv2.COLORMAP_JET)
    image_new = np.stack((image, image, image), axis=-1).astype(np.uint8)
    counting_map = cv2.addWeighted(counting_attention_heatmap, 0.4, image_new, 0.6, 0.)
    return counting_map


# 计算编辑距离
def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n  # 垂直

    # 创建初始矩阵
    dp = [[0]*(n+1) for _ in range(m+1)]   # m+1行n+1列
    for i in range(m+1):
        dp[i][0] = i  # 第一列赋值
    for j in range(n+1):
        dp[0][j] = j  # 第一行赋值

    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]


def compute_edit_distance(prediction, label):
    prediction = prediction.strip().split(' ')
    label = label.strip().split(' ')
    distance = cal_distance(prediction, label)
    return distance
