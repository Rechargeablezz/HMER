import os
import time
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
# from torch.cuda.amp import GradScaler
from utils import load_config, save_checkpoint, load_checkpoint
from dataset import get_crohme_dataset
from models.can import CAN
from training import train, eval

parser = argparse.ArgumentParser(description='model training')  # 创建一个 ArgumentParser 对象，包含将命令行解析成 Python 数据类型所需的全部信息
# 添加参数：一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的
parser.add_argument('--dataset', default='CROHME', type=str, help='数据集名称')
parser.add_argument('--check', action='store_true', help='测试代码选项')  # 若触发--check，则为true
# 解析参数：ArgumentParser 通过 parse_args() 方法解析参数
args = parser.parse_args()

if not args.dataset:
    print('请提供数据集名称')
    exit(-1)

if args.dataset == 'CROHME':
    config_file = 'config.yaml'
elif args.dataset == 'CROHME2':
    config_file = 'config_2025.yaml'
"""加载config文件"""
params = load_config(config_file)   # params来自config.yaml

"""设置随机种子"""
random.seed(params['seed'])
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])
torch.cuda.manual_seed(params['seed'])

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 选择要使用的设备
params['device'] = device

if args.dataset == 'CROHME' or args.dataset == 'CROHME2':
    # train_loader, eval_loader = get_crohme_dataset(params, use_aug=use_aug)
    train_loader, eval_loader = get_crohme_dataset(params, use_aug=False)

model = CAN(params)
now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())   # 返回以可读字符串表示的当地时间
model.name = f'{params["experiment"]}_{now}_decoder-{params["decoder"]["net"]}'   # CAN_2022-xx-xx-xx-xx_decoder-AttDecoder

print(model.name)
model = model.to(device)

if args.check:
    writer = None
else:  # 不触发--check时，check默认为False
    writer = SummaryWriter(f'{params["log_dir"]}/{model.name}')

optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                      eps=float(params['eps']), weight_decay=float(params['weight_decay']))

# optimizer = nn.DataParallel(optimizer)
if params['finetune']:
    print('加载预训练模型权重')
    print(f'预训练权重路径: {params["checkpoint"]}')
    load_checkpoint(model, optimizer, params['checkpoint'])

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
if not args.check:
    if not os.path.exists(os.path.join(params['checkpoint_dir'], model.module.name)):  # checkpoint_dir\model.name不存在就创建目录
        os.makedirs(os.path.join(params['checkpoint_dir'], model.module.name), exist_ok=True)
    os.system(f'cp {config_file} {os.path.join(params["checkpoint_dir"], model.module.name, model.module.name)}.yaml')  # 对应.yaml文件

"""在CROHME上训练"""
if args.dataset == 'CROHME' or args.dataset == 'CROHME2':
    min_score, init_epoch = 0, 0

    for epoch in range(init_epoch, params['epochs']):
        train_loss, train_word_score, train_exprate = train(params, model, optimizer, epoch, train_loader, writer=writer)   # train_loss, train_word_score, train_exprate

        if epoch >= params['valid_start']:  # valid_start: 0
            eval_loss, eval_word_score, eval_exprate = eval(params, model.module, epoch, eval_loader, writer=writer)  # eval_loss, eval_word_score, eval_exprate
            print(f'Epoch: {epoch+1} loss: {eval_loss:.4f} word score: {eval_word_score:.4f} ExpRate: {eval_exprate:.4f}')
            # 50 epoch 存一次
            # if(epoch % 50 == 0) : save_checkpoint(model.module, optimizer, eval_word_score, eval_exprate, epoch+1,
            #                                 optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])
            if eval_exprate > min_score and not args.check and epoch >= params['save_start']:  # 存档条件
                min_score = eval_exprate
                save_checkpoint(model.module, optimizer, eval_word_score, eval_exprate, epoch + 1,
                                optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])
        # gc.collect()
        # torch.cuda.empty_cache()
