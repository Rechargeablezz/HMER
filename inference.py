import os
import argparse
import torch
import json
import time
import pickle as pkl
from tqdm import tqdm
from utils import load_config, load_checkpoint, compute_edit_distance
from models.infer_model import Inference
from dataset import Words
# from counting_utils import gen_counting_label

parser = argparse.ArgumentParser(description='model testing')
parser.add_argument('--dataset', default='CROHME', type=str, help='数据集名称')
parser.add_argument('--image_path', default='/dataset/14_test_images.pkl', type=str, help='测试image路径')
parser.add_argument('--label_path', default='/dataset/14_test_labels.txt', type=str, help='测试label路径')
# parser.add_argument('--image_path', default='../syntactic_HME_generation/val_2014_pic.pkl', type=str, help='测试image路径')
# parser.add_argument('--label_path', default='../syntactic_HME_generation/14_test_labels.txt', type=str, help='测试label路径')
parser.add_argument('--word_path', default='/dataset/words_dict.txt', type=str, help='测试dict路径')

parser.add_argument('--draw_map', default=False)
args = parser.parse_args()

if not args.dataset:
    print('请提供数据集名称')
    exit(-1)

if args.dataset == 'CROHME':
    config_file = 'config.yaml'

"""加载config文件"""
params = load_config(config_file)

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device)
params['device'] = device
words = Words(args.word_path)
params['word_num'] = len(words)

if 'use_label_mask' not in params:
    params['use_label_mask'] = False
print(params['decoder']['net'])
model = Inference(params, draw_map=args.draw_map)
model = model.to(device)

load_checkpoint(model, None, params['checkpoint'])

# 针对model 在训练时和评价时不同的 Batch Normalization 和 Dropout 方法模式
# model.eval()  不启用BatchNormalization和Dropout，自动把BN和Dropout固定住，不会取平均，而是用训练好的值
# 不然的话，一旦test的batch_size过小，很容易就会被BN层影响结果
model.eval()

with open(args.image_path, 'rb') as f:
    images = pkl.load(f)

with open(args.label_path) as f:
    lines = f.readlines()

line_right = 0
e1, e2, e3 = 0, 0, 0
bad_case = {}
model_time = 0
mae_sum, mse_sum = 0, 0

with torch.no_grad():  # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False
    for line in tqdm(lines):
        name, *labels = line.split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        input_labels = labels
        labels = ' '.join(labels)  # split后没空格了，需要补上
        img = images[name]

        # save_path = "vis/original/" + name + ".jpg"
        # cv2.imwrite(save_path, img)

        img = torch.Tensor(255-img) / 255  #
        # img = torch.Tensor(img) / 255  #
        img = img.unsqueeze(0).unsqueeze(0)
        img = img.to(device)
        a = time.time()

        input_labels = words.encode(input_labels)   # return label_index = [self.words_dict[item] for item in labels]
        input_labels = torch.LongTensor(input_labels)

        input_labels = input_labels.unsqueeze(0).to(device)

        probs, _, mae, mse = model(img, input_labels, os.path.join(params['decoder']['net'], name))  # os.path.join()：用于路径拼接文件路径，可以传入多个路径
        mae_sum += mae  # 平均绝对值误差，也可以看做L1损失  nn.L1Loss(reduction='mean')
        mse_sum += mse  # 均方误差，可以看做是一种L2损失    nn.MSELoss(reduction='mean')
        model_time += (time.time() - a)  # 计算一次时间

        prediction = words.decode(probs)  # return label= ' '.join([self.words_index_dict[int(item)] for item in label_index])

        with open("bad_case.txt", 'a') as f:
            if prediction == labels:  # 推理正确
                line_right += 1
            else:
                bad_case[name] = {
                    'label': labels,
                    'predi': prediction
                }
                print(name, prediction, '|| ' + labels)  # 输出推理错误的(name, prediction, labels)
                str1 = name + '  ' + prediction + '  ||  ' + labels
                f.write(str1)
                f.write('\n')
        distance = compute_edit_distance(prediction, labels)  # 计算编辑距离（衡量两个字符串差异化程度）
        if distance <= 1:  # 一个符号级别错误
            e1 += 1
        if distance <= 2:
            e2 += 1
        if distance <= 3:
            e3 += 1

print(f'model time: {model_time}')
print(f'ExpRate: {line_right / len(lines)}')
print(f'mae: {mae_sum / len(lines)}')
print(f'mse: {mse_sum / len(lines)}')
print(f'e1: {e1 / len(lines)}')
print(f'e2: {e2 / len(lines)}')
print(f'e3: {e3 / len(lines)}')

with open(f'{params["decoder"]["net"]}_bad_case.json', 'w') as f:
    json.dump(bad_case, f, ensure_ascii=False)
