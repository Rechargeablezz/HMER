import torch
import time
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, RandomSampler


class HMERDataset(Dataset):
    def __init__(self, params, image_path, label_path, words, is_train=True, use_aug=False):
        super(HMERDataset, self).__init__()
        if image_path.endswith('.pkl'):
            with open(image_path, 'rb') as f:
                self.images = pkl.load(f)   # 加载所有图像(字典)  'TrainData2_9_sub_9': array([[0, 0, 0, ..., 0, 0, 0],......[0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
        elif image_path.endswith('.list'):
            with open(image_path, 'r') as f:
                lines = f.readlines()
            self.images = {}
            print(f'data files: {lines}')
            for line in lines:
                name = line.strip()
                print(f'loading data file: {name}')
                start = time.time()
                with open(name, 'rb') as f:
                    images = pkl.load(f)
                self.images.update(images)
                print(f'loading {name} cost: {time.time() - start:.2f} seconds!')

        with open(label_path, 'r') as f:
            self.labels = f.readlines()  # 'name    latex'

        self.words = words   # words = Words(params['word_path']) 根据该地址文件建立字典（对latex公式编解码）
        self.is_train = is_train
        self.params = params

    def __len__(self):
        assert len(self.images) == len(self.labels)  # 确保公式数量和label数量相同
        return len(self.labels)

    def __getitem__(self, idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        image = self.images[name]
        image = torch.Tensor(255 - image) / 255   # CROHME  8K

        image = image.unsqueeze(0)
        labels.append('eos')
        words = self.words.encode(labels)    # 编码labels获得words

        words = torch.LongTensor(words)
        return image, words  # 返回处理后的图像和编码后的label


def get_crohme_dataset(params, use_aug):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    # print(params.keys())
    print(f"训练数据路径 images: {params['train_image_path']} labels: {params['train_label_path']}")
    print(f"验证数据路径 images: {params['eval_image_path']} labels: {params['eval_label_path']}")

    train_dataset = HMERDataset(params, params['train_image_path'], params['train_label_path'], words, is_train=True)
    eval_dataset = HMERDataset(params, params['eval_image_path'], params['eval_label_path'], words, is_train=False)

    train_sampler = RandomSampler(train_dataset)  # 随机采样
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=False)
    eval_loader = DataLoader(eval_dataset, batch_size=params['batch_size'], sampler=eval_sampler,
                             num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=False)

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
    return train_loader, eval_loader


def collate_fn(batch_images):
    max_width, max_height, max_length = 0, 0, 0
    _, channel = len(batch_images), batch_images[0][0].shape[0]   # batch_images[b, item]    # batch, channel =
    proper_items = []
    for item in batch_images:  # item[image, label]  image[c, h, w]
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:  # 当一个batch开始时若第一个图很大可能会使内存暴涨
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)  # 过滤后的[image, label]对

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))  # [bch'w']、[b1h'w']
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))  # [b, l']

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape  # 该batch第i个图片形状，channel数不需要用到
        images[i][:, :h, :w] = proper_items[i][0]  # 图片贴到对应的全零img[h' * w']中，左上角开始
        image_masks[i][:, :h, :w] = 1  # 有图片的部分对应mask为1
        length = proper_items[i][1].shape[0]  # 该batch的label的长度
        labels[i][:length] = proper_items[i][1]  # label贴到对应的全零label[b, l']中        [11,12,13,14,0,  0,0,0,0]
        labels_masks[i][:length] = 1  # 有label的部分mask为1
    return images, image_masks, labels, labels_masks


class Words:
    # 建立字典
    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_dict = {words[i].strip(): i for i in range(len(words))}   # latex符号 ==> 数字
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}  # 数字 ==> latex符号

    def __len__(self):
        return len(self.words_dict)

    # latex序列编码为对应数字序列label_index
    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    # 数字序列label解码成对应latex公式，每个latex符号间空格隔开
    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label


collate_fn_dict = {
    'collate_fn': collate_fn
}
