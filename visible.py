import cv2
import pickle as pkl
from tqdm import tqdm

# with open('datasets/CROHME/train_images.pkl', 'rb') as f:
#     images = pkl.load(f)

# with open('datasets/CROHME/train_labels.txt') as f:
#     lines = f.readlines()

# for line in tqdm(lines):
#     name, *labels = line.split()
#     name = name.split('.')[0] if name.endswith('jpg') else name
#     input_labels = labels
#     labels = ' '.join(labels)  # split后没空格了，需要补上
#     img = images[name]

#     save_path = "train_img/" + name + ".jpg"
#     cv2.imwrite(save_path, img)


with open('datasets/CROHME/train_images.pkl', 'rb') as f:
    images = pkl.load(f)

with open('datasets/CROHME/train_labels.txt') as f:
    lines = f.readlines()

for line in tqdm(lines):
    name, *labels = line.split()
    name = name.split('.')[0] if name.endswith('jpg') else name
    input_labels = labels
    labels = ' '.join(labels)  # split后没空格了，需要补上
    img = images[name]

    save_path = "train_img/" + name + ".jpg"
    cv2.imwrite(save_path, img)
