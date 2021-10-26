import os
from PIL import Image
from torch.utils.data import Dataset
import json

class ImageNet(Dataset):
    def __init__(self, path, transform, train=True, num_class=100):
        super(ImageNet, self).__init__()
        self.train = train
        if num_class == 100:
            with open("/mnt/lustre/zhangshaofeng/workspace/whitening_224/imagenet100.txt") as f:
                self.classes = f.readlines()
                self.classes = [name.replace('\n', '') for name in self.classes]
        else:
            self.classes = os.listdir(path)
        self.path = path
        self.transform = transform
        self.ids = []
        self.label = []
        for category in self.classes:
            if category[0] == 'n':
                # get img path
                self.ids += [path+category+"/"+name for name in os.listdir(path+category)]

    def __getitem__(self, item):
        if self.train == True:
            img_PIL = Image.open(self.ids[item]).convert('RGB')
            img1 = self.transform.transform(img_PIL)
            img2 = self.transform.transform_prime(img_PIL)
            label = self.classes.index(self.ids[item].split('/')[-2])
            return img1, img2, label
        else:
            img_PIL = Image.open(self.ids[item]).convert('RGB')
            img1 = self.transform(img_PIL)
            label = self.classes.index(self.ids[item].split('/')[-2])
            return img1, label

    def __len__(self):
        return len(self.ids)

class ImageNetVal(Dataset):
    def __init__(self, path, transform, idx2idx):
        super(ImageNetVal, self).__init__()
        self.idx2idx = idx2idx
        self.path = path
        self.transform = transform
        self._get_img_id()

    def __getitem__(self, item):
        img_PIL = Image.open(self.path + self.x_path[item]).convert('RGB')
        img1 = self.transform(img_PIL)
        label = self.label[item]
        return img1, label

    def __len__(self):
        return len(self.label)

    def _get_img_id(self):
        self.x_path = []
        self.label = []
        with open("/mnt/lustre/share/images/meta/val.txt") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line == "":
                    continue
                line = f.readline()
                name = line.split(" ")[0]
                cls_id = int(line.split(' ')[1])
                if cls_id in self.idx2idx:
                    self.x_path.append(name)
                    self.label.append(int(self.idx2idx[cls_id]))
        return


class ImageNetTrain(Dataset):
    def __init__(self, path, transform, num_class=100):
        super(ImageNetTrain, self).__init__()
        if num_class == 100:
            with open("/mnt/lustre/zhangshaofeng/workspace/whitening_224/imagenet100.txt") as f:
                self.classes = f.readlines()
                self.classes = [name.replace('\n', '') for name in self.classes]
        else:
            self.classes = os.listdir(path)
            self.classes = [i for i in self.classes if i[0] == 'n']
        # with open("/mnt/lustre/zhangshaofeng/workspace/barlowtwins/name2idx.json", "r") as f:
        #     self.name2idx = json.load(f)
        self.name2idx = name2idx_transform()
        self.idx2idx = {}
        for i in range(len(self.classes)):
            self.idx2idx[self.name2idx[self.classes[i]]] = i
        self.path = path
        self.transform = transform
        self.ids = []
        self.label = []
        for category in self.classes:
            if category[0] == 'n':
                # get img path
                self.ids += [path+category+"/"+name for name in os.listdir(path+category)]
                self.label += ([self.idx2idx[self.name2idx[category]]] * len(os.listdir(path+category)))

    def __getitem__(self, item):
        img_PIL = Image.open(self.ids[item]).convert('RGB')
        img1 = self.transform(img_PIL)
        label = int(self.label[item])
        return img1, label

    def __len__(self):
        return len(self.ids)

def name2idx_transform():
    name2idx = {}
    f = open("/mnt/lustre/share/images/meta/train.txt")
    line = f.readline()
    while line:
        name = line.split('/')[0]
        idx = int(line.split(' ')[1].replace('n', ''))
        name2idx[name] = idx
        line = f.readline()
    return name2idx

# if __name__ == '__main__':
#     print(name2idx_transform())
