import os
from PIL import Image
from torch.utils.data.dataset import Dataset
def make_dataset_with_labels(dir, classnames):
    images = []
    labels = []
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            dirname = os.path.split(root)[-1]
            if dirname not in classnames:
                continue

            label = classnames.index(dirname)

            path = os.path.join(root, fname)
            images.append(path)
            labels.append(label)
    return images, labels

classes=['0', '1', '2', '3',
        '4', '5', '6', '7',
        '8', '9', '10', '11']

class ImageCLEF(Dataset):
    def __init__(self, root, domain='webcam', transform=None,classnames=classes):
        super(ImageCLEF, self).__init__()
        self.root=root
        self.domain=domain
        self.transform=transform
        self.data_paths = []
        self.data_labels = []
        self.data_root=os.path.join(self.root, self.domain)
        self.classnames = classnames
        self.data_paths, self.data_labels = make_dataset_with_labels(self.data_root, self.classnames)
        self.imgs=[]
        for index in range(len(self.data_paths)):
            path = self.data_paths[index]
            img = Image.open(path).convert('RGB')
            self.imgs.append(img)
        assert(len(self.data_paths) == len(self.data_labels)), \
            'The number of images (%d) should be equal to the number of labels (%d).' % \
            (len(self.data_paths), len(self.data_labels))

    def __getitem__(self, index):
        img=self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.data_labels[index]
        return img, label

    def make_dataset_classwise(self, category):
        imgs=[]
        labels=[]
        for _ in range(len(self.data_labels)):
            if self.data_labels[_]==category:
                imgs.append(self.imgs[_])
                labels.append(self.data_labels[_])
        return imgs,labels

    def __len__(self):
        return len(self.data_paths)
