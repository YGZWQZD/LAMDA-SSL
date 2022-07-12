from LAMDA_SSL.Dataset.SemiDataset import SemiDataset
from LAMDA_SSL.Base.TextMixin import TextMixin
from torchtext.utils import download_from_url,extract_archive
import io
from LAMDA_SSL.Split.DataSplit import DataSplit
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Dataset.TrainDataset import TrainDataset
import os

class  IMDB(SemiDataset,TextMixin):
    URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    MD5 = "7c2ac02c03563afcf9b574c7e56c153a"
    NUM_LINES = {
        "train": 25000,
        "test": 25000,
        "unsup": 50000,
    }
    _PATH = "aclImdb_v1.tar.gz"
    DATASET_NAME = "IMDB"
    def __init__(self,root,
        default_transforms=False,
        pre_transform=None,
        transforms=None,
        transform = None,
        target_transform = None,
        unlabeled_transform=None,
        valid_transform=None,
        test_transform=None,
        valid_size=None,
        stratified=False,
        shuffle=True,
        random_state=None,
        download: bool = False,
        word_vocab=None, vectors=None, length=300, unk_token='<unk>', pad_token='<pad>',
        min_freq=1, special_first=True, default_index=None):

        SemiDataset.__init__(self,pre_transform=pre_transform,transforms=transforms,transform=transform, target_transform=target_transform,
                             unlabeled_transform=unlabeled_transform,test_transform=test_transform,
                             valid_transform=valid_transform,valid_size=valid_size,
                             stratified=stratified,shuffle=shuffle,random_state=random_state)

        TextMixin.__init__(self,length=length,vectors=vectors,word_vocab=word_vocab,unk_token=unk_token,
                               pad_token=pad_token,min_freq=min_freq,special_first=special_first,default_index=default_index)
        self.root=root
        self.default_transforms=default_transforms
        self.classes=['neg','pos']
        self.class_to_idx={'neg':0,'pos':1}

        if download:
            self.download()


        walk = os.walk(os.path.join(self.root,'aclImdb'))
        self.extracted_files = []
        for root, dirs, files in walk:
            for item in files:
                self.extracted_files.append(os.path.join(root, item))
        if self.default_transforms:
            self.init_default_transforms()
        self.init_dataset()


    def download(self):
        dataset_tar = download_from_url(self.URL,root=self.root,
                                    hash_value=self.MD5, hash_type='md5')
        extract_archive(dataset_tar)


    def _init_dataset(self):
        test_X=[]
        test_y=[]
        labeled_X=[]
        labeled_y=[]
        unlabeled_X=[]
        for fname in self.extracted_files:
            if 'urls' in fname or 'Bow' in fname:
                continue
            if 'train' in fname:
                if 'pos' in fname :
                    with io.open(fname, encoding="utf8") as f:
                        labeled_X.append(f.read())
                    labeled_y.append(1)
                elif 'neg' in fname:
                    with io.open(fname, encoding="utf8") as f:
                        labeled_X.append(f.read())
                    labeled_y.append(0)
                elif 'unsup' in fname:
                    with io.open(fname, encoding="utf8") as f:
                        unlabeled_X.append(f.read())

            elif 'test' in fname:
                if 'pos' in fname :
                    with io.open(fname, encoding="utf8") as f:
                        test_X.append(f.read())
                    test_y.append(1)
                elif 'neg' in fname:
                    with io.open(fname, encoding="utf8") as f:
                        test_X.append(f.read())
                    test_y.append(0)

        if self.valid_size is not None:
            valid_X, valid_y, labeled_X, labeled_y = DataSplit(X=labeled_X, y=labeled_y,
                                                                   size_split=self.valid_size,
                                                                   stratified=self.stratified,
                                                                   shuffle=self.shuffle,
                                                                   random_state=self.random_state
                                                                   )
        else:
            valid_X = None
            valid_y = None

        self.test_dataset = LabeledDataset(pre_transform=self.pre_transform,transform=self.test_transform)
        self.test_dataset.init_dataset(test_X, test_y)
        self.valid_dataset = LabeledDataset(pre_transform=self.pre_transform,transform=self.valid_transform)
        self.valid_dataset.init_dataset(valid_X, valid_y)
        self.train_dataset = TrainDataset(pre_transform=self.pre_transform,transforms=self.transforms, transform=self.transform,
                                          target_transform=self.target_transform,
                                          unlabeled_transform=self.unlabeled_transform)
        labeled_dataset = LabeledDataset(pre_transform=self.pre_transform,transforms=self.transforms, transform=self.transform,
                                       target_transform=self.target_transform)
        labeled_dataset.init_dataset(labeled_X, labeled_y)
        unlabeled_dataset = UnlabeledDataset(pre_transform=self.pre_transform,transform=self.unlabeled_transform)
        unlabeled_dataset.init_dataset(unlabeled_X)
        self.train_dataset.init_dataset(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset)



