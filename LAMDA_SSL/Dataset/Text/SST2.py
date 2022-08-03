from LAMDA_SSL.Dataset.SemiDataset import SemiDataset
from LAMDA_SSL.Base.TextMixin import TextMixin
from torchtext.utils import download_from_url,extract_archive
from LAMDA_SSL.Split.DataSplit import DataSplit
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Dataset.TrainDataset import TrainDataset
import os

class  SST2(SemiDataset,TextMixin):
    URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
    MD5 = "9f81648d4199384278b86e315dac217c"
    NUM_LINES = {
        "train": 67349,
        "dev": 872,
        "test": 1821,
    }
    path= {
        "train": os.path.join("SST-2", "train.tsv"),
        "dev": os.path.join("SST-2", "dev.tsv"),
        "test": os.path.join("SST-2", "test.tsv"),
    }
    def __init__(self,root,
        default_transforms = False,
        pre_transform=None,
        transforms=None,
        transform = None,
        target_transform = None,
        unlabeled_transform=None,
        valid_transform=None,
        test_transform=None,
        labeled_size=None,
        valid_size=None,
        stratified=False,
        shuffle=True,
        random_state=None,
        download: bool = False,
        word_vocab=None, vectors=None, length=50, unk_token='<unk>', pad_token='<pad>',
        min_freq=1, special_first=True, default_index=None):

        SemiDataset.__init__(self,pre_transform=pre_transform,transforms=transforms,transform=transform, target_transform=target_transform,
                             unlabeled_transform=unlabeled_transform,test_transform=test_transform,
                             valid_transform=valid_transform,valid_size=valid_size,labeled_size=labeled_size,
                             stratified=stratified,shuffle=shuffle,random_state=random_state)


        self.default_transforms=default_transforms
        self.root=root

        if download:
            self.download()
        self.init_dataset()
        TextMixin.__init__(self,length=length,vectors=vectors,word_vocab=word_vocab,unk_token=unk_token,
                               pad_token=pad_token,min_freq=min_freq,special_first=special_first,default_index=default_index)
        if self.default_transforms:
            self.init_default_transforms()




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
        train_path=os.path.join(self.root,self.path['train'])
        with open(train_path) as infile:
            for line in infile:
                tsplit = line.split("\t")
                if tsplit[0]=='sentence':
                    continue
                labeled_X.append(tsplit[0])
                labeled_y.append(int(tsplit[1]))

        test_path=os.path.join(self.root,self.path['dev'])
        with open(test_path) as infile:
            for line in infile:
                tsplit = line.split("\t")
                if tsplit[0]=='sentence':
                    continue
                test_X.append(tsplit[0])
                test_y.append(int(tsplit[1]))

        unlabeled_path=os.path.join(self.root,self.path['test'])
        with open(unlabeled_path) as infile:
            for line in infile:
                tsplit = line.split("\t")
                if tsplit[0]=='index':
                    continue
                unlabeled_X.append(tsplit[1])


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

        if self.labeled_size is not None:
            labeled_X, labeled_y, unlabeled_X,unlabeled_y = DataSplit(X=labeled_X, y=labeled_y,
                                                               size_split=self.labeled_size,
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



