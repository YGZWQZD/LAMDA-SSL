import numpy as np
from LAMDA_SSL.Dataset.SemiDataset import SemiDataset
from LAMDA_SSL.Base.VisionMixin import VisionMixin
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import os
import pickle
from LAMDA_SSL.Split.DataSplit import DataSplit
from LAMDA_SSL.Dataset.TrainDataset import TrainDataset
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset

class CIFAR10(SemiDataset,VisionMixin):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2471, 0.2435, 0.2616]

    def __init__(
        self,
        root: str,
        default_transforms=False,
        pre_transform=None,
        transforms=None,
        transform = None,
        target_transform = None,
        unlabeled_transform=None,
        valid_transform=None,
        test_transform=None,
        valid_size=None,
        labeled_size=0.1,
        stratified=False,
        shuffle=True,
        random_state=None,
        download: bool = False,

    ) -> None:
        self.default_transforms=default_transforms
        self.labeled_X=None
        self.labeled_y=None
        self.unlabeled_X=None
        self.unlabeled_y=None
        self.valid_X=None
        self.valid_y=None
        self.test_X=None
        self.test_y=None

        self.labeled_dataset=None
        self.unlabeled_dataset=None
        self.train_dataset=None
        self.valid_dataset = None
        self.test_dataset=None

        self.data_initialized=False

        self.len_test=None
        self.len_valid = None
        self.len_labeled=None
        self.len_unlabeled=None

        self.labeled_X_indexing_method=None
        self.labeled_y_indexing_method =None
        self.unlabeled_X_indexing_method =None
        self.unlabeled_y_indexing_method =None
        self.valid_X_indexing_method=None
        self.valid_indexing_method=None
        self.test_X_indexing_method=None
        self.test_y_indexing_method=None


        SemiDataset.__init__(self,pre_transform=pre_transform,transforms=transforms,transform=transform, target_transform=target_transform,
                             unlabeled_transform=unlabeled_transform,test_transform=test_transform,
                             valid_transform=valid_transform,labeled_size=labeled_size,valid_size=valid_size,
                             stratified=stratified,shuffle=shuffle,random_state=random_state)
        VisionMixin.__init__(self,mean=self.mean,std=self.std)


        if isinstance(root, (str, bytes)):
            root = os.path.expanduser(root)
        self.root = root

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self._load_meta()
        if self.default_transforms:
            self.init_default_transforms()
        self.init_dataset()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _init_dataset(self):
        test_X = []
        test_y = []
        for file_name, checksum in self.test_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                test_X.append(entry["data"])
                if "labels" in entry:
                    test_y.extend(entry["labels"])
                else:
                    test_y.extend(entry["fine_labels"])
        test_X = np.vstack(test_X).reshape(-1, 3, 32, 32)
        test_X = test_X.transpose((0, 2, 3, 1))

        train_X = []
        train_y = []
        for file_name, checksum in self.train_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                train_X.append(entry["data"])
                if "labels" in entry:
                    train_y.extend(entry["labels"])
                else:
                    train_y.extend(entry["fine_labels"])
        train_X = np.vstack(train_X).reshape(-1, 3, 32, 32)
        train_X = train_X.transpose((0, 2, 3, 1))

        if self.valid_size is not None:
            valid_X, valid_y, train_X, train_y = DataSplit(X=train_X, y=train_y,
                                                                   size_split=self.valid_size,
                                                                   stratified=self.stratified,
                                                                   shuffle=self.shuffle,
                                                                   random_state=self.random_state
                                                                   )
        else:
            valid_X=None
            valid_y=None

        if self.labeled_size is not None:
            labeled_X, labeled_y, unlabeled_X, unlabeled_y = DataSplit(X=train_X,y=train_y,
                                                                   size_split=self.labeled_size,
                                                                   stratified=self.stratified,
                                                                   shuffle=self.shuffle,
                                                                   random_state=self.random_state
                                                                   )
        else:
            labeled_X, labeled_y=train_X,train_y
            unlabeled_X, unlabeled_y=None,None
        self.test_dataset=LabeledDataset(pre_transform=self.pre_transform,transform=self.test_transform)
        self.test_dataset.init_dataset(test_X,test_y)
        self.valid_dataset=LabeledDataset(pre_transform=self.pre_transform,transform=self.valid_transform)
        self.valid_dataset.init_dataset(valid_X,valid_y)
        self.train_dataset = TrainDataset(pre_transform=self.pre_transform,transforms=self.transforms,transform=self.transform,
                                          target_transform=self.target_transform,unlabeled_transform=self.unlabeled_transform)
        labeled_dataset=LabeledDataset(pre_transform=self.pre_transform,transforms=self.transforms,transform=self.transform,
                                          target_transform=self.target_transform)
        labeled_dataset.init_dataset(labeled_X, labeled_y)
        unlabeled_dataset=UnlabeledDataset(pre_transform=self.pre_transform,transform=self.unlabeled_transform)
        unlabeled_dataset.init_dataset(unlabeled_X, unlabeled_y)
        self.train_dataset.init_dataset(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset)


