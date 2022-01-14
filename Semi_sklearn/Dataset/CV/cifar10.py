import numpy as np
from .SemiVisionDataset import  SemiVisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import os
import pickle
from Semi_sklearn.Split.SemiSplit import SemiSplit
from Semi_sklearn.Dataset.CV.SemiTrainVisionDataset import SemiTrainVisionDataset
from Semi_sklearn.Dataset.CV.LabledVisionDataset import LabledVisionDataset
from Semi_sklearn.Dataset.CV.UnlabledVisionDataset import UnlabledVisionDataset
class CIFAR10(SemiVisionDataset):
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
        transform = None,
        target_transform = None,
        labled_size=0.1,
        stratified=False,
        shuffle=True,
        random_state=None,
        download: bool = False

    ) -> None:
        print(1)
        self.labled_X=None
        self.labled_y=None
        self.unlabled_X=None
        self.unlabled_y=None
        self.test_X=None
        self.test_y=None
        self.labled_dataset=None
        self.unlabled_dataset=None
        self.train_dataset=None
        self.test_dataset=None
        self.data_initialized=False
        self.len_test=None
        self.len_labled=None
        self.len_unlabled=None
        self.labled_X_indexing_method=None
        self.labled_y_indexing_method =None
        self.unlabled_X_indexing_method =None
        self.unlabled_y_indexing_method =None
        self.test_X_indexing_method=None
        self.test_y_indexing_method=None
        super().__init__(root, transform=transform, target_transform=target_transform,labled_size=labled_size,
                        stratified=stratified,shuffle=shuffle,random_state=random_state)
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self._load_meta()

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
        print(5)
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
        # test_X = test_X.transpose((0, 2, 3, 1))

        self.train_X = []
        self.train_y = []
        for file_name, checksum in self.train_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.train_X.append(entry["data"])
                if "labels" in entry:
                    self.train_y.extend(entry["labels"])
                else:
                    self.train_y.extend(entry["fine_labels"])
        self.train_X = np.vstack(self.train_X).reshape(-1, 3, 32, 32)
        # self.train_X = self.train_X.transpose((0, 2, 3, 1))
        labled_X, labled_y, unlabled_X, unlabled_y = SemiSplit(X=self.train_X, y=self.train_y,
                                                               labled_size=self.labled_size,
                                                               stratified=self.stratified,
                                                               shuffle=self.shuffle,
                                                               random_state=self.random_state
                                                               )
        self.test_dataset=LabledVisionDataset()
        self.test_dataset.init_dataset(test_X,test_y)
        self.train_dataset = SemiTrainVisionDataset()
        labled_dataset=LabledVisionDataset()
        labled_dataset.init_dataset(labled_X, labled_y)
        unlabled_dataset=UnlabledVisionDataset()
        unlabled_dataset.init_dataset(unlabled_X, unlabled_y)
        self.train_dataset.init_dataset(labled_dataset=labled_dataset,unlabled_dataset=unlabled_dataset)





