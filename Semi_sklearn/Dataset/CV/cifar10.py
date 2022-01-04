import numpy as np
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

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        labled_size=0.1,
        test_size=None,
        stratified=False,
        shuffle=True,
        random_state=None,
        download: bool = False

    ) -> None:

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        super().__init__(root, transform=transform, target_transform=target_transform,labled_size=labled_size,
                        test_size=test_size,stratified=stratified,shuffle=shuffle,random_state=random_state)


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
    def get_data(self,test,labled):
        if test:
            if self.test_X is None:
                self.test_X: Any = []
                self.test_y=[]
                for file_name, checksum in self.test_list:
                    file_path = os.path.join(self.root, self.base_folder, file_name)
                    with open(file_path, "rb") as f:
                        entry = pickle.load(f, encoding="latin1")
                        self.test_X.append(entry["data"])
                        if "labels" in entry:
                            self.test_y.extend(entry["labels"])
                        else:
                            self.test_y.extend(entry["fine_labels"])
                self.test_X = np.vstack(self.test_X).reshape(-1, 3, 32, 32)
                self.test_X = self.test_X.transpose((0, 2, 3, 1)) 
            return self.test_X,self.test_y
        else:
            if self.labled_X is None:
                self.train_X: Any = []
                self.train_y=[]
                for file_name, checksum in self.test_list:
                    file_path = os.path.join(self.root, self.base_folder, file_name)
                    with open(file_path, "rb") as f:
                        entry = pickle.load(f, encoding="latin1")
                        self.train_X.append(entry["data"])
                        if "labels" in entry:
                            self.train_y.extend(entry["labels"])
                        else:
                            self.train_y.extend(entry["fine_labels"])
                self.train_X = np.vstack(self.train_X).reshape(-1, 3, 32, 32)
                self.train_X = self.train_X.transpose((0, 2, 3, 1))
                self.labled_X,self.labled_y,self.unlabled_X,self.unlabled_y=SemiSplit(X=self.train_X,y=self.train_y,
                                                                labled_size=self.labled_size,
                                                                stratified=self.stratified,
                                                                shuffle=self.shuffle,
                                                                random_state=self.random_state,
                                                                X_indexing=self.labled_X_indexing, 
                                                                y_indexing=self.labled_y_indexing
                                                                )
            if labled:
                return self.labled_X,self.labled_y
            else:
                return self.unlabled_X,self.unlabled_y

