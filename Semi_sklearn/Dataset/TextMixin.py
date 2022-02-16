class TextMixin:
    def __init__(self,transform,target_transform,transforms,unlabled_transform,test_trasform):
        self.transform=transform
        self.transforms=transforms
        self.target_transform=target_transform
        self.unlabled_transform=unlabled_transform
        self.test_transform=test_trasform
    def default_transform(self):
        pass
        #构建字典
        #统计之类的
        #绘制词云
        #default transforms