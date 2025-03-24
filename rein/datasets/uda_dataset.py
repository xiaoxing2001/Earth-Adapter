# from .basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class UDA_dataset(object):
    def __init__(self,**cfg):
        self.source_dataset = DATASETS.build(cfg['source_dataset'])
        self.target_dataset = DATASETS.build(cfg['target_dataset'])
    
    def __getitem__(self,idx):
        source_data = self.source_dataset[idx % len(self.source_dataset)]
        target_data = self.target_dataset[idx % len(self.target_dataset)]
        data = {
            'source_data': source_data,
            'target_data': target_data
        }
        return data
    
    def __len__(self):
        return max(len(self.source_dataset),len(self.target_dataset))