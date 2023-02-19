import torch
import numpy as np
import pyspark.sql.functions as F
from models.transformation import default_transforms
from torch.utils.data import Dataset, DataLoader, random_split

class PointCloudData(Dataset):
    def __init__(self, df, num_classes=10, split="train", transform=default_transforms()):
        """
        Load dataset from df based on the split parameter.
        """

        assert df is not None, "invalid dataframe"
        assert split in ['train', 'val', 'test'], "split error value!"

        self.samples = []
        self.transforms = transform

        if num_classes == 4:
            self.classes = [ 'airplane', 'car', 'chair', 'table' ]
        elif num_classes == 10:
            self.classes =  [ 'airplane', 'car', 'chair', 'table', 'bench', 'lamp', 'sofa', 'rifle', 'speaker', 'vessel' ]
        else:
            raise "invalid num classes!"
        
        self.cat2label = { item: item_idx for item_idx, item in enumerate(self.classes) }
        self.id2label = { item_idx: item for item_idx, item in enumerate(self.classes) }
        self.load_data(df, split)

    def load_data(self, df, split):
        self.sampels = []
        samplesPerLabel = {}

        for row in df.filter(df['split'] == split).collect():
            if not row.label in self.cat2label:
                continue

            if not row.label in  samplesPerLabel:
                samplesPerLabel[row.label] = 0

            sample = { "features": row.features, "label": self.cat2label[row.label] }
            samplesPerLabel[row.label] += 1
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features = np.array(self.samples[idx]['features'])
        label = self.samples[idx]['label']

        if self.transforms:
            features = self.transforms(features)
            
        return features, label
