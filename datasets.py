import torch
import pandas as pd
from torch.utils.data import Dataset

class FallDataset(Dataset):
    def __init__(self,csv_file, transform =None):
        self.data = pd.read_csv(csv_file, error_bad_lines=False)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        acc_g = self.data.iloc[idx, 0:3].values
        out = self.data.iloc[idx, 3]
        acc_g = acc_g.astype('float32')
        target = out.astype('long')

        sample = {'data': acc_g,
                  'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):

    def __call__(self, sample):
        data, target = \
            sample['data'], sample['target']


        data_t = torch.from_numpy(data)
        # target_t = target
        
        out_dict = {'data': data_t,
                  'target': target}

        return out_dict
