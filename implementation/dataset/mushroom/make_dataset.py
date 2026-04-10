import pandas as pd
from torch.utils.data import Dataset
import torch
from typing import List

qs = pd.read_csv('questions.csv')
sols = pd.read_csv('solutions.csv')

class ModSeekDataset(Dataset):
    def __init__(self, qs: pd.DataFrame, sols = pd.DataFrame):
        self.qs_cols = qs.columns
        self.sols_cols = sols.columns
        
        self.num_qs = len(self.qs_cols)
        self.num_sols = len(self.sols_cols)
        
        self.qs_inst_tensor = torch.tensor(qs.to_numpy(), dtype=torch.float32)
        self.sols_inst_tesor = torch.tensor(sols.to_numpy(), dtype=torch.float32)
        
    def __getitem__(self, index):
        return self.qs_inst_tensor[index], self.sols_inst_tesor[index]
    
    def __len__(self):
        return len(self.qs_inst_tensor)
    
msd = ModSeekDataset(qs, sols)
torch.save(msd, 'dataset.pt')