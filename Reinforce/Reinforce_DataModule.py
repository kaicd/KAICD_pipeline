import pandas as pd
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from typing import Optional, Union, List, Dict

from Utility.utils import ProteinDataset


class Reinforce_DataModule(pl.LightningDataModule):
    def __init__(self, protein_data_path, test_protein_id, **kwargs):
        super(Reinforce_DataModule, self).__init__()
        self.protein_data_path = protein_data_path
        self.test_protein_id = test_protein_id

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = ProteinDataset(self.protein_data_path, self.test_protein_id)

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(dataset=self.dataset, batch_size=1, shuffle=True)
