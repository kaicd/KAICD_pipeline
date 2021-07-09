import torch as th
import pandas as pd
from torch.utils.data.dataset import Dataset
from pytoda.files import read_smi


class ProteinDataset(Dataset):
    """
    Protein data for conditioning
    """

    def __init__(
        self, protein_data_path, protein_test_idx, transform=None, *args, **kwargs
    ):
        """
        :param protein_data_path: protein data file(.smi or .csv) path
        :param transform: optional transform
        """
        # Load protein sequence data
        if protein_data_path.endswith(".smi"):
            self.protein_df = read_smi(protein_data_path, names=["Sequence"])
        elif protein_data_path.endswith(".csv"):
            self.protein_df = pd.read_csv(protein_data_path, index_col="entry_name")
        else:
            raise TypeError(
                f"{protein_data_path.split('.')[-1]} files are not supported."
            )

        self.transform = transform

        # Drop protein sequence data used in testing
        self.origin_protein_df = self.protein_df
        self.protein_df = self.protein_df.drop(self.protein_df.index[protein_test_idx])

    def __len__(self):
        return len(self.protein_df)

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
        sample = self.protein_df.iloc[idx].name

        if self.transform:
            sample = self.transform(sample)

        return sample
