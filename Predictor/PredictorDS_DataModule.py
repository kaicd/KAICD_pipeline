import os.path as osp
import dill
import pickle
import json
from argparse import ArgumentParser

import torch as th
import pytorch_lightning as pl
from typing import Optional, Union, Dict, List
from pytoda.datasets import DrugSensitivityDataset
from pytoda.smiles.smiles_language import SMILESLanguage
from torch.utils.data.dataloader import DataLoader


class PredictorDS_DataModule(pl.LightningDataModule):
    @classmethod
    def add_argparse_args(
        cls, parent_parser: ArgumentParser, **kwargs
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--train_sensitivity_filepath",
            type=str,
            default="data/pretraining/Predictor/ccle_ic50_train_fraction_0.9_id_246_seed_42.csv",
            help="Path to the drug sensitivity data.",
        )
        parser.add_argument(
            "--test_sensitivity_filepath",
            type=str,
            default="data/pretraining/Predictor/ccle_ic50_test_fraction_0.1_id_246_seed_42.csv",
            help="Path to the drug sensitivity data.",
        )
        parser.add_argument(
            "--train_dataset_filepath",
            type=str,
            default="preprocessing/PredictorDS_train_dataset.pkl",
            help="Path to the drug sensitivity data.",
        )
        parser.add_argument(
            "--test_dataset_filepath",
            type=str,
            default="preprocessing/PredictorDS_test_dataset.pkl",
            help="Path to the drug sensitivity data.",
        )
        parser.add_argument(
            "--gep_filepath",
            type=str,
            default="data/pretraining/Predictor/ccle-rnaseq_gene-expression.csv",
            help="Path to the gene expression profile data.",
        )
        parser.add_argument(
            "--smi_filepath",
            type=str,
            default="data/pretraining/Predictor/ccle.smi",
            help="Path to the SMILES data.",
        )
        parser.add_argument(
            "--gene_filepath",
            type=str,
            default="data/pretraining/Predictor/PredictorDS_gene.pkl",
            help="Path to the pickle object containing list of genes.",
        )
        parser.add_argument(
            "--smiles_language_filepath",
            type=str,
            default="data/pretraining/language_models/PredictorDS_smiles_language.pkl",
            help="Path to the SMILES language object.",
        )

        return parent_parser

    def __init__(
        self,
        project_filepath,
        train_sensitivity_filepath,
        test_sensitivity_filepath,
        train_dataset_filepath,
        test_dataset_filepath,
        gep_filepath,
        smi_filepath,
        smiles_language_filepath,
        gene_filepath,
        params_filepath,
        *args,
        **kwargs,
    ):
        super(PredictorDS_DataModule, self).__init__()
        self.train_sensitivity_filepath = project_filepath + train_sensitivity_filepath
        self.test_sensitivity_filepath = project_filepath + test_sensitivity_filepath
        self.train_dataset_filepath = project_filepath + train_dataset_filepath
        self.test_dataset_filepath = project_filepath + test_dataset_filepath
        self.gep_filepath = project_filepath + gep_filepath
        self.smi_filepath = project_filepath + smi_filepath
        self.smiles_language_filepath = smiles_language_filepath
        self.params_filepath = params_filepath
        self.smiles_language = SMILESLanguage.load(self.smiles_language_filepath)
        self.gene_filepath = gene_filepath
        self.params = {}

        # Process parameter file
        with open(self.params_filepath) as f:
            self.params.update(json.load(f))
        # Load the gene list
        with open(self.gene_filepath, "rb") as f:
            self.gene_list = pickle.load(f)

    def setup(self, stage: Optional[str] = None) -> None:
        drug_sensitivity_filepath = [
            self.train_sensitivity_filepath,
            self.test_sensitivity_filepath,
        ]
        augment = [
            self.params.get("augment_smiles", True),
            self.params.get("augment_test_smiles", False),
        ]
        drug_sensitivity_processing_parameters = [
            self.params.get("drug_sensitivity_processing_parameters", {})
        ]
        gene_expression_processing_parameters = [
            self.params.get("gene_expression_processing_parameters", {})
        ]

        if osp.exists(self.train_dataset_filepath) and osp.exists(
            self.test_dataset_filepath
        ):
            print("Preprocessing file already exists!\nLoading...")

            with open(self.train_dataset_filepath, "rb") as f:
                self.train_dataset = dill.load(f)
            with open(self.test_dataset_filepath, "rb") as f:
                self.test_dataset = dill.load(f)
            print("Done...!")
        else:
            print("Data preprocessing...")
            for i in range(2):
                dataset = DrugSensitivityDataset(
                    drug_sensitivity_filepath=drug_sensitivity_filepath[i],
                    smi_filepath=self.smi_filepath,
                    gene_expression_filepath=self.gep_filepath,
                    smiles_language=self.smiles_language,
                    gene_list=self.gene_list,
                    drug_sensitivity_min_max=self.params.get(
                        "drug_sensitivity_min_max", True
                    ),
                    drug_sensitivity_processing_parameters=drug_sensitivity_processing_parameters[
                        i
                    ],
                    augment=augment[i],
                    canonical=self.params.get("canonical", False),
                    kekulize=self.params.get("kekulize", False),
                    all_bonds_explicit=self.params.get("all_bonds_explicit", False),
                    all_hs_explicit=self.params.get("all_hs_explicit", False),
                    randomize=self.params.get("randomize", False),
                    remove_bonddir=self.params.get("remove_bonddir", False),
                    remove_chirality=self.params.get("remove_chirality", False),
                    selfies=self.params.get("selfies", False),
                    add_start_and_stop=self.params.get("smiles_start_stop_token", True),
                    padding_length=self.params.get("smiles_padding_length", None),
                    gene_expression_standardize=self.params.get(
                        "gene_expression_standardize", True
                    ),
                    gene_expression_min_max=self.params.get(
                        "gene_expression_min_max", False
                    ),
                    gene_expression_processing_parameters=gene_expression_processing_parameters[
                        i
                    ],
                    device=th.device("cpu"),
                    backend="eager",
                )

                if i == 0:
                    self.train_dataset = dataset
                    drug_sensitivity_processing_parameters.append(
                        self.params.get(
                            "drug_sensitivity_processing_parameters",
                            self.train_dataset.drug_sensitivity_processing_parameters,
                        )
                    )
                    gene_expression_processing_parameters.append(
                        self.params.get(
                            "gene_expression_processing_parameters",
                            self.train_dataset.gene_expression_dataset.processing,
                        )
                    )
                else:
                    self.test_dataset = dataset

            print("Saving...")
            with open(self.train_dataset_filepath, "wb") as f:
                dill.dump(self.train_dataset, f)
            with open(self.test_dataset_filepath, "wb") as f:
                dill.dump(self.test_dataset, f)
            print("Done...!")

    def dataloader(self, dataset, shuffle, **kwargs):
        return DataLoader(
            dataset=dataset,
            batch_size=self.params["batch_size"],
            shuffle=shuffle,
            drop_last=True,
            num_workers=self.params.get("num_workers", 4),
        )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return self.dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.dataloader(self.test_dataset, shuffle=False)
