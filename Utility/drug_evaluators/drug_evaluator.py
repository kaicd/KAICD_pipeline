"""Drug evaluator."""
import json
import os

import torch as th
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.smiles.transforms import (
    Canonicalization,
    Kekulize,
    NotKekulize,
    RemoveIsomery,
    Selfies,
    SMILESToTokenIndexes,
)
from pytoda.transforms import Compose, ToTensor

from Toxicity.Toxicity_Module import Toxicity_Module


class DrugEvaluator:
    """
    Abstract definition of DrugEvaluator class.
    This scaffold is supposed  to be extended by specific
    drug evaluation metrics.
    """

    def __init__(self):
        pass

    def __call__(self, smiles: str):

        raise NotImplementedError

    def load_mca(self, project_path: str, params_path: str, model_path: str):
        """
        Restores pretrained MCA

        Arguments:
            model_path {String} -- Path to the model
        """
        with open(params_path) as f:
            params = json.load(f)
        # Set up language and transforms
        self.smiles_language = SMILESLanguage.load(
            os.path.join(project_path, "Config", "PredictorBA_smiles_language.pkl")
        )
        self.transforms = self.compose_smiles_transforms(params)
        # Initialize and restore model weights
        self.model = Toxicity_Module.load_from_checkpoint(
            os.path.join(project_path, model_path), params_filepath=params_path
        )
        self.model.eval()

    def compose_smiles_transforms(self, params: dict) -> Compose:
        """
        Create transforms that were applied during model training

        Arguments:
            params {dict} -- Model parameter to retrieve transforms
        Returns:
            Compose -- Object to perform transforms
        """

        self.canonical = params.get("canonical", False)
        self.kekulize = params.get("kekulize", False)
        self.canonical = params.get("canonical", False)
        self.all_bonds_explicit = params.get("all_bonds_explicit", False)
        self.all_hs_explicit = params.get("all_hs_explicit", False)
        self.remove_bonddir = params.get("remove_bonddir", False)
        self.remove_chirality = params.get("remove_chirality", False)
        self.selfies = params.get("selfies", False)

        transforms = []
        if self.canonical:
            transforms += [Canonicalization()]
        else:
            if self.remove_bonddir or self.remove_chirality:
                transforms += [
                    RemoveIsomery(
                        bonddir=self.remove_bonddir, chirality=self.remove_chirality
                    )
                ]
            if self.kekulize:
                transforms += [
                    Kekulize(
                        all_bonds_explicit=self.all_bonds_explicit,
                        all_hs_explicit=self.all_hs_explicit,
                    )
                ]
            elif self.all_bonds_explicit or self.all_hs_explicit:
                transforms += [
                    NotKekulize(
                        all_bonds_explicit=self.all_bonds_explicit,
                        all_hs_explicit=self.all_hs_explicit,
                    )
                ]
            if self.selfies:
                transforms += [Selfies()]

        transforms += [SMILESToTokenIndexes(smiles_language=self.smiles_language)]
        transforms += [ToTensor(device=self.device)]
        return Compose(transforms)

    def preprocess_smiles(self, smiles: str) -> th.Tensor:
        """
        Apply transforms that were applied during model training.

        Arguments:
            smiles {str} -- SMILES of molecules
        Returns:
            smiles (torch.Tensor) -- Tensor of shape 2 x T (unpadded but
                repeated twice to circumvent possible batch norm difficulties).
        """
        return th.unsqueeze(self.transforms(smiles), 0).repeat(2, 1)
