# %%
"""SIDER evaluator."""
import torch

from .drug_evaluator import DrugEvaluator


class SIDER(DrugEvaluator):
    """
    SIDER evaluation class.
    Inherits from DrugEvaluator and evaluates the side effects of a SMILES.
    Model predicts 27 classes
    """

    def __init__(
        self, project_path: str, params_path: str, model_path: str, device: torch.device
    ):
        """

        Arguments:
            model_path {string} -- Path to the pretrained model
        """

        super(SIDER, self).__init__()
        self.device = device
        self.load_mca(project_path, params_path, model_path)

    def __call__(self, smiles: str) -> float:
        """
        Forward pass through the model.

        Arguments:
            smiles {str} -- SMILES of molecule
        Returns:
            float -- Averaged  predictions from the model

        TODO: Should be able to understand iterables
        """
        # Error handling.
        if not type(smiles) == str:
            raise TypeError(f"Input must be String, not :{type(smiles)}")

        smiles_tensor = self.preprocess_smiles(smiles)
        return self.sider_score(smiles_tensor)

    def sider_score(self, smiles_tensor: torch.Tensor) -> float:
        """
        Forward pass through the model.

        Arguments:
            smiles_tensor {torch.Tensor} -- Tensor of shape 2 x SMILES_tokens

        Returns:
            float -- Averaged  predictions from the model
        """

        # Test the compound
        predictions, _ = self.model(smiles_tensor)
        # To allow accessing the raw predictions from outside
        self.predictions = predictions[0, :]

        return 1.0 - float(self.predictions.mean())
