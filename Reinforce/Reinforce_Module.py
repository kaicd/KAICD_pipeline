"""PaccMann^RL: Policy gradient class"""
import os

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
from rdkit import Chem
from wandb import Image, Table
from PIL import Image as pilimg

from .Reinforce_Base import Reinforce_base
from Utility.utils import generate_mols_img, plot_and_compare_proteins


class Reinforce(Reinforce_base):
    """
    Pipeline to reproduce the results using pytorch_lightning of the paper
    Data-driven molecular design for discovery and synthesis of novel ligands:
    a case study on SARS-CoV-2(Machine Learning: Science and Technology, 2021).
    """

    def __init__(self, **kwargs):
        super(Reinforce, self).__init__(**kwargs)
        # Define for save result(good molecules!) in dataframe
        self.biased_ratios, self.tox_ratios = [], []
        self.rewards, self.rl_losses = [], []
        self.gen_mols, self.gen_prot, self.gen_affinity, self.toxes = [], [], [], []
        self.non_toxic_useful_smiles, self.non_toxic_useful_preds = [], []

    """
    Implementation of the policy gradient algorithm.
    """

    def forward(self, protein_name):
        super().forward(protein_name)
        # Encode the protein
        latent_z = self.encode_protein(protein_name, self.batch_size)
        # Produce molecules
        valid_smiles, valid_nums, valid_idx = self.get_smiles_from_latent(
            latent_z, remove_invalid=True
        )
        # Get rewards (list, one reward for each valid smiles)
        rewards = self.reward_fn(valid_smiles, protein_name)
        # valid_nums is a list of torch.Tensor, each with varying length,
        padded_nums = th.nn.utils.rnn.pad_sequence(valid_nums)
        num_mols = padded_nums.shape[1]
        self.decoder._update_batch_size(num_mols, device=self.device)
        # Batch processing
        lrps = 1
        if self.decoder.latent_dim == 2 * self.encoder.latent_size:
            lrps = 2
        hidden = self.decoder.latent_to_hidden(
            latent_z.repeat(self.decoder.n_layers, 1, lrps)[:, valid_idx, :]
        ).to(self.device)
        stack = self.decoder.init_stack.to(self.device)

        return padded_nums, hidden, stack, rewards

    def on_train_start(self):
        self.update_params(self.params)

    def training_step(self, batch, *args, **kwargs):
        padded_nums, hidden, stack, rewards = self(batch[0])
        rewards = rewards.detach().cpu()
        rl_loss = 0
        for p in range(len(padded_nums) - 1):
            output, hidden, stack = self.decoder(
                th.unsqueeze(padded_nums[p], 0), hidden, stack
            )
            output = self.decoder.output_layer(output).squeeze()
            log_probs = F.log_softmax(output, dim=1)
            target_char = th.unsqueeze(padded_nums[p + 1], 1)
            reward_tensor = th.unsqueeze(th.Tensor(rewards), 1).to(self.device)
            rl_loss -= th.mean(log_probs.gather(1, target_char) * reward_tensor)

        summed_reward = th.mean(th.Tensor(rewards).to(self.device))
        if self.grad_clipping is not None:
            th.nn.utils.clip_grad_norm_(
                list(self.decoder.parameters()) + list(self.encoder.parameters()),
                self.grad_clipping,
            )
        # Save and log results
        self.rewards.append(summed_reward)
        self.rl_losses.append(rl_loss)
        self.log("mean_rewards", summed_reward)
        self.log("rl_loss", rl_loss)

        return rl_loss

    def training_epoch_end(self, *args, **kwargs):
        smiles, preds = self.generate_compounds_and_evaluate(
            batch_size=self.batch_size, protein=self.protein_test_name
        )
        preds = preds.detach().cpu().numpy()
        # Filtering (affinity > 0.5, tox == 1.0)
        useful_smiles = [s for i, s in enumerate(smiles) if preds[i] > 0.5]
        useful_preds = preds[preds > 0.5]
        for p, s in zip(useful_preds, useful_smiles):
            self.gen_mols.append(s)
            self.gen_prot.append(self.protein_test_name)
            self.gen_affinity.append(p)

            tox = self.tox21(s)
            self.toxes.append(tox)
            if tox == 1.0:
                self.non_toxic_useful_smiles.append(s)
                self.non_toxic_useful_preds.append(p)
        # Log efficacy and non toxicity ratio
        plot_and_compare_proteins(
            self.unbiased_preds,
            preds,
            self.protein_test_name,
            self.current_epoch,
            self.project_path,
            "train",
            self.batch_size,
        )
        biased_ratio = np.round((np.sum(preds > 0.5) / len(preds)) * 100, 1)
        self.biased_ratios.append(biased_ratio)
        all_toxes = np.array([self.tox21(s) for s in smiles])
        tox_ratio = np.round((np.sum(all_toxes == 1.0) / len(all_toxes)) * 100, 1)
        self.tox_ratios.append(tox_ratio)
        self.log("efficacy_ratio", biased_ratio)
        self.log("non_tox_ratio", tox_ratio)
        # Log distribution plot

        self.logger.experiment.log(
            {
                "NAIVE and BIASED binding compounds distribution": [
                    Image(
                        pilimg.open(
                            os.path.join(
                                self.project_path,
                                "binding_images",
                                f"train_{self.protein_test_name}_epoch_{self.current_epoch}_eff_{biased_ratio}.png",
                            )
                        )
                    )
                ]
            }
        )
        # Log top 4 generate molecule
        idx = np.argsort(self.non_toxic_useful_preds)[::-1]
        lead = []
        captions = []
        for i in idx:
            mol = Chem.MolFromSmiles(self.non_toxic_useful_smiles[i])
            if mol:
                lead.append(mol)
                captions.append(str(self.non_toxic_useful_preds[i]))
                if len(lead) == 4:
                    break

        if len(lead) > 0:
            self.logger.experiment.log(
                {
                    "Top N Generative Molecules": [
                        Image(generate_mols_img(lead, legends=captions))
                    ]
                }
            )

    def on_train_end(self):
        df = pd.DataFrame(
            {
                "protein": self.gen_prot,
                "SMILES": self.gen_mols,
                "Binding probability": self.gen_affinity,
                "Tox21": self.toxes,
            }
        )
        df.to_csv(
            os.path.join(
                self.project_path, "results", self.protein_test_name + "_generated.csv"
            )
        )
        self.logger.experiment.log(Table(dataframe=df))
