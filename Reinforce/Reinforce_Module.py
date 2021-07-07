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
        self.gen_mols, self.gen_prot, self.gen_affinity = [], [], []

    """
    Implementation of the policy gradient algorithm.
    """

    def forward(self, protein_name):
        # Encode the protein
        latent_z = self.encode_protein(protein_name, self.batch_size)
        # Produce molecules
        valid_smiles, valid_nums, valid_idx = self.get_smiles_from_latent(
            latent_z, remove_invalid=True
        )
        # Get rewards (list, one reward for each valid smiles)
        rewards = self.reward_fn(valid_smiles, protein)
        # valid_nums is a list of torch.Tensor, each with varying length,
        padded_nums = th.nn.utils.rnn.pad_sequence(valid_nums)
        num_mols = padded_nums.shape[1]
        self.decoder._update_batch_size(num_mols, device=self.device)
        # Batch processing
        lrps = 1
        if self.generator.decoder.latent_dim == 2 * self.encoder.latent_size:
            lrps = 2
        hidden = self.decoder.latent_to_hidden(
            latent_z.repeat(self.generator.decoder.n_layers, 1, lrps)[:, valid_idx, :]
        ).to(self.device)
        stack = self.decoder.init_stack.to(self.device)

        return padded_nums, hidden, stack, rewards

    def training_step(self, batch, *args, **kwargs):
        padded_nums, hidden, stack, rewards = self(batch[0])

        rl_loss = 0
        for p in range(len(padded_nums) - 1):
            output, hidden, stack = self.decoder(
                th.unsqueeze(padded_nums[p], 0), hidden, stack
            )
            output = self.decoder.output_layer(output).squeeze()
            log_probs = F.softmax(output, dim=1)
            target_char = th.unsqueeze(padded_nums[p + 1], 1)
            rl_loss -= th.mean(
                log_probs.gather(1, target_char)
                * th.unsqueeze(th.Tensor(rewards).to(self.device), 1)
            )

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
        toxes = th.Tensor([self.tox21(s) for s in smiles]).to(self.device)
        # Filtering (affinity > 0.5)
        useful_idx = preds > 0.5
        useful_smiles = smiles[useful_idx]
        useful_preds = preds[useful_idx]
        useful_toxes = toxes[useful_idx]
        for p, s in zip(useful_preds, useful_smiles):
            self.gen_mols.append(s)
            self.gen_prot.append(self.protein_test_name)
            self.gen_affinity.append(p)
        # Filtering (tox == 1.0 -> non-toxic)
        non_toxic_useful_idx = useful_toxes == 1.0
        non_toxic_useful_smiles = useful_smiles[non_toxic_useful_idx]
        non_toxic_useful_preds = useful_preds[non_toxic_useful_idx]
        # Log efficacy and non toxicity ratio
        biased_ratio = th.round(100 * (th.sum(preds > 0.5) / len(preds)), 1)
        self.biased_ratios.append(biased_ratio)
        tox_ratio = th.round(100 * (th.sum(toxes == 1.0) / len(toxes)), 1)
        self.tox_ratios.append(tox_ratio)
        self.log("efficacy_ratio", biased_ratio)
        self.log("non_tox_ratio", tox_ratio)
        # Log distribution plot
        plot_and_compare_proteins(
            self.unbiased_preds,
            preds.detach().cpu().numpy(),
            self.protein_test_name,
            self.current_epoch,
            self.fig_save_path,
            "train",
            self.batch_size,
        )
        self.logger.experiment.log(
            {
                "NAIVE and BIASED binding compounds distribution": [
                    Image(
                        pilimg.open(
                            os.path.join(
                                self.fig_save_path,
                                f"train_{self.protein_test_name}_epoch_{self.current_epoch}_eff_{biased_ratio}.png",
                            )
                        )
                    )
                ]
            }
        )
        # Log top 4 generate molecule
        idx = np.argsort(non_toxic_useful_preds)[::-1]
        lead = []
        captions = []
        for i in idx:
            mol = Chem.MolFromSmiles(non_toxic_useful_smiles[i])
            if mol:
                lead.append(mol)
                captions.append(str(non_toxic_useful_preds[i]))
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
        df.to_csv(os.path.join(self.fig_save_path, "results", "generated.csv"))
        self.logger.experiment.log(Table(dataframe=df))
