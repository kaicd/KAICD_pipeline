### Download data and pretrained models

Download the [data](https://ibm.ent.box.com/v/paccmann-sarscov2-data) as reported in the [requirements section](#requirements).
From now on, we will assume that they are stored in the root of the repository in a folder called `data`, following this structure:

```console
data
├── pretraining
│   ├── ProteinVAE
│   ├── SELFIESVAE
│   ├── affinity_predictor
│   ├── language_models
│   └── toxicity_predictor
└── training
```
This is around **6GB** of data, required for pretaining multiple models.
Also, the workload required to run the full pipeline is intensive and might not be straightforward to run all the steps on a desktop laptop.


# PaccMann_rl generator(Pytorch-Lightning)
It will be implemented soon...

### affinity predictor(Pytorch-Lightning)
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python ./DTI/DTI_Main.py.py
```

### toxicity predictor(Pytorch-Lightning)
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python ./Toxicity/Toxicity_Main.py
```

### protein VAE(Pytorch-Lightning)
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python ./ProtVAE/ProtVAE_Main.py
```

### SELFIES VAE(Pytorch-Lightning)
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python ./ChemVAE/ChemVAE_Main.py
```

### SMILESDataset data to SMILES
```console
smiles_batch = []

for data in batch:
  mol = crop_start_stop(data, self.smiles_language)
  mol = self.smiles_language.token_indexes_to_smiles(mol)
  mol = self.smiles_language.selfies_to_smiles(mol) if self.selfies else mol
  smiles_batch.append(mol)
```

# PaccMann_rl generator(train_conditional_generator.py : Not implemented by Pytorch-Lightning)
```console
(paccmann_sarscov2) $ for protein_id in (seq 40)
  python ./Reinforce/Reinforce_Main.py \
  /raid/paccmann-covid/models/SELFIESVAE \
  /raid/paccmann-covid/models/ProteinVAE \
  /raid/paccmann-covid/models/affinity \
  /raid/paccmann-covid/data/training/merged_sequence_encoding/uniprot_covid-19.csv \
  /raid/paccmann-covid/params/conditional_generator.json paccmann_sarscov2 $protein_id \
  /raid/paccmann-covid/data/training/unbiased_predictions --tox21_path \
  /raid/paccmann-covid/models/Tox21
  end
```
