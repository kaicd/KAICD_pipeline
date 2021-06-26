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


# PaccMann_rl generator
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python ./main/pl_train_generator.py
```

### affinity predictor
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python ./main/pl_train_affinity.py
```

### toxicity predictor
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python ./main/pl_train_toxicity.py
```

### protein VAE
It will be implemented soon...

### SELFIES VAE
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python ./main/pl_train_selfies.py
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
