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

## Main(Must enter the model name)
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python Main.py --model Reinforce
or
(paccmann_sarscov2) $ PYTHONPATH='./' python Main.py --model ChemVAE
or
(paccmann_sarscov2) $ PYTHONPATH='./' python Main.py --model ProtVAE
or
(paccmann_sarscov2) $ PYTHONPATH='./' python Main.py --model Predictor
or
(paccmann_sarscov2) $ PYTHONPATH='./' python Main.py --model Toxicity
```

### reinforce generator(Pytorch-Lightning)
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python ./Reinforce/Reinforce_Main.py
```

### selfies VAE(Pytorch-Lightning)
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python ./ChemVAE/ChemVAE_Main.py
```

### protein VAE
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python ./ProtVAE/ProtVAE_Main.py
```

### affinity predictor(Pytorch-Lightning)
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python ./Predictor/Predictor_Main.py
```

### toxicity predictor(Pytorch-Lightning)
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python ./Toxicity/Toxicity_Main.py
```
