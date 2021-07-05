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


# PaccMann Pipeline (Pytorch-Lightning)
## Model: DTI, Toxcity, ChemVAE, ProtVAE, Reinforce
### Run
```console
(paccmann_sarscov2) $ PYTHONPATH='./' python PaccMann_Main.py --config_filepath Config/paccMann.yaml
```
