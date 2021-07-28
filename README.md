### Download data and pretrained models

Download the [data](https://ibm.ent.box.com/v/paccmann-sarscov2-data) as reported in the [requirements section](#requirements).
From now on, we will assume that they are stored in the root of the repository in a folder called `data`, following this structure:

```console
data
├── pretraining
│   ├── ChemVAE
│   ├── ProtVAE
│   ├── Predictor
│   ├── Toxicity
│   └── language_models
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
(paccmann_sarscov2) $ PYTHONPATH='./' python Main.py --model Predictor --mode BA(or DS)
or 
(paccmann_sarscov2) $ PYTHONPATH='./' python Main.py --model Toxicity
```
