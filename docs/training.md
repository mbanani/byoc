# Training a model

We use the hydra configuration system in this work. 
To train a model, one simply needs to define an experimental config. A few experimental configs are
provided in `byoc/configs/experiment`. Please note that you could overwrite model or dataset options
within the experiment config. 

Once the experiment config is defined, one could run the experiment by running the following
command. 

```bash 
# assuming byoc/configs/new_experiment.yaml
python train.py +experiment=new_experiment
```


