# Stacked Capsule Autoencoder

This repository contains the code for "Stacked Capsule Autoencoder"

We trained and tested the model on two datasets: MNIST and ShapeNet(Partial)

# File structure
The folder contains a jupyter notebook that contains the code used for developinga dataloader for the shapenet dataset. The directory stacked_capsule_autoencoders contains modified code base of the official implementation of the Stacked Capsule Autoencoder from Google. For more details, refer to [README](stacked_capsule_autoencoders/README.md)

# How to use

To run experiments, following instructions, and run code from this directory. 

## Preliminaries

You must install following packages before you can run this code:
```
numpy, Pillow, tqdm, scikit-learn, matplotlib, opencv-python, glob, tensorflow, absl_py, imageio, monty, tensorflow_probability, tensorflow_datasets
```
The requirements can also be found in `requirements.txt` in this directory. 

## Train Capsule networks
### To train under `MNIST` dataset:
```python
python -m stacked_capsule_autoencoders.trai\
  --name=mnist_30_16  --model=scae  --dataset=mnist  --max_train_steps=300000  --batch_size=128  --lr=3e-5  --use_lr_schedule=True  --canvas_size=40  --n_part_caps=30  --n_obj_caps=16  --colorize_templates=True  --use_alpha_channel=True  --posterior_between_example_sparsity_weight=0.2  --posterior_within_example_sparsity_weight=0.7  --prior_between_example_sparsity_weight=0.35  --prior_within_example_constant=4.3  --prior_within_example_sparsity_weight=2.  --color_nonlin='sigmoid'  --template_nonlin='sigmoid'  "$@"
```

### To train under `ShapeNet` dataset:
```python
python -m stacked_capsule_autoencoders.train  --name=shapenet_30_16  --model=scae  --dataset=shapenet  --max_train_steps=300000  --batch_size=1024  --lr=3e-5  --use_lr_schedule=True  --canvas_size=128  --n_part_caps=30  --n_obj_caps=16  --colorize_templates=True  --use_alpha_channel=True  --posterior_between_example_sparsity_weight=0.2  --posterior_within_example_sparsity_weight=0.7  --prior_between_example_sparsity_weight=0.35  --prior_within_example_constant=4.3  --prior_within_example_sparsity_weight=2.  --color_nonlin='sigmoid'  --template_nonlin='sigmoid'  "$@"
```

## Test MoCo

To test on MNIST: 
```python
python -m stacked_capsule_autoencoders.eval_mnist_model\
  --snapshot=stacked_capsule_autoencoders/checkpoints/mnist_60_40/model.ckpt-132696\
  --canvas_size=40\
  --n_part_caps=60\
  --n_obj_caps=40\
  --colorize_templates=True\
  --use_alpha_channel=True\
  --posterior_between_example_sparsity_weight=0.2\
  --posterior_within_example_sparsity_weight=0.7\
  --prior_between_example_sparsity_weight=0.35\
  --prior_within_example_constant=4.3\
  --prior_within_example_sparsity_weight=2.\
  --color_nonlin='sigmoid'\
  --template_nonlin='sigmoid'\
  "$@"
```

To test on Shepnet: 
```python
python -m stacked_capsule_autoencoders.eval_mnist_model\
  --snapshot=stacked_capsule_autoencoders/checkpoints/shapenet_test/model.ckpt-2068\
  --dataset=shapenet\
  --canvas_size=128\
  --n_part_caps=60\
  --n_obj_caps=40\
  --colorize_templates=True\
  --use_alpha_channel=True\
  --posterior_between_example_sparsity_weight=0.2\
  --posterior_within_example_sparsity_weight=0.7\
  --prior_between_example_sparsity_weight=0.35\
  --prior_within_example_constant=4.3\
  --prior_within_example_sparsity_weight=2.\
  --color_nonlin='sigmoid'\
  --template_nonlin='sigmoid'\
  "$@"
```

