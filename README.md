# Basic unlearning method
This is our Capstone project of Machine Unlearning
3 different Unlearning methods are included: bad_teaching and scrub which based on Knowledge Distillation framework; neggrad directly using negative gradient(gradient ascent) of forget samples.
Besides, we conduct 2 self-supervised methods to help unlearning, both of which are based on contrastive learning.


# Self supervised
The simpler one is directly comparing the features similarity between model and model pretrained by SimCLR. The original one is implementing constrative learning during unlearning process, i.e. augmenting dataset then
calculating similarity loss.

# Augment method
In order to handle the small size instance-wise unlearning query we implemented 2 generative method to augment forget dataset, feature level OpenGAN and ARPL and a simple augmentation method based on image simple adjustment such as adding some noise, random flips and so on.

# Run the experiment


## Retrain
##Instance-wise
```python
$ python mu_main.py --method retrain --mode random --saved_data_path mu/saved_data --lr 0.005 --epoches 2 --loss_weight 0
```

## Neggrad
##Instance-wise
```python
$ python mu_main.py --method neggrad --mode random --saved_data_path mu/saved_data --lr 0.005 --epoches 2 --loss_weight 0
```
## Basic salUN
##Instance-wise
```python
$ python mu_main.py --method salUN --mode random --saved_data_path mu/saved_data --lr 0.0013 --epoches 2 

```
## Basic bad-teaching
##Instance-wise
```python
$ python mu_main.py --method bad_teaching --mode random --saved_data_path mu/saved_data --lr 0.005 --epoches 2 --loss_weight 0

```
##Classwise-wise
```python
$ python mu_main.py --method bad_teaching --mode class_wise --lr 0.001 --epoches 1 --loss_weight 0

```
## Basic Scrub
##Instance-wise
```python
$ python mu_main.py --method scrub --mode random --saved_data_path mu/saved_data --lr 0.005 --epoches 2 --loss_weight 0

```
##Classwise-wise
```python
$ python mu_main.py --method bad_teaching --mode class_wise --lr 0.001 --epoches 1 --loss_weight 0

```

## With self-supervised
##Simple supervised
```python
$ python mu_main.py --method scrub --mode random --saved_data_path mu/saved_data --lr 0.005 --epoches 2 --loss_weight 0.5

```
##Original contrastive learning
```python
$ python mu_main.py --method scrub --mode random --saved_data_path mu/saved_data --lr 0.005 --epoches 2 --loss_weight 0.5 --supervised_mode original

```

## With data-augmentation
##ARPL
```python
$ python mu_main.py --method scrub --mode random --saved_data_path mu/saved_data --lr 0.005 --epoches 2 --loss_weight 0 --data_augment aplr --augment_num 3000

```

##OpenGAN
```python
<<<<<<< HEAD
$ python mu_main.py --method scrub --mode random --saved_data_path mu/saved_data --lr 0.005 --epoches 2 --loss_weight 0 --data_augment opengan
=======
$ python mu_main.py --method scrub --mode random --saved_data_path mu/saved_data --lr 0.005 --epoches 2 --loss_weight 0 --data_augment opengan
##Simple
<<<<<<< HEAD
$ python mu_main.py --method scrub --mode random --saved_data_path mu/saved_data --lr 0.005 --epoches 2 --loss_weight 0 --data_augment simple
=======
$ python mu_main.py --method scrub --mode random --saved_data_path mu/saved_data --lr 0.005 --epoches 2 --loss_weight 0 --data_augment simple
>>>>>>> e27ab431e10ff052b8a0cd24e8119eb29ce017e3

```

