# Basic unlearning method
This is our Capstone project on the topic of Machine Unlearning
There are total 3 different Unlearning methods including: bad_teaching and scrub which based distillation framework; neggra which directly uses negative gradient of forget samples.
Besides, we have 2 self-supervised methods to help unlearning and all of both based on contrastive learning.


# Self supervised
The simpler one is directly compare the features similarity between model and model pretrained by simclr. The original one is implementing constrative learning during unlearning process, i.e. augment dataset then
calculate similarity loss.

# Augment method
In order to handle the small size instance-wise unlearning we implemented 2 generative method to augment forget dataset, feature level opengan and arpl

To
