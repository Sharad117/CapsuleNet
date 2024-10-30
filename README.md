# Introduction
This is our submission to the Vision Capsule Endoscopy Challenge 2024 hosted by MISAHUB. Capsule Endoscopy is a wireless endoscopy technique that results in 70,000 to 100,000 image frames. The doctor is then required to examine each frame to determine the ailment that the patient is suffering from. This task is therefore extremely monotonous, time-consuming and prone to human-errors. The VCE challenge aims to solve this problem by having the participants train a Deep Learning model that is able to determine the frames that are of interest to the doctor and also classify the frame as having one of the multiple types of diseases possible in the GI tract. There are total 10 classes of diseases covered in this challenge:
- Angioectasia
- Bleeding
- Erosion
- Erythema
- Foreign Body
- Lymphangiectasia
- Normal
- Polyp
- Ulcer
- Worms
For our solution, we attempted to train a CNN based model that is able to classify the frames into one of the 10 classes. Please check out our model weights and the corresponding paper below.

Do add the best_accuracy.ckpt from [here](https://drive.google.com/file/d/191zDIFdaZkmYwJUGG0N5AGj0-MRDjopF/view?usp=sharing)

Fig share [paper](https://figshare.com/articles/preprint/CapsuleNet_A_Deep_Learning_Model_To_Classify_GI_Diseases_Using_EfficientNet-b7/27297267?file=49974297)

arXiv [paper](http://arxiv.org/abs/2410.19151)
# Model
Our proposed model uses an EfficientNet B7 model from project MONAI as the backbone, followed by two hidden linear layers using PReLU activations. The output is a linear layer with softmax activation with 10 nodes.

<p align="center">
  <img src="metrics/capsulenet.png" alt="Proposed Efficientnet model" width="300"/>
</p>

# Training
Training and validation was done on the training and validation set provided by MISAHUB [here](https://doi.org/10.6084/m9.figshare.26403469.v1) The train set was extremely imbalanced, with over 28000 images for the largest class, while the smalles class (Worms) had just over 250 images. To address this issue we attempted using augmentation of the classes with lesser number of instances, and randomly sampled images from larger classes. We also attempted using Focal loss, and class weigts to address the imbalance issue, with poor results. The final model was trained with different augmentations for different classes.
- Erosion and Normal classes had 5000 images each (erosion was augmented, while Normal images were randomly sampled)
- Angioectasia and Polyp with 4000 images each (after augmentation)
- Worms with 1264 images (after augmentation)
- remaining classes had 3000 images each.

The loss and accuracy metrics on the training set were as follows: 
<p align="center">
  <figure style="display:inline-block; margin: 10px;">
    <img src="metrics/train_accuracy.jpg" alt="Image 1" width="45%"/>
    <figcaption style="text-align: center;">Train accuracy vs train steps
    </figcaption>
  </figure>
  <figure style="display:inline-block; margin: 10px;">
    <img src="metrics/train_loss_epoch.jpg" alt="Image 2" width="45%"/>
    <figcaption style="text-align: center;">Train loss vs train step</figcaption>
  </figure>
</p>


The loss and accuracy metrics on the validation set were as follows:
<p align="center">
  <figure style="display:inline-block; margin: 10px;">
    <img src="metrics/val_accuracy.jpg" alt="Image 1" width="45%"/>
    <figcaption style="text-align: center;">Validation accuracy vs Validation steps
    </figcaption>
  </figure>
  <figure style="display:inline-block; margin: 10px;">
    <img src="metrics/val_loss_epoch.jpg" alt="Image 2" width="45%"/>
    <figcaption style="text-align: center;">Validation loss vs Validation step</figcaption>
  </figure>
</p>


