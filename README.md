#Introduction
This is our submission to the Vision Capsule Endoscopy Challenge 2024 hosted by MISAHUB. Capsule Endoscopy is a wireless endoscopy that results in 70,000 to 100,000 image frames. The doctore is then required to examine each frame to determine the ailment that the patient is suffering. This task is therefore extremely monotonous, time-consuming and prone to human-errors. The VCE challenges to solve this problem by having the participants train a Deep Learning model that is able to determine the frames that are of interest to the doctor and also classify the frame as having one of the multiple types of diseases possible in the GI tract. There are total 10 classes of diseases covered in this challenge:
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
