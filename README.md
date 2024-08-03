# nnUNet_uncertainty-aware-learning

- This repository is based on nnUNet v2 (https://github.com/MIC-DKFZ/nnUNet) and serves as the backbone for our project. We have modified the inference and training stages to quantify aleatoric uncertainty and apply uncertainty-aware learning to a 3D multi-class intracranial hemorrhage (ICH) dataset for image segmentation tasks.

# Overview
- Our project aims to enhance the performance and reliability of neural networks in medical image segmentation by integrating uncertainty estimation techniques. By quantifying aleatoric uncertainty, we can better understand the confidence of the model in its predictions, particularly in the context of noisy or ambiguous data.

# Modifications
- Inference Stage:
We have implemented methods to quantify aleatoric uncertainty during the inference stage. This helps in identifying regions of high uncertainty in the predictions, which can be crucial for clinical decision-making.

- Training Stage:
The training process has been adapted to incorporate uncertainty-aware learning. By doing so, the model is trained to focus on reducing uncertainty in challenging areas, improving overall segmentation accuracy.

- Dataset
We used a 3D multi-class ICH dataset for this project. The dataset includes various types of intracranial hemorrhages, and our modifications aim to improve the segmentation of these different classes with better uncertainty quantification.

# Paper
- click [here](https://github.com/WeiHan00457030/nnUNet_uncertainty-aware-learning/blob/main/dissertation.pdf)