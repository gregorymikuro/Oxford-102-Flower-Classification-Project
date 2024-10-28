# 102 Category Flower Classification Project

## Overview
This project focuses on building an image classification model for recognizing different types of flowers using the **102 Category Flower Dataset**. The dataset, curated by Maria-Elena Nilsback and Andrew Zisserman, consists of 102 flower categories, which are commonly occurring flowers in the United Kingdom. Each class contains between 40 and 258 images, resulting in a challenging task due to variations in scale, lighting, pose, and intra-class diversity. The goal of this project is to leverage deep learning techniques to accurately classify these flower categories.

### Dataset Contents
The dataset is organized as follows:
- **jpg/**: Contains the images of the flowers in JPEG format.
- **segim/**: Contains the segmentation masks for each flower.
- **102flowers.tgz**: Original compressed file containing flower images.
- **102segmentations.tgz**: Original compressed file containing segmentation masks.
- **distanceMatrices102.mat**: Contains chi-square distances used in the ICVGIP 2008 publication.
- **imagelabels.mat**: Contains labels for each image.
- **setid.mat**: Specifies the training, validation, and test splits for the dataset.

## Project Structure
The directory structure of the project is as follows:
```
Flower-Dataset/
  ├── data/
  │     ├── jpg/
  │     ├── segim/
  │     ├── 102flowers.tgz
  │     ├── 102segmentations.tgz
  │     ├── distanceMatrices102.mat
  │     ├── imagelabels.mat
  │     └── setid.mat
  └── index.ipynb
```
- **data/**: Contains all dataset-related files.
- **src/**: Contains all the scripts for preprocessing, data loading, model training, evaluation, and utility functions.

## Getting Started

### Prerequisites
To set up and run the project, ensure you have the following installed:
- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Pandas
- OpenCV
- Pillow
- Matplotlib
- Scikit-learn

### Setting Up the Dataset
1. **Extract Data**: The `102flowers.tgz` and `102segmentations.tgz` files need to be extracted to access the images and segmentations.
2. **Load Labels & Splits**: The `.mat` files (`imagelabels.mat` and `setid.mat`) provide the image labels and dataset splits. These can be loaded using `scipy.io` for further use in training and evaluation.

### Project Workflow
1. **Data Preprocessing**:
   - **Image Resizing and Augmentation**: Images are resized to a consistent size of 224x224 pixels, and data augmentation (rotation, flipping, cropping) is applied to enhance model generalization.
   - **Normalization**: Images are normalized using mean and standard deviation values typical for pre-trained models like ResNet.

2. **Dataset Creation**:
   - A custom dataset class is used to manage loading and processing of images and labels.

3. **Model Training**:
   - A pre-trained **ResNet18** model is used for transfer learning. The final fully connected layer is replaced with a layer for classifying 102 classes.
   - The model is trained using cross-entropy loss and an optimizer like Adam.

4. **Training Loop**:
   - The training loop iteratively trains the model, computes the loss, and optimizes weights using backpropagation. The model can be trained on a GPU if available.

5. **Evaluation**:
   - The model is evaluated using accuracy metrics on validation and test sets.
   - A **confusion matrix** is used to visualize model performance across all 102 classes, highlighting areas where the model may struggle.

## Results
- The model achieves **top-k accuracy** to measure how often the correct label is in the top predictions.
- A confusion matrix is used to analyze which classes are confused with each other, providing insights for further improvements.

## Deployment
Once the model is trained, it can be deployed as a web API using **Flask** or **FastAPI**.
- The deployment allows for real-time predictions by uploading an image of a flower, which the model then classifies.

## Challenges and Future Improvements
- **Intra-class Variation**: Some classes have significant variations due to changes in scale, lighting, and pose.
- **Class Imbalance**: Techniques like oversampling or using class weights are recommended for improvement.
- **Advanced Augmentation**: Utilizing techniques like **Color Jitter**, **CutMix**, or **MixUp** can help improve model robustness.

## References
- Nilsback, M-E. and Zisserman, A., "Automated flower classification over a large number of classes," Proceedings of the Indian Conference on Computer Vision, Graphics, and Image Processing (2008).
- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

## Acknowledgments
- Thanks to Maria-Elena Nilsback and Andrew Zisserman for creating the dataset.
- Inspired by the PyTorch transfer learning tutorial.

---
Feel free to contribute to this project by creating a pull request or submitting issues on GitHub.

