# ML-RashIO

## Summary 
RashIO is the aplication can detect skin diseases based on rashes or redness on the skin. We built our model using CNN, where Convolutional Neural Network (CNN) is one type of neural network commonly used for image data. CNN can be used to detect and recognize objects in an image. So, in this case is very match with the application we made. And we use pre-trained model with EfficentNetB0. 
EfficientNet is capable of a wide range of image classification tasks. This makes it a good model for transfer learning.

This repository mainly consist of 2 files : 
1. 'Capstone Project-EfficientNetB0-model12.ipynb' that the notebook files of this project 
2. 'label.txt' that used to store a list of classes or labels used in classification tasks.

## Capstone Project-EfficientNetB0-model12.ipynb 
### How to Make the Model? 
1. Load the dataset from the dataset link : https://drive.google.com/file/d/1WQ8MaVpx1Z01NG4NcesJ97e1aO7IeTto/view?usp=sharing using Python and the zipfile module to extract a zip file. 
2. sets up directory paths and counts the number of files in each directory

### Data Preprocessing for Modelling 
1. using the TensorFlow Keras ImageDataGenerator class to generate batches of image data for training and testing for efficiently loading and preprocessing large amounts of image data. he purpose of using ImageDataGenerator and the corresponding generators (train_generator and test_generator) is to dynamically load and preprocess the images in batches during model training and testing. 
2. obtain the class indices and labels associated with the training data generator and save them for reference. The printed class indices provide the mapping between the class names and their corresponding numeric labels. The saved label text file ('label.txt') can be used to understand the class labels used in the dataset.

### Modelling Process
1. building and training a classification model using the EfficientNetB0 as a base model.
2. The weights parameter is set to 'imagenet', indicating that the pre-trained weights of the model trained on the ImageNet dataset will be used.
3. `include_top` is set to False to exclude the fully connected layers of the base model.
4. `input_shape` is set to (224, 224, 3), specifying the input shape of the images expected by the model.
5. In this scope, the model uses ` Global average pooling, Dropout and Several dense layers`. The activation function used in the model is `ReLu`. The last activation function is `softmax` activation function to produce probability predictions for the three classes.
6. The model is compiled using the Adam optimizer with a learning rate of 0.0001.
7. The loss function is set to `categorical_crossentropy` as it is commonly used for multi-class classification tasks.
8. The training is performed for 200 epochs.
9. The history of the training is stored in the history variable.

### Evaluation 
1. performing evaluation and visualization on the trained model. The purpose is to visualize the training and validation accuracy, loss curves, and to evaluate the trained model's performance on the test dataset. Additionally, it calculates and displays the confusion matrix, which provides insights into the model's classification performance for each class.
2. demonstrates how to obtain a batch of images and labels from the test generator and make predictions on a specific image. The purpose is to demonstrate how to extract a specific image and its label from the test generator and make predictions on that image using the trained model. 
