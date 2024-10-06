# Digit_Detection_MNIST

This code is a complete machine learning workflow for classifying handwritten digits from the MNIST dataset using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## 1. Import Libraries: 
Various libraries like Pandas, NumPy, and Matplotlib are imported for data manipulation and visualization. TensorFlow and Keras are used for building and training the CNN model. The code also suppresses TensorFlow logs to display only errors.

## 2. Dataset Loading: 
The training and testing datasets are loaded from CSV files containing the MNIST dataset, where each image is represented by pixel values.

## 3. Data Preprocessing: 
The features (pixel values) and labels (digits) are extracted from the datasets. The feature data is reshaped into 28x28 pixel images with a single channel (grayscale), and normalized to have values between 0 and 1. Labels are converted to one-hot encoded vectors, allowing the model to handle them as categorical data.

## 4. Building the CNN Model: 
A custom CNN model is built using the Sequential API. The model has an input layer, followed by two convolutional layers to extract features, each followed by a pooling layer to reduce the feature map size. The features are then flattened and passed through a dense fully connected layer for learning non-linear combinations of features. Dropout is applied to prevent overfitting. Finally, the output layer uses the softmax function to classify the digit into one of ten possible classes (0-9).

## 5. Model Compilation: 
The model is compiled using the Adam optimizer, which helps in efficient gradient descent, and categorical cross-entropy as the loss function, which is suitable for multi-class classification problems.

## 6. Model Training: 
The model is trained on the training dataset for 10 epochs, with a batch size of 32. During training, 20% of the training data is used for validation to monitor the model’s performance and prevent overfitting.

## 7. Model Evaluation: 
After training, the model is evaluated on the test dataset to determine its accuracy. The accuracy score indicates how well the model generalizes to unseen data.

## 8. Predictions Visualization: 
The model makes predictions on the test set, and a function is used to visualize a sample of predictions. The function plots some of the test images along with their true labels and the model's predicted labels, providing a visual check on the performance of the model.

Overall, the workflow follows the steps of data preparation, building, training, evaluating, and visualizing the results of a CNN model for classifying handwritten digits. The highest achieved accuracy provides insight into how well the model has learned the characteristics of different digits.
