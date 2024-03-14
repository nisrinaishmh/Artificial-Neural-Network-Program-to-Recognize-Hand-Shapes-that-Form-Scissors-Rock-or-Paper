# Machine Learning for Beginners : Artificial Neural Network Program to Recognize Hand Shapes that Form Scissors Rock or Paper by DICODING🌈🪄 

## I have successfully completed the Rock, Paper, Scissors project for Dicoding Machine Learning. The detailed explanation of this project is as follows:

### Project Steps📝📂:

1. Library Import: You started by importing essential libraries such as TensorFlow, ImageDataGenerator, and other libraries needed to build and train a machine learning model.

2. Dataset Import: The dataset containing images of hands forming Rock, Paper, and Scissors was downloaded from Dicoding Academy. It was then extracted using the zipfile function.

3. ImageDataGenerator: ImageDataGenerator was used to perform data augmentation. Data augmentation aims to increase artificial data with transformations like rotation, shifting, and shear. This helps the model to be more generalizable and less prone to overfitting on the training data.

The parameters used in ImageDataGenerator include:

* rescale: scales the pixel values of the image from [0-255] to [0-1].
* rotation_range: randomly rotates the image within a specific range (degrees).
* width_shift_range, height_shift_range: randomly shifts the image on the horizontal and vertical axes.
* shear_range: randomly bends the image.
* zoom_range: randomly zooms in or out (reduces) the image.
* horizontal_flip: randomly flips the image horizontally.
* validation_split: splits the dataset into a training set (60%) and a validation set (40%).

4. Splitting the Dataset: ImageDataGenerator was then used to create generators for the training and validation sets. The training set was used to train the model, while the validation set was used to evaluate the model's performance during training.

5. Sequential Model: You built the model using a sequential approach, where layers were added sequentially. The model consists of:

⚙️Convolutional Layer (Conv2D): This layer extracts spatial features from the image. In this project, you used 3 Conv2D layers, each with 32, 64, and 128 filters. Each layer was followed by MaxPooling2D to reduce the data dimension and prevent overfitting.

⚙️Flatten Layer: This layer converts the data from a 3-dimensional tensor to a 1-dimensional tensor before entering the hidden layer.

⚙️Dense Layer (Dense): The Dense layer acts as a fully-connected layer. You used 2 Dense layers with 512 and 256 neurons each to learn high-level representations of the features extracted by the convolutional layer.

⚙️Dropout Layer (Dropout): The Dropout layer is used to prevent overfitting by randomly disabling neurons during training. You used dropout with a rate of 0.5.

⚙️Output Layer (Dense): The output layer uses the softmax activation function to predict the probability of the three classes (Rock, Paper, Scissors).

6. Compile Model: The model was compiled with the Adam optimizer, categorical_crossentropy loss function, and accuracy metrics. The Adam optimizer helps to accelerate the training process to find optimal weights. Categorical crossentropy is used as the loss function for multi-class classification problems. Accuracy is used as a metric to measure the model's performance.

7. Early Stopping: The EarlyStopping callback was used to automatically stop training when the accuracy on the validation set did not improve significantly for 5 consecutive epochs. This prevents overfitting and saves training time.

8. Model Training: The model was trained using data from the training set. You used a batch size of 32 and 100 epochs. During training, the accuracy on the training and validation sets was monitored.

9. Save Model: After training was completed, the model was saved with the name "rps_model.keras".

10. Total Training Time: The time required to train the model for 100 epochs was calculated and displayed in minutes.

11. Plot Accuracy and Loss: Plots of accuracy and loss on the training and validation sets were created using the Plotly library. These plots help to visualize how the model's performance changes during training. Ideally, the accuracy on the training set should increase and the loss should decrease with the epoch. The accuracy on the validation set is expected to be not too different from the accuracy on the training set, which indicates that the model is not overfitting.

12. Model Testing: To test the model, you can upload images of hands forming Rock, Paper, or Scissors. The model will predict the class of the image.

### Project Result💎:
1) Total training time: 17.65 minutes
2) Model Training accuracy is 95%
