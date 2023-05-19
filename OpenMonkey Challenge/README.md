# Open Monkey Challenge
https://openmonkeychallenge.com

## Data Preprocessing: 
The dataset used for this project was obtained from the Open Monkey Challenge, consisting of 17 landmark points to detect for each image. The data was divided into two parts: a training set and a validation set. The dataset was preprocessed to ensure that the images and landmarks were compatible with the neural network models (ResNet-50 and YOLOv8).

During the preprocessing stage, the images in both the training and validation sets were resized to fixed dimensions. For the ResNet-50 model, images were resized to 128x128 pixels, and for the YOLOv8 model, images were resized to 640x640 pixels. Corresponding landmark coordinates were scaled accordingly to match the resized image dimensions, maintaining the spatial relationship between the keypoints and the image.

A custom dataset class PoseEstimationDataset was created, which extended the torch.utils.data.Dataset class. This class was responsible for loading the images and corresponding landmarks, as well as performing the necessary preprocessing steps. The images were then converted to PyTorch tensors and normalized using the mean and standard deviation values of the ImageNet dataset. In this experiment, data augmentation techniques such as random cropping, flipping, or rotation were not applied.

## Model Architectures: 
Two different models were used for the comparative study: a pre-trained ResNet-50 model and a YOLOv8 model.
* ResNet-50: A pre-trained ResNet-50 model was utilized as the backbone for feature extraction in this pose estimation project. The model's original fully connected layer was removed, and a new fully connected layer was added to output 34 values (17 landmarks * 2 coordinates: x and y). This new layer served as the regression head for pose estimation, allowing the model to predict the 17 landmark points directly from the input image.
* YOLOv8: YOLOv8 is an anchor-free object detection model that reduces the number of box predictions and speeds up the Non-Maximum Suppression (NMS) process. YOLOv8 uses mosaic augmentation during training; however, this augmentation is disabled for the last ten epochs, as it has been found that this augmentation can be detrimental if used during the entire training process.


## Model Training and Optimization: 
Both models were trained using different configurations. The ResNet-50 model was trained for 5 epochs with a batch size of 256, while the YOLOv8 model was trained for 3 epochs with a batch size of 256. The Mean Squared Error (MSE) loss function was used as the criterion for measuring the difference between the predicted landmarks and the ground truth landmarks for both models. The Adam optimizer was employed for weight updates with a learning rate of 1e-4 for both models.
During the training loop, the models were set to training mode, and the optimizer's gradient values were reset to zero. The input images were passed through the models to obtain the predicted landmarks. The MSE loss between the predicted and ground truth landmarks was calculated, and the gradients were backpropagated through the models. Finally, the optimizer updated the model weights to minimize the loss.
After each epoch, the models' performances were evaluated on the validation set. The same MSE loss function was used to calculate the loss between the predicted and ground truth landmarks for the validation data.


## Model Evaluation:
To thoroughly evaluate the performance of both the ResNet-50 and YOLOv8 models on the validation set, several metrics were used:
* Mean Absolute Error (MAE): The average absolute difference between the predicted and ground truth landmark coordinates.
* Mean Squared Error (MSE): The average squared difference between the predicted and ground truth landmark coordinates.
* Mean Per Joint Position Error (MPJPE): The average Euclidean distance between the predicted and ground truth landmarks for all keypoints.
* Percentage of Correct Keypoints (PCK): The percentage of predicted keypoints that are within a certain threshold distance (e.g., 5% or 10%) from the ground truth keypoints.


## Results:

ResNet-50 Model:
* Mean Absolute Error: 0.2825
* Mean Squared Error: 5.6003
* Mean Per Joint Position Error: 15.1842
* Percentage of Correct Keypoints at 5% Threshold: 23.13%
* Percentage of Correct Keypoints at 10% Threshold: 54.42%

YOLOv8 Model:
* Mean Absolute Error: 0.0249
* Mean Squared Error: 3.6358
* Mean Per Joint Position Error: 67.2967
* Percentage of Correct Keypoints at 5% Threshold: 44.35%
* Percentage of Correct Keypoints at 10% Threshold: 64.33%

## Conclusion:

Based on the evaluation metrics, the YOLOv8 model outperforms the ResNet-50 model in terms of the Mean Absolute Error, Mean Squared Error, and Percentage of Correct Keypoints at both 5% and 10% thresholds. However, the ResNet-50 model has a lower Mean Per Joint Position Error compared to the YOLOv8 model.
In conclusion, the comparative study between the ResNet-50 and YOLOv8 models demonstrates that both models have their strengths and weaknesses in the task of pose estimation. While the YOLOv8 model has better overall accuracy in terms of detecting keypoints within a given threshold, the ResNet-50 model has a lower average error in terms of the distance between predicted and ground truth keypoints.
