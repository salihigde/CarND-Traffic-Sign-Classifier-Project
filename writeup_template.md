# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[random]: ./writeup_images/random-images.png "Random Images"
[random2]: ./writeup_images/random-images2.png "Random Images"
[testimg]: ./writeup_images/test-images.png "Test Images"
[modified]: ./writeup_images/rotated-and-scaled.png "Rotated And Scaled Images"
[hist]: ./writeup_images/horizontal-hist.png "Histogram Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy array's 'shape' method to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34799
* The size of the validation set is 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: 32x32 with 3 color channels
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.
Here is some random images from the dataset. Same 3 images are converted and normalized and re-printed below to compare the difference.

![alt text][random]

Here is an exploratory visualization of the classes in data set. It is a horizontal bar chart showing each class name with the count of images they have

![alt text][hist]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale and normalized the images by using (img - 128)/128 formula because it was recommended in the project itself. After doing a bit more research and printing out the images I realized that there is a lot difference between original and grayscale + normalized image. Specially the dark images looks much better after normalization. 

Here is some examples of a traffic sign images before and after grayscaling with normalization.

![alt text][random2]

I decided to generate additional data because again after doing a research I found out that different variations of same image would help improving the results. Additionally since we were adding more data to our dataset that we already know what the class of the images, it was a easy way to increase our data amount (easier that handpicking and finding class of the traffic sign) and let our model to work with more data.

To add more data to the the dataset, I used the rotation and scaling techniques because I found it helped in improving the performance of my model. I mean from rotation is for each class I picked 60 image and did a random rotation of +/- 25 with additional scaling. This way I increased my data amount and had different angle and scaled images of same classes that I have.

Here is an example of an original image and an augmented image:

![alt text][modified]

The difference between the original data set and the augmented data set is around 2580 amount of images.  


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale, normalized image   		| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten   	      	| 5x5x16,  outputs 400      					|
| Fully connected		| 400, outputs 120								|
| RELU					|												|
| DROPOUT				| 0.5 keep probability							|
| Fully connected		| 120, outputs 84								|
| RELU					|												|
| DROPOUT				| 0.5 keep probability							|
| Fully connected		| 84, outputs 43								|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I applied similar way of training that was in LeNet course. Picked AdamOptimizer with 200 batch size and 30 epochs. Finally learning rate I picked is 0.001. Different than the LeNet course I found that increasing the epoch and batch size gave me better accuracy results.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.3
* validation set accuracy of 94.9
* test set accuracy of 93.3%

If an iterative approach was chosen:  
* What was the first architecture that was tried and why was it chosen?  
*Firstly, I tried the exact same architecture which we saw in LeNet course. My purpose of choosing this was to try what accuracy would I get without any modifications. By this way, I got around 89 percent accuracy in validation set.*

* What were some problems with the initial architecture?  
*In current architecture I didn't see any issues but it was basically not enough for getting good enough percentage in accuracy. After doing some research and revisiting the courses I decided to use dropout right after the two fully connected layers and re-runned the training process. This way with grayscale, normalization and adding more data I got 94.9 percent of accuracy in validation set.*

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.  
*I choosed to add two layers of dropout with 0.5 of keep probability which helped me to increase my validation data set percentage to 94.9 percent. *

* Which parameters were tuned? How were they adjusted and why?  
*I didn't adjust any parameters in my architecture.*

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?  
*Convolution layer works well with images, 3x3 or 5x5 filters do computations in small pieces of the image and keep sliding until it covers all of the image. this way our network can focus on details, edges, curves of the image and produce output regarding to this to be an input to our next layer.

*Dropout with 0.5 keep probability drops out half of the training set and may result our network to learn more robust features by letting it focus on random features this way in my case validation accuracy increased. In addition to this, dropout also prevents over-fitting.*

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][testimg]

I didn't face any classification difficulties in the images that I pick only 'Right-of-way at the next intersection' had relatively lower accuracy level than others because the image itself was kind a blurry.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image							 		|Prediction								| 
|:-------------------------------------:|:-------------------------------------:| 
| Stop									| Stop   								| 
| No entry     							| No entry 								|
| Yield									| Yield									|
| No entry	      						| No entry					 			|
| Keep right							| Keep right 							|
| Children crossing						| Children crossing 					|
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|
| Keep right							| Keep right      						|


The model was able to correctly guess all the images correctly with 100% accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

As it can be seen from the results below the predictions for the picked images were pretty accurate.

Original Image:  Stop  
Predictions:  
  0.99998354911804199219% - Stop  
  0.00001205563239636831% - Keep right  
  0.00000275256638815335% - Turn left ahead  
  0.00000049292839321424% - Yield  
  0.00000048047490963654% - Speed limit (30km/h)  
  
Original Image:  No entry  
Predictions:   
  0.99964046478271484375% - No entry  
  0.00033540723961777985% - Stop  
  0.00001728532697597984% - Keep right  
  0.00000680716766510159% - Speed limit (30km/h)  
  0.00000004669956155112% - Turn left ahead  
  
Original Image:  Yield  
Predictions:   
  1.00000000000000000000% - Yield  
  0.00000000000000005654% - No vehicles  
  0.00000000000000000850% - Priority road  
  0.00000000000000000032% - Ahead only  
  0.00000000000000000003% - Road work  
  
Original Image:  No entry   
Predictions:   
  1.00000000000000000000% - No entry  
  0.00000003830985662034% - Stop  
  0.00000000422651913468% - Speed limit (30km/h)  
  0.00000000018174833860% - Keep right  
  0.00000000001740516932% - Priority road  
  
Original Image:  Keep right  
Predictions:   
  1.00000000000000000000% - Keep right  
  0.00000000000000017106% - Dangerous curve to the right  
  0.00000000000000001143% - Traffic signals  
  0.00000000000000000075% - Turn left ahead  
  0.00000000000000000002% - General caution  

Original Image:  Children crossing  
Predictions:   
  1.00000000000000000000% - Children crossing  
  0.00000000000724487831% - Dangerous curve to the right  
  0.00000000000025738900% - Go straight or right  
  0.00000000000000187509% - End of no passing  
  0.00000000000000099337% - Slippery road  

Original Image:  General caution  
Predictions:   
  0.99999952316284179688% - General caution  
  0.00000042827747392948% - Traffic signals  
  0.00000000001601610858% - Pedestrians  
  0.00000000000007258879% - Go straight or left  
  0.00000000000000330775% - Roundabout mandatory  
  
Original Image:  Children crossing  
Predictions:   
  0.98498904705047607422% - Children crossing  
  0.01449771691113710403% - Slippery road  
  0.00029708424699492753% - Dangerous curve to the right  
  0.00021184467186685652% - Bicycles crossing  
  0.00000338463223670260% - Road narrows on the right  
  
Original Image:  Right-of-way at the next intersection  
Predictions:  
  0.97354954481124877930% - Right-of-way at the next intersection  
  0.02588187716901302338% - Beware of ice/snow  
  0.00051246094517409801% - Pedestrians  
  0.00002032580778177362% - General caution  
  0.00001693003468972165% - Double curve  
  
Original Image:  Keep right  
Predictions:   
  0.99699676036834716797% - Keep right  
  0.00299559161067008972% - Go straight or left  
  0.00000392966103390791% - General caution  
  0.00000322310211231525% - Stop  
  0.00000026217270487905% - No entry  

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


