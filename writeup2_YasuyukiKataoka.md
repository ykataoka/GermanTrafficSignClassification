#**Traffic Sign Recognition** 

## Report written by Yasuyuki Kataoka

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


This provides a Writeup / README that includes all the rubric points and how you addressed each one. 

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[img1]: ./figure/label_data_num.png "Data Exploration"
[img2]: ./figure/sample_color2gray.png "Sample data of color conversion"
[img3]: ./figure/confusion_matrix_before.png "Confusion Matrix"
[img4]: ./figure/sample_webdata2.png "new sample of german signals"
[img5]: ./figure/feature_map.png "Sample of Feature Map"
[img6]: ./figure/softmax_example.png "example of softmax function"

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

## Main Contents
First, here is a link to [my solution](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) to traffic sign classification problem.

### Data Set Summary & Exploration

#### 1. basic summary of the data set
I used the pandas library to calculate the number of dataset for each classes.
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

This part is not changed from the sample file
(The **second** code cell of the IPython notebook contains this code.)

#### 2. Include an exploratory visualization of the dataset

Here is an exploratory visualization of the data set.
It is a bar chart showing how many data is in training data for each label.

![hoge][img1]

(The **third** code cell of the IPython notebook contains this code.)

### Design and Test a Model Architecture

#### 1. Data Augmentation

I decided to generate additional data because the class which has less data tends to be misclassified due to the imbalanced data for some labeled data. 
The target label should be wisely chosen based on the 1st result shown later on.

To add more data to the the data set, I implemented data augmentation functionality to get more dataset.
The idea is simple random rotation.
Rotation should give more robustness to the camera's posture.

If the tflearn library is available(only available tensorflow v1.0, maybe?), we can use more library function, such as blur. For this project, I simply implemented one augmentation way.

I added 4 times dataset for each target labels.
My final training set had 44519 number of images, while the initial size is 37229.

(The **fourth** code cell of the IPython notebook contains the code for augmenting the data set.)


#### 2. Pre Process the Data Set

I used the conventional technique,
* converting to gray scale to RGB for reducing the size.
* normalization the value from [0, 255] to [-1, 1] for reducing the redundant computation of optimizer
* shuffling the data in the end to avoid the effect of initial order of original dataset

This process is implemented as the pipeline so I or others can easily add more processing module.
Here are examples of a traffic sign image before and after grayscaling.

![alt text][img2]

(The **fifth** code cell of the IPython notebook contains this code.)

#### 3. DNN Model
The code for my final model is located in the **seventh** cell of the ipython notebook. 

My final model consisted of the following layers based on the LeNet:

| Layer         	|     Description				| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x3 RGB image   				| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU			|						|
| MaxPooling		| 2x2 patch, 2x2 stride, outputs 14x14x18	|
| Dropout		| keep_prob = 0.75	 	 		|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x54 	|
| RELU			|						|
| MaxPooling		| 2x2 patch, 2x2 stride, outputs 5x5x54		|
| Dropout		| keep_prob = 0.75	 	 		|
| flatten		| 1350 (= 5x5x54)				|	
| full connect		| 1350 -> 300					|
| RELU			|						|
| Dropout		| keep_prob = 0.75	 	 		|
| full connect		| 300 -> 100					|
| RELU			|						|
| Dropout		| keep_prob = 0.75	 	 		|
| full connect		| 100 -> 10					|
| RELU			|						|
| softmax		| when loss computation				|

(The **sixth** code cell of the IPython notebook contains this code.)

#### 4. Training & My Approach to this Solution

The hyper parameters for the training process are

* Loss Function : cross entropy (to the result of softmax)
* optimizer : AdamOptimizer
* batch size : 128
* epoch : 20
* learning rate : 0.001

To train the model, I used mini-batch technique.
And, once new epoch starts, training set is reshuffled to avoid the order effect.

My 1st approach was 

* tweaked parameters LeNet by increasing the number of filters
* added Maxpooling (1. reduce the computation cost, 2.capture characteristic feature, 3.strong to the location variation, 4. avoid overfitting, )
* added dropout (avoid overfitting)
* increased epoch num as the model gets bigger and takes more time to train

This approach resulted in 93.3% at most by this approach.

Next, I evaluated **confusion matrix** to see which label data tends to be misclassified. Here is the result.

![alt text][img3]

The labels which have less than 80% accuracy classification are
[0, 16, 20, 21, 24, 27, 34, 40, 41]

Remark that all of these images are less data. They all have only 250ish images in training data!

Thus, I applied the data augmentation technique to these label data.

My 2nd approach is

* augment data for specific class

This approach resulted in 94.0% at most by this approach.


(The **seventh** code cell of the IPython notebook contains this code.)



#### 5. Evaluation & 

My final model results were:
* training set accuracy of 0.990
* validation set accuracy of 0.940
* test set accuracy of 0.919


(Also, the **seventh** code cell of the IPython notebook contains this code.)




The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.


If an iterative approach was chosen:
* Q. What was the first architecture that was tried and why was it chosen?

First architecture is orignal LeNet. Because it can be the benchmark.
Whenever we work on model refinement, we ought to have benchmark to make comparison.

* Q. What were some problems with the initial architecture?

Overfitting could be one of the problem due to the lack of dropout. When we want to increase the number of filters, the computation cost increases. Applying maxpooling actually helps to reduce the size of the model.

* Q. How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

In this project, usually the training accuracy was high. After several adjustments, I achieved training~0.99, validation~0.94, test~0.92, which would be okay quality.(please correct me if I am wrong.) The dropout or maxpooling techniques helped to avoid the overfitting.

* Q. Which parameters were tuned? How were they adjusted and why?

"size of the filter" to enhance the representation capability(feature extraction).
They are easily tweaked by changing the size of the output of each networks in tensorflow.

* Q. What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convolutional network can capture object, even if the location of the object is not fixed to center.
Also, it can capture the characterstic features(line, curve number) too.
Dropout helps to avoid the overfitting.

If a well known architecture was chosen:
* Q. What architecture was chosen?

AlexNet, ImageNet... I do not know which one is the best. What I suppose is that we can transfer the pre-trained model, if the lab
el data is similar.

* Q. Why did you believe it would be relevant to the traffic sign application?

no answer.

* Q.  How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

If the accuracy difference between training and validation (or test) set is big, that is overfitting. This is not the evidence that the mode works well to new data. If the accuracy is high and the difference is small, you can say the model works well

If the accuracy between validation and test set is big, this could be caused by how splitting is conducted. You may need to consider how to shuffle data again.


### Test a Model on New Images

#### 1. Working on new data from web.

**(Fixed here for resubmission, as I understood the intention of this exercise...)**

Here are five traffic signs from not only germany but also countries such as Japan or China.
They are found found on the web and cropped from original images.

![alt text][img4]

The size is randome, so resizing them is necessary.

(The **eighth** code cell of the IPython notebook contains this code.)


#### 2.Evaluation to web data

**(Fixed here for resubmission, as I understood the intention of this exercise...)**

The sample data(5 data) is 0.200 accuracy, while test data accuracy is 0.919.

Why this happened? Ther are several reasons why this did not work.

* Unknown sign

Some of the signs are non-germany. This means model created by German dataset can not classify them. Every country has different signs, so we need to collect dataset to accurately classify the signs.
For example, speed limit 40 is not included in the german dataset.
The first one is stop sign, but obviously the shape itself is different from germany version.

* Similar sign

The third one should mean 'share one road for both direction'.
This is somewhat similar to 'Road narrows on the right' in germany dataset.
Thus, it goes to the Road narrows on the right with high probability.

* Angle of the sign
As the fourth image shows, some of the sign might be rotated by yaw / roll angle.
This affects the accuracy.
When we do apply data augmentation, we need to consider this two different type of the rotation.
(usually, the library, e.g., tflearn, provides only roll angle.)

* Image is too vague / jittered

The first sign says 'stop' in Japanese though, we human may not be able to read it without the knowledge. same for the number. if the image is too vague, model misclassify them.

* Background Objects

If there is something in background, model may capture that features too. This may cause misclassification

The prediction result is [7  3 24  5 33], which corrensponds to 
['Speed limit (100km/h)', 'Speed limit (60km/h)', 'Road narrows on the right', 'Speed limit (80km/h)', 'Turn right ahead'], respectively.

The true label should be
y_web_name = ['Stop',
              'Speed limit (40km/h)', 
              'No Label in Germany Data', 
              'Speed limit (40km/h)', 
              'Turn right ahead']


(The **ninth and tenth** code cell of the IPython notebook contains this code.)


#### 3. Softmax prediction

The result of softmax ended up as follows.

| Probability         	|     Prediction     	       	       	      	| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         		| 'Speed limit (70km/h)'			| 
| 1.00    		| 'Road work' 					|
| 0.910			| 'Priority road'				|
| 1.00	      		| 'Turn left ahead'				|
| 0.998			| 'Speed limit (60km/h)'			|
| 1.00	      		| 'Yield'					|
| 1.00	      		| 'Ahead only'					|
| 1.00	      		| 'Stop'					|
| 1.00	      		| 'No entry'					|

Maybe, most of the dataset that I found is included in training set, as most of prediction constantly shows nearly 100.0% probability.
One exception could be the priority road as it goes down to 90% sometimes.

As an exmpale of the softmax distribution visualization, here I show the result of the "Priority road", which may not be included in the training set.
![alt text][img6]

Anyhow, the idea of topK probability works well on my code.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

(The **11th** code cell of the IPython notebook contains this code.)


#### 4. Feature Mapping

This is the visualization of the how 1st weights in the convolution NN perform to the input of the sign 'Turn Left Ahead'.
As you can see here, the shapes of cicleand arrow turning to left were obtained.
This is the huge advantage of deep learning, feature extraction.

![alt text][img5]

(The **12th** code cell of the IPython notebook contains this code.)
