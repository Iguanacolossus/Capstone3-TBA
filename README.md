# **Identifing Seizures From Wrist Mounted Accelerometer data**



The data for this project was generated with healthy participants simulating the class activities of performed. Data was collected from 6 participants using a tri-axial accelerometer on the dominant wrist whilst conducting 4 different activities including mimicking a seizure. The dat acan be found 
[here](http://www.timeseriesclassification.com/description.php?Dataset=Epilepsy). Each participant performed each activity 10 times at least. The mimicked seizures were trained and controlled, following a protocol defined by an medical expert. The sampling frequency was 16 Hz. Samples where truncated to the length of the shortest recording retained at about 30 seconds.


## **Data Pre-Processing**
I recieved the data in .arff format which I read in to my python environment with scipy.io.arff loadarff then converted to pandas for easier insepection and manipulation. Each row of the data was a sample of one physical activity. the signal was packed into list of  lists corisponding to the signal in the x, y and z directions. every sample was 206 time steps. There is also a column of the label for each activity. <br>
![raw](images/raw_data_pd.png)<br>
In the wild a this technology would be used in smart watches and I wanted to make sure that if a person were to ware there watch at a different angle on there wrist that the data still reflected the same acceleration. I compaired the data before and after 45% rotation aroung the z access. I observed that if the data was left in cartisien coordinants that that both x and y coordinants change when the acceleromter is rotated but if i converted the data to scherical coordiance, then oonly one of the coordinants changed. In hopes of keeping the most information from the data but still taking into consideration rotation of the accelrometer I converted the whole data set to scherical coordinance and did not use the theta coordinance moving forward.  <br> 
![coord compare](images/compare_coords_all.png)
<br>

## **A First Approach With Unsupervised Learning**
My fisrt thought was that if the seizure activity was unique enough from the other activity it would be very handy to be able to cluster it out from other activities so this model could be used on different data with other unknow activities.  

## Unsupervised Model 
To perform my clustering I wanted to pick a model that would utilize the time information and pattern as apposed to feature engineering that might take some domain spacific expertise. I utalised a model class from a Python package called tslearn that build on scikit-learn, numpy and scipy. For this clustering model I used the TimeSeriesKmeans clustering algorithm from th emodule tslear.clustering. FOr this model only one channel of accelrerometer dat could be used as input so I chose to use the rho coordinant in this model.

## Distance Metric Used in Model
To find the distance or similarity of time series samples I chose to use the inter model distance metric  Dynamic Time Warping. DTW deals with time shifts in such a way that alow th emodel to compair signal patterens even if the time or phase is shifted. Also, cluster centers are computed as the barycenters with respect to DTW, hence they allow to retrieve a sensible average shape whatever the temporal shifts in the cluster.

## Determining how many different activities the model pick oout of the data.
Below is a plot of 9 models with k hyperparameters ranginf from 1 to 9.<br>
![elbow](images/elbow_dtw.png)

This plot shows that the optimal number of clusters is 2,3 or 4. I will now make silhouette plots to compare the kmeans clustering model with 2 to 4 clusters. The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation).<br>
## Evaluate the model a diferent choses for cluster number
![km_sil_2](images/sil_plot_c2.png)
![km_sil_3](images/sil_plot_c3.png)
![km_sil_4](images/sil_plot_c4.png)


The fisrt plot is of the model with a k assignment of 2. Since this model has the best cohesion and speratin of clusters, it may be picking up on a seizure cluster and a not seizure cluster.  Lets take a look at the ratio of cluster assignents on the true seizure data.<br>
>26.5% of seizure data in one cluster and 75.5% of seizure data assigned to the other cluster. This means that with the time series clustering aproach this model would miss over 25% of seizures in people.<br>
I would like to make this better so I did some further exploration.

## More eploration

On further exploration it looks like the seizure data can vary greatly in it aplitude. In the figure below the orange signal is one center from the 2 k clustering model above, the orange signal is the other cluster center. The gray signals are two seperate seizure events.<br>
![compare](images/raw_series_comparison.png)<br>
My next step in making my model better is to normalise each sample such that the values are between 0,1.  My hope is to have the clustering model focus on the changes in apletude and pattern as aposed to the ampletude its self.<br>

## Same Aproach but With Normalized Signals

Here is a new elbow plot for the normalized data set.<br>
![normelbow](images/elbow_dtw_normalized.png)<br>
This elbow in this plot is even more elusive then the previouse one on the un-normalized data set. this makes me think that this is not the right direction to go. <br>

This non elbow may be due to the fact that the amplitude is useful in differentiating the clusters. <br>

<!--This leads me to the idea of pivoting from a time series clustering model to  makeing a dataframe of extracted features from each time series . then clustering with not time series methods.<br> -->

## Clustering extracted features
??

The clustering approach to identifying features May not be fiezable with this data set. A supurvised learning approach may have to be used in the field so next I will pivot to that.<br>

## **Classification with CNN**
In order to preserve the pattern of the signal for the classification process I want to try a tehnique from Jiang and Yin (2015). They proposed to transform the 1D acceleration signal, into a (1 x time x channel) image like tensor. This way the data can be model wit a convelutional neural network wich are very good at picking up patterns. 

## Data preprocessing
Here, I use the rho and phi channels of my accelerometer data. I used numpy to reshape the data set into (n_samples x series_length x n_channels) shapped tensor which came out to be (137 x 206 x 2)

I performed a validation train test split at 25/75 ratio for fitting. 

## Architecture
I chose this becasue it is similar to leNet, A classic simple architecture with two convolution layers, max pooling, and drop out, and two dense layers. I also chose one nueron and sigmoid activation for the output layer because I wanted to make a model that would predict seizure or no seizer in a binary manner.<br> 
![summary](images/model1_summary_2channel.png)

## Metrics
I chose to use recall as my primary metric in evaluating my model becasue the classes are imbalenced and a False negative could be life threatening. It would be better for a family member to contact a person with a false alarm over the persons family to not be contacted at all during a seizure event.

## How The CNN  Learned
![lr_plot](images/lr_plot_model1_rhophi_300epochs.png)

## How Can I Make This Better?
Upon researching how i could imporov my score with this CNN arcitecture I came across a thesis paper by Xiaoyo Yu on Human activity recovnition that referenced an idea from Jiang and Yin (2015) who converted there 1 demensional time series  data into 2 dementional descrete fourier transformations. Using this idea I decided to make my own 2d fourier transformation by take the fast forier transformation and sliding windows of time for each sample to make a spectragram like the ones use in audio data analysis.  <br>


For each channel of each time series sample I created a spectrogram. an example of a sample and its two channels in below. <br>
![s](images/spectrogram_sample2.png)<br>
A spectrogram is used in speech and music analysis becasue its a way to visualize the fourier transformation as time passes.

I then reshaped my data set to be (n_samples, spec_height, spec_width, n_channels) in preporation to fit with a CNN. I usedf almost the exact architecture for the last 1D CNN with some changes for 2D convolution.

## How did it learned
![spectrogram_lr](images/model4_specs_300.png)
Since the validation loss is goin up and up , instead of more training time I will make the architecture more complex (more trainable parameters) to see if it will help the model by extracting more feature out of the images.
<br>

## New Architecture
![deeper1](images/deeper1_sumary.png)<br>
![depper1_lr](images/model_deeper1_specs_300.png)

This deeper model did not do better interms of recall score. As far as loss, It did worse learning even with of double the amount of trainable parameters. After talking to an assosiet about audio analysis with spectrograms I have concluded that the reason why my spectrograms are doing so poorying maybe do to the fact they are that they are very small. I do not know that the sample rate or length of samples was used in the  Jiang and Yin (2015) paper. but compair to music audio data that utilize spectrogram modeling, this accelerometer sampling rate is 3000 times lower leading to much less time windows to map to the spectrogram. 

## LSTM

## Ensemble LSTM 1d CNN
Combining the predictions from multiple neural networks adds a bias that in turn counters the variance of a single trained neural network model. The results are predictions that are less sensitive to the specifics of the training data, choice of training scheme, and the serendipity of a single training run.


## Results 
confusion matrix


## moving forward
