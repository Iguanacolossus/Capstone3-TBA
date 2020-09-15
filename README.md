# **Identifing Seizures From Wrist Mounted Sensors**



> - data can be found here: http://www.timeseriesclassification.com/description.php?Dataset=Epilepsy<br>
>- The data was generated with healthy participants simulating the class activities of performed. Data was collected from 6 participants using a tri-axial accelerometer on the dominant wrist whilst conducting 4 different activities: walking, running, sawing, and mimicing seizure. 


## MVP:
create model to cluster accelerometr data into the k movement groups. 

## Other things:
Use a neural network to classify seizure not seizure movements.

## Data Pre-Processing
why did i choose to convert to poLAR AND CHOOSE rho. yada yada.
![coord compare](images/compare_coords.png)
<br>

## Model and metrics
kmeans time eries . say a biv about this

**distance metric**<br>
say why DTW

**determining how many clusters are in data set**<br>
![elbow](images/elbow_dtw.png)
