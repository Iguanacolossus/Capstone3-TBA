# Capstone3-TBA

## **1. Forcasting with ARIMA or RNN and prediction with linear regression:**
> ## forcasting short term solar generation from plant in india for better grid managment.
>-  https://www.kaggle.com/anikannal/solar-power-generation-data<br>
>This data has been gathered at two solar power plants in India over a 34 day period. It has two pairs of files - each pair has one power generation dataset and one weather sensor readings dataset. <br>

**my idea:** I am thinking that first i would make a regression model that takes in the weather sensor data and predicts the power generationfrom the plant. then I would forcast the weather data for 48 hours , then use that with my regression model to predict the energy generation for then next two days




## **2. classification of sequencial data with RNN (and maybe cluster if that makes sence):** 
>## Identifying seizures form wrist mounted accelerometers.
> - data can be found here: http://www.timeseriesclassification.com/description.php?Dataset=Epilepsy<br>
>- The data was generated with healthy participants simulating the class activities of performed. Data was collected from 6 participants using a tri-axial accelerometer on the dominant wrist whilst conducting 4 different activities: walking, running, sawing, and mimicing seizure. 

**motivation:** Having an app on your smart watch that could alert your family or 911 if you are having a seizure could help with anxiety of epileptic people and possible be life saving if person is in a dangerous situation. (I also want to try my hand at sequentioal data that is not NLP)

**my worries**: each sample/series consists of three channels (acceleraomiter directions in space).  the rnn examples in my books are either for sentiment  prediction with NLP or time series clasification with one 'channel'. I am assuming I can feed an RNN a tensor of shape (batch x 1 x lengths of series x num of channels)  but have not seen any examples . I read a couple scientific journal about 'human activity recognition' and  i found one that uses CNN. so if I go that direction im not sure if i should choose this becasue I did CNNs for my last capstone.




## 2. **unsupervised learning:**
> ### image clustering 
>http://www.cs.toronto.edu/~kriz/cifar.html

