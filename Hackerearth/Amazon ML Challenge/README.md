# Data Screenshot
<img src="images/data_screenshot.png" width="800"/>


# A Description of our Methodology
<img src="images/Amazon_ML_Challenge.jpg" width="800"/>

### Feature Engineering & Approach

1. An initial k-fold cross validation of the numeric features were performed to find feature importance
2. Features "effectiveness_rating" and "number_of_times_prescribed" were found to be the most important
3. Using these two features, K-means was performed on the data and 80 clusters were generated. The distance of each sample from all cluster centroids was used as features
4. The drug and disease columns were one-hot-encoded
5. Some of the drugs had a different name in the test set, so they were replaced with the ones in the training data
6. All diseases that begun with "<\span>" were given a single variable in the one-hot-encoding scheme
7. Finally, all of the features were concatenated, resulting in a total of 3809 features
8. LightGBM model was used and it's hyper-parameters were set via k-fold cross validation

### Tools used
1. Python for programming
2. sklearn and numpy libraries for methodology
3. Tensorflow library for the final model
4. matplotlib and seaborn was used for plotting and analyzing the data


# Competition Result
1. Rank: **12th** on public and private LB
2. [Link to leaderboard](https://www.hackerearth.com/challenges/competitive/amazon-ml-challenge/leaderboard/)
