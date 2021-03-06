# Data Screenshot
<img src="images/data_screenshot.png" width="700"/>


# A Description of my Methodology

As the data was anonymized, there was no scope for feature engineering. It could be clearly understood from the data that some features had been standardized (treated as numeric) while others were not (treated as categorical). feature_8 was one-hot-encoded (ohe) and other features were left as is. I noticed that XGBoost worked better on ohe dataset while LightGBM handled categorical variables internally and required no ohe. RandomForest too was better off without ohe. I therefore maintained two separate datasets, one for XGBoost and the other for RandomForest and LightGBM. I tried reverse-engineering the features in order to derive some insight but it did not help. Also it was clear from plotting the data that none of the features were able to conclusively separate the positive and negative samples, so no variables were removed. Having finished with the initial analysis and data transformation, I shifted my focus on building better models. I developed two separate sets of ensembles.

Since the dataset was highly imbalanced, the first ensemble was built using cost-sensitive versions of the RandomForest and LightGBM classifiers. Specifically, each classifier’s hyper-parameters were fixed through 10 fold stratified CV and their probability outputs were combined through weighted average.

Due to the data imbalance and relatively large sample space, the second ensemble was performed on majority-class randomly undersampled data. This ensemble consisted of RandomForest and LightGBM classifiers as base-estimators. The output probabilities, along with their weighted average output was concatenated with the entire data and fed to an XGBoost model that produced probabilities as ouput.

Finally, the probabilities from both ensembles were combined using a weighted average to yield the final result.

# Competition Result
1. Rank: 2nd
2. [Link to leaderboard](https://www.machinehack.com/course/insurance-churn-prediction-weekend-hackathon-2/leaderboard)
3. [Analytics India Magazine published an article of the top 3 winners](https://analyticsindiamag.com/insurance-churn-prediction-challenge-winners/)
