import csv
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

# Function for preprocessing the corpus
# We take (title + desc + brand) and if desc is not available then (title + bullet_points + brand)
def pre_process(corpus1, corpus2, corpus3, corpus4):
    corpus = list()
    for i in tqdm(range(len(corpus1))):
    	# Get title
    	if len(corpus1[i]) > 0:
    		title = corpus1[i]
    	else:
    		title = ""
    	# Get product description
    	if len(corpus2[i]) > 0:
    		desc = corpus2[i]
    	else:
    		if len(corpus4[i]) > 0:
    			desc = corpus4[i]
    		else:
    			desc = ""
    	# Get product brand
    	if len(corpus3[i]) > 0:
    		brand = corpus3[i]
    	else:
    		brand = ""
    	
    	if len(title) < 1 and len(desc) < 1 and len(brand) < 1:
    		sample = "NAN"
    	else:
    		sample = title + " " + desc + " " + brand
    	corpus.append(sample)
    return corpus




# Read the data in pandas
df = pd.read_csv("../dataset/cleaned_train.csv", escapechar="\\", quoting=csv.QUOTE_NONE)
df = df.replace(np.nan, '', regex=True)
# Get all the different columns including the target variables
corpus1 = df["cleaned_TITLE"].values
corpus2 = df["cleaned_DESCRIPTION"].values
corpus3 = df["cleaned_BRAND"].values
corpus4 = df["cleaned_BULLET_POINTS"].values
Y = df["BROWSE_NODE_ID"].values
del df

# Pre-processing
corpus = pre_process(corpus1, corpus2, corpus3, corpus4)
del corpus1, corpus2, corpus3, corpus4

# Generate TFIDF vectors
vectorizer = TfidfVectorizer(max_features=12000, min_df=3, ngram_range=(1, 1))
vec = vectorizer.fit_transform(corpus)
# Keep you ram clean
del corpus

# Check out the top features
print("\nTop Feature Names:", vectorizer.get_feature_names()[:100])
print("\nTrain Data Shape:", vec.shape)

# Save the vectorized data
fp = open("../dataset/train_tf_idf_vec.bin", "wb")
pickle.dump(vec, fp)
fp.close()

# Save the target variable
fp = open("../dataset/target.bin", "wb")
pickle.dump(Y, fp)
fp.close()

# Ram maintenance
del vec, Y



# Load the test set
df = pd.read_csv("../dataset/cleaned_test.csv", escapechar="\\", quoting=csv.QUOTE_NONE)
df = df.replace(np.nan, '', regex=True)
# Get all the different columns including the product ids
corpus1 = df["cleaned_TITLE"].values
corpus2 = df["cleaned_DESCRIPTION"].values
corpus3 = df["cleaned_BRAND"].values
corpus4 = df["cleaned_BULLET_POINTS"].values
prod_ids = df["PRODUCT_ID"].values
del df

# Pre-processing
corpus = pre_process(corpus1, corpus2, corpus3, corpus4)
del corpus1, corpus2, corpus3, corpus4

# Generate TFIDF vectors
vec = vectorizer.transform(corpus)
# Keep you ram clean
del corpus

# Check out data shape
print("\nTest Data Shape:", vec.shape)

# Save the vectorized data
fp = open("../dataset/test_tf_idf_vec.bin", "wb")
pickle.dump(vec, fp)
fp.close()

# Save the target variable
fp = open("../dataset/prod_ids.bin", "wb")
pickle.dump(prod_ids, fp)
fp.close()
