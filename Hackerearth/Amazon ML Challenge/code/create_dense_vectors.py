import csv
from tqdm import tqdm
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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
del df

# Pre-processing
corpus_train = pre_process(corpus1, corpus2, corpus3, corpus4)
del corpus1, corpus2, corpus3, corpus4

# Read the data in pandas
df = pd.read_csv("../dataset/cleaned_test.csv", escapechar="\\", quoting=csv.QUOTE_NONE)
df = df.replace(np.nan, '', regex=True)
# Get all the different columns including the target variables
corpus1 = df["cleaned_TITLE"].values
corpus2 = df["cleaned_DESCRIPTION"].values
corpus3 = df["cleaned_BRAND"].values
corpus4 = df["cleaned_BULLET_POINTS"].values
del df

# Pre-processing
corpus_test = pre_process(corpus1, corpus2, corpus3, corpus4)
del corpus1, corpus2, corpus3, corpus4

# We are going to generate document vectors for both train and test
corpus = corpus_train + corpus_test
print("\n\nTotal corpus size", len(corpus))
print("Average corpus length:", np.average([len(txt.split()) for txt in corpus]))
index = len(corpus_train)
del corpus_train, corpus_test

# Generate doc2vec vectors
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
model = Doc2Vec(documents, vector_size=300, window=30, min_count=3, workers=64, dm=0, epochs=10)
# Keep you ram clean
del documents

# Get the train doc2vec vectors
train_vec = np.zeros((index, 300))
for i in tqdm(range(index)):
	train_vec[i] = model[i]
# Save the vectorized data
fp = open("../dataset/train_doc2vec.bin", "wb")
pickle.dump(train_vec, fp)
fp.close()
del train_vec

# Get the test doc2vec vectors
test_vec = np.zeros((len(corpus)-index, 300))
for i in tqdm(range(index, len(corpus))):
	test_vec[i-index] = model[i]
# Save the vectorized data
fp = open("../dataset/test_doc2vec.bin", "wb")
pickle.dump(test_vec, fp)
fp.close()
