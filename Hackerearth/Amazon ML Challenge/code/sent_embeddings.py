# clean data for bert (sentence embeddings)

#pip install -q transformers
#pip install -q sentence-transformers
#pip install -q pytorch-lightning
import csv
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

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

def create_corpus(df):
    corpus1 = df["cleaned_TITLE"].values
    corpus2 = df["cleaned_DESCRIPTION"].values
    corpus3 = df["cleaned_BRAND"].values
    corpus4 = df["cleaned_BULLET_POINTS"].values

    corpus = pre_process(corpus1, corpus2, corpus3, corpus4)

    return corpus

if __name__ == "__main__":

    train_file_path = "/content/input/new_cleaned/new_cleaned_train_bert.csv"
    test_file_path = "/content/input/new_cleaned/new_cleaned_test_bert.csv"


    train_df = pd.read_csv(train_file_path, escapechar="\\", quoting=csv.QUOTE_NONE)
    test_df = pd.read_csv(test_file_path, escapechar="\\", quoting=csv.QUOTE_NONE)

    train_df = train_df.replace(np.nan, '', regex=True)
    test_df = test_df.replace(np.nan, '', regex=True)

    train_corpus = create_corpus(train_df)
    del train_df

    test_corpus = create_corpus(test_df)
    del test_df

    # create SentenceTransformer
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2', device='cuda:0')

    train_embeds = model.encode(train_corpus, batch_size = 1024, device='cuda:0', show_progress_bar=True)
    del train_corpus
    
    test_embeds = model.encode(test_corpus, batch_size = 1024, device='cuda:0', show_progress_bar=True)
    del test_corpus

    # save embeddings
    with open('sm_train.pkl', "wb") as f:
        pickle.dump({'embeddings': train_embeds}, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('sm_test.pkl', "wb") as f:
        pickle.dump({'embeddings': test_embeds}, f, protocol=pickle.HIGHEST_PROTOCOL)