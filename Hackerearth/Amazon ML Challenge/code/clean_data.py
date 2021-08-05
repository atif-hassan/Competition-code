# clean data for tfidf, doc2vec and countvectorizer

import os 
import re
import csv
import spacy
import string
import pandas as pd


en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words
load_model = spacy.load('en', disable=["tok2vec", "tagger", "parser", "attribute_ruler", "ner"])

emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)

def text_preprocess(x):
  x = x.lower()  # lowercase
  x = ' '.join([word for word in x.split(' ') if word not in sw_spacy]) # stopwords
  x = x.encode('ascii', 'ignore').decode()  # unicode
  x = re.sub(r'https*\S+', ' ', x) # url
  x = re.sub(r'@\S+', ' ', x)   # mentions
  x = re.sub(r'#\S+', ' ', x)   # hastags
  x = x.replace("'", "")    # remove ticks
  x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x) # punctuation
  x = re.sub(r'\w*\d+\w*', '', x) # numbers
  x = re.sub(r'\s{2,}', ' ', x) # over spaces
  x = emoji_pattern.sub(r'', x) # emojis
  x = re.sub('[^A-Za-z0-9]+', ' ', x) # special charachters
  x = load_model(x)
  x = " ".join([token.lemma_ for token in x])

  return x

def clean_data(df, columns_to_clean):
    for col in columns_to_clean:
        df[f"cleaned_{col}"] = df[col].progress_apply(text_preprocess)
    
    df = df.drop(['TITLE', 'DESCRIPTION', 'BULLET_POINTS', 'BRAND'], axis=1)
    return df



if __name__ == "__main__":

    BASE_DIR = "/content/input"

    train_path = os.path.join(BASE_DIR, "train.csv")
    test_path = os.path.join(BASE_DIR, "train.csv")
    sample_submission_path = os.path.join(BASE_DIR, "sample_submission.csv")

    train = pd.read_csv(train_path, escapechar="\\", quoting=csv.QUOTE_NONE)
    train_na_free = train.fillna(value="NaN")
    train_cleaned = clean_data(train_na_free, ['TITLE', 'DESCRIPTION', 'BULLET_POINTS', 'BRAND'])
    
    # save cleaned train file
    train_cleaned.to_csv("cleaned_train.csv", index=False)

    test = pd.read_csv(test_path, escapechar="\\", quoting=csv.QUOTE_NONE)
    test_na_free = test.fillna(value="NaN")
    test_cleaned = clean_data(test_na_free, ['TITLE', 'DESCRIPTION', 'BULLET_POINTS', 'BRAND'])
    
    # save cleaned test file
    train_cleaned.to_csv("cleaned_test.csv", index=False)
