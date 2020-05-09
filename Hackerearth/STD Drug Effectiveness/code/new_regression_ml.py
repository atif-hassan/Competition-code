# IMPORTANT NOTES:
# ------------------
# Drug "Esomeprazole / naproxen" was present as "Vimovo" in test set. Replaced it. Improved performance
# Drug "Senna S" was present as "Docusate / senna" in test set. Replaced it. Improved performance
# Drug "Empagliflozin" was present as "Synjardy" in test set. Replaced it. Improved performance

import csv
import re
import math
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, plot_importance
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
import pickle

#---------------------------------------------------------------------- DEFINE ALL GLOBAL VARIABLES ----------------------------------------------------------------------------------

word_dims = 50

stops = set(stopwords.words('english'))
not_stop = ['wasn', 'doesn', 'shouldn', 'against', 'didn', 'aren', 'couldn', 'can', 'but', 'won', 'needn', 'isn', 'mustn', 'did', 'weren', 'ain', 'mightn', 'hasn', 'haven', "very", 'shan', 'hadn', "aren't","couldn't","didn't","doesn't","don't","hadn't","hasn't","haven't","isn't","mightn't","mustn't","needn't","no","nor","not","shan't","shouldn't","wasn't","weren't","wouldn't"]
for i in not_stop:
    if i in stops:
        stops.remove(i)

#-------------------------------------------------------------------- ALL GLOBAL VARIABLE DEFINITIONS END -----------------------------------------------------------------------------






#--------------------------------------------------------------------------- DEFINE ALL FUNCTIONS -------------------------------------------------------------------------------------

# Pre-process and load data
def pp_and_load_data(train_file_path, test_file_path, write_path):
    # Load the training data. It consists of both string and integer
    fp = open(train_file_path, encoding='utf-8')
    csvreader = csv.reader(fp)
    # Skip over header
    next(csvreader)
    X_train_ids, X_train_drug, X_train_disease, X_train_review, X_train_meta, Y = list(), list(), list(), list(), list(), list()
    for row in tqdm(csvreader, total=32165):
        X_train_ids.append(int(row[0]))
        X_train_drug.append(row[1])
        # Clean the disease column
        if "</span>" in row[2]:
            disease = "</span> users found this comment helpful."
        else:
            disease = row[2]
        X_train_disease.append(disease)
        X_train_review.append(review_to_words(row[3]))
        X_train_meta.append([int(row[4]), int(row[6])])
        Y.append(float(row[-1]))
    fp.close()

    # Load the test data. It consists of both string and integer
    fp = open(test_file_path, encoding='utf-8')
    csvreader = csv.reader(fp)
    # Skip over header
    next(csvreader)
    X_test_ids, X_test_drug, X_test_disease, X_test_review, X_test_meta = list(), list(), list(), list(), list()
    for row in tqdm(csvreader, total=10760):
        X_test_ids.append(int(row[0]))
        X_test_drug.append(row[1])
        # Clean the disease column
        if "</span>" in row[5]:
            disease = "</span> users found this comment helpful."
        else:
            disease = row[5]
        X_test_disease.append(disease)
        X_test_review.append(review_to_words(row[2]))
        X_test_meta.append([int(row[6]), int(row[4])])
    fp.close()

    return np.array(X_train_ids), X_train_drug, X_train_disease, np.array(X_train_review), np.array(X_train_meta), np.array(X_test_ids), X_test_drug, X_test_disease, np.array(X_test_review), np.array(X_test_meta), np.array(Y)


# Cleans reviews
def review_to_words(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    #meaningful_words = [w for w in words if not w in stops]
    # 6. Stemming
    #stemming_words = [stemmer.stem(w) for w in meaningful_words]
    return words


# One-Hot-Encodes drugs and diseases
def ohe(X_train_drug, X_train_disease, X_test_drug, X_test_disease):
    # Find all drugs
    drugs = list(set(X_train_drug + X_test_drug))
    drugs.sort()
    # Give the drugs, indices
    drugs_inds, index = dict(), 0
    for drug in drugs:
        drugs_inds[drug] = index
        index+=1
    # Now OHE the drugs
    tmp = np.zeros((len(X_train_drug), len(drugs_inds)), dtype=np.float32)
    for i in X_train_drug:
        tmp[drugs_inds[i]] = 1
    X_train_drug = tmp
    tmp = np.zeros((len(X_test_drug), len(drugs_inds)), dtype=np.float32)
    for i in X_test_drug:
        tmp[drugs_inds[i]] = 1
    X_test_drug = tmp

    # Find all diseases
    diseases = list(set(X_train_disease + X_test_disease))
    diseases.sort()
    # Give the diseases, indices
    diseases_inds, index = dict(), 0
    for disease in diseases:
        diseases_inds[disease] = index
        index+=1
    # Now OHE the diseases
    tmp = np.zeros((len(X_train_disease), len(diseases_inds)), dtype=np.float32)
    for i in X_train_disease:
        tmp[diseases_inds[i]] = 1
    X_train_disease = tmp
    tmp = np.zeros((len(X_test_disease), len(diseases_inds)), dtype=np.float32)
    for i in X_test_disease:
        tmp[diseases_inds[i]] = 1
    X_test_disease = tmp

    return np.array(X_train_drug), np.array(X_train_disease), np.array(X_test_drug), np.array(X_test_disease)


# Performs k-fold cross validation
def k_fold(num_folds, X_train_drug, X_train_disease, X_train_meta, Y):
    kf, scores, scores_raw = KFold(n_splits=num_folds), list(), list()
    for train_index, test_index in kf.split(X_train_meta):
        x_train_drug, x_train_disease, x_train_meta, y_train = X_train_drug[train_index], X_train_disease[train_index], X_train_meta[train_index], Y[train_index]
        x_test_drug, x_test_disease, x_test_meta, y_test = X_train_drug[test_index], X_train_disease[test_index], X_train_meta[test_index], Y[test_index]

        # Perform k-means clustering and use the distance form centroids as features
        kmeans = KMeans(n_clusters=80, n_jobs=4, random_state=0)
        x_train_k = kmeans.fit_transform(x_train_meta)
        x_test_k = kmeans.transform(x_test_meta)

        # Concatenate all the features together
        x_train_meta = np.concatenate((x_train_k, x_train_drug, x_train_disease, x_train_meta), axis=1)
        x_test_meta = np.concatenate((x_test_k, x_test_drug, x_test_disease, x_test_meta), axis=1)
        print(x_train_meta.shape, x_test_meta.shape)

        # Perform ergression
        model = LGBMRegressor(random_state=0, n_estimators=5000, learning_rate=0.126, num_leaves=29)
        model.fit(x_train_meta, y_train)
        preds = model.predict(x_test_meta)
        scores.append(max(0, 1 - math.sqrt(mean_squared_error(y_test, preds))) * 100), scores_raw.append(math.sqrt(mean_squared_error(y_test, preds)))
        print("Score: ", scores[-1], scores_raw[-1])
    print("Average Score: ", sum(scores)/len(scores))


# Final model run and submission
def final_submission(X_train_drug, X_train_disease, X_train_meta, X_test_drug, X_test_disease, X_test_meta, Y):
    # Perform k-means clustering and use the distance form centroids as features
    kmeans = KMeans(n_clusters=80, n_jobs=4, random_state=0) #80
    X_train_k = kmeans.fit_transform(X_train_meta)
    X_test_k = kmeans.transform(X_test_meta)

    # Concatenate all the features together
    X_train = np.concatenate((X_train_k, X_train_drug, X_train_disease, X_train_meta), axis=1)
    X_test = np.concatenate((X_test_k, X_test_drug, X_test_disease, X_test_meta), axis=1)
    print(X_train.shape, X_test.shape)

    # Test model and make a submission
    model = LGBMRegressor(random_state=0, n_estimators=5000, learning_rate=0.126, num_leaves=29)#n_estimators=5300, lr=0.126, num_leaves=29
    model.fit(X_train, Y)
    preds = model.predict(X_test)

    fp = open("submit.csv", "w")
    fp.write("patient_id,base_score\n")
    for id_, pred in zip(X_test_ids, preds):
        fp.write(str(id_)+","+str(pred)+"\n")
    fp.close()

#---------------------------------------------------------------------- ALL FUNCTION DEFINITIONS END ------------------------------------------------------------------------------------






#-------------------------------------------------------------------------------- MAIN CODE ---------------------------------------------------------------------------------------------

# Pre-process and load the data
X_train_ids, X_train_drug, X_train_disease, X_train_review, X_train_meta, X_test_ids, X_test_drug, X_test_disease, X_test_review, X_test_meta, Y = pp_and_load_data("../dataset/train.csv", "../dataset/test.csv", "../dataset/pickles/")#load_pp_data("../dataset/pickles/")
# OHE the drugs and diseases
X_train_drug, X_train_disease, X_test_drug, X_test_disease = ohe(X_train_drug, X_train_disease, X_test_drug, X_test_disease)

# Perform k-fold cross validation
k_fold(10, X_train_drug, X_train_disease, X_train_meta, Y)
# Finally, run the model with set parameters and make submission
final_submission(X_train_drug, X_train_disease, X_train_meta, X_test_drug, X_test_disease, X_test_meta, Y)
