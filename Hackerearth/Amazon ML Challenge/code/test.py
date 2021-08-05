seed_value= 12321
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow
tensorflow.random.set_seed(seed_value)
session_conf = tensorflow.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tensorflow.compat.v1.Session(graph=tensorflow.compat.v1.get_default_graph(), config=session_conf)
tensorflow.compat.v1.keras.backend.set_session(sess)

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, BatchNormalization, Dropout, Concatenate, Activation
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam, SGD
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier


# Load up the 12K tfidf vectors
fp = open("../dataset/test_tf_idf_vec.bin", "rb")
X = pickle.load(fp)
fp.close()

# Load up the 10K count vectors
fp = open("../dataset/test_tf_idf_vec_1.bin", "rb")
X1 = pickle.load(fp)
fp.close()

# Load up the 8K count vectors
fp = open("../dataset/test_tf_idf_vec_2.bin", "rb")# We take different sized vectors in order to get less correlated predictions
X2 = pickle.load(fp)
fp.close()

# Get the target
fp = open("../dataset/target.bin", "rb")
Y = pickle.load(fp)
fp.close()

# Get the product IDS (required for submittision file)
fp = open("../dataset/prod_ids.bin", "rb")
prod_ids = pickle.load(fp)
fp.close()

# Get the dense embeddings from small sentence transformer
fp = open("../dataset/sm_test_new.pkl", "rb")
stored_embeds = pickle.load(fp)
X_test_d1 = stored_embeds['embeddings']
print("\n\nDense Embedding shape:", X_test_d1.shape ,"\n\n")
del stored_embeds
fp.close()

# Get the hand crafter features
df = pd.read_csv("../dataset/test_fe_scaled.csv")
hand_fe = df.values[:,1:-3]
del df

# Find all the classes
classes = np.unique(Y)
del Y
classes.sort()
classes_dict_rev = dict()
for index, i in enumerate(classes):
	classes_dict_rev[index] = i
# Be good to your ram
del classes

# Convert all sparse vectors to dense
X = X.toarray()
X1 = X1.toarray()
X2 = X2.toarray()
#print("Test data shape:", X.shape)

# Load up all the models
model5 = tensorflow.keras.models.load_model('mlp_model')#66
model1 = tensorflow.keras.models.load_model('mlp_model_1')#67
model2 = tensorflow.keras.models.load_model('mlp_model_2')#64
model3 = tensorflow.keras.models.load_model('mlp_model_3')#66
model4 = tensorflow.keras.models.load_model('mlp_model_4')#65
preds = list()

# Pairwise layered weighted averaging (MY INVENTION :D )
for i in tqdm(range(len(X)//525)):
	preds1 = model2.predict([X1[i*525:(i+1)*525], X_test_d1[i*525:(i+1)*525], hand_fe[i*525:(i+1)*525]], batch_size=256, verbose=0)
	preds2 = model4.predict([X[i*525:(i+1)*525], X_test_d1[i*525:(i+1)*525], hand_fe[i*525:(i+1)*525]], batch_size=256, verbose=0)
	preds_f1 = preds1*0.5 + preds2*0.5
	
	preds3 = model1.predict([X1[i*525:(i+1)*525], X_test_d1[i*525:(i+1)*525], hand_fe[i*525:(i+1)*525]], batch_size=256, verbose=0)
	preds4 = model3.predict([X2[i*525:(i+1)*525], X_test_d1[i*525:(i+1)*525], hand_fe[i*525:(i+1)*525]], batch_size=256, verbose=0)
	preds_f2 = preds1*0.6 + preds2*0.4
	
	preds_f = preds_f1*0.5 + preds_f2*0.5
	tmp = [classes_dict_rev[i] for i in np.argmax(preds_f, axis=1)]
	preds.extend(tmp)

# Make final submission
fp = open("submit.csv", "w")
fp.write("PRODUCT_ID,BROWSE_NODE_ID")
for prod_id, pred in zip(prod_ids, preds):
	fp.write("\n"+str(prod_id)+","+str(pred))
fp.close()
