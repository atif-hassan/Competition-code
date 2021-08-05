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

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, BatchNormalization, Dropout, Concatenate, Activation, Embedding, Flatten
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
import pickle
import pandas as pd


# Get the sparse vectorized training data
fp = open("../dataset/train_tf_idf_vec.bin", "rb")
X = pickle.load(fp)
fp.close()

# Get the target variable
fp = open("../dataset/target.bin", "rb")
Y = pickle.load(fp)
fp.close()

# Get the dense embeddings from small sentence transformer
fp = open("../dataset/sm_train_new.pkl", "rb")
stored_embeds = pickle.load(fp)
X_train_d1 = stored_embeds['embeddings']
print("\n\nDense Embedding Small shape:", X_train_d1.shape ,"\n\n")
del stored_embeds
fp.close()

# Get the doc2vec dense vectors
fp = open("../dataset/train_doc2vec.bin", "rb")
X_train_doc2vec = pickle.load(fp)
fp.close()

# Get the standard handcrafted features
df = pd.read_csv("../dataset/train_fe_scaled.csv")
hand_fe = df.values[:,1:-3]# Some features irrelevant
del df

# The classes are not in sequence.
# Hence, create a mapping
classes = np.unique(Y)
classes.sort()
num_classes = len(classes)
classes_dict, classes_dict_rev = dict(), dict()
for index, i in enumerate(classes):
    classes_dict[i] = index
    classes_dict_rev[index] = i
# Be good to your ram
del classes

# OHE the target variable
Y_train = np.zeros((len(Y), num_classes))
for i in range(len(Y)):
    Y_train[i, classes_dict[Y[i]]] = 1
# Memory management is important
del Y

# Convert sparse vectors to dense
# NNs work with dense only :/
X_train = X.toarray()
# Maintenance
del X

# Check out the data
print("Train data shape:", X_train.shape, Y_train.shape)


# Defining the model architecture
inp1 = Input(shape=(X_train.shape[1],))
inp2 = Input(shape=(X_train_d1.shape[1],))
inp3 = Input(shape=(hand_fe.shape[1],))

new_input = Concatenate()([inp1, inp2, inp3])
x = Dense(4096, activation='linear', kernel_initializer=initializers.he_normal(seed=30))(new_input)
x = BatchNormalization()(x)
x = Activation(activations.relu)(x)
x = Dense(2048, activation='linear', kernel_initializer=initializers.he_normal(seed=32))(x)
x = BatchNormalization()(x)
x = Activation(activations.relu)(x)
x = Dense(4096, activation='linear', kernel_initializer=initializers.he_normal(seed=34))(x)
x = BatchNormalization()(x)
x = Activation(activations.relu)(x)
out = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=[inp1, inp2, inp3, inp4], outputs=out)
model.summary()

# Now train
#model_checkpoint_callback = ModelCheckpoint(filepath="mlp_model_{epoch:02d}", monitor='accuracy', mode='max')
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])
model.fit([X_train, X_train_d1, hand_fe], Y_train, batch_size=512, epochs=1, shuffle=True)#, callbacks=[model_checkpoint_callback])#, validation_split=0.2)
del X_train, Y_train, X_train_d1, hand_fe
# Dont forget to save the model
model.save('mlp_model')
