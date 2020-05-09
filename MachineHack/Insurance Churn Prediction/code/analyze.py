import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from collections import Counter
import math


# Feature 1, 3, 4, 5, 6


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 12

fp = open("Train.csv")
csvreader = csv.reader(fp)
header = next(csvreader)
X, Y = list(), list()
for row in tqdm(csvreader):
    X.append([float(i) for i in row[:-1]])
    Y.append(int(row[-1]))
X, Y = np.array(X), np.array(Y)



fp = open("Test.csv")
csvreader = csv.reader(fp)
header = next(csvreader)
X_test = list()
for row in tqdm(csvreader):
    X_test.append([float(i) for i in row])
X_test = np.array(X_test)





'''xx = Counter(X[:,3])
print(len(xx))
for i, j in xx.items():
    print(i, j)'''
unique = list(set(X[:,0]))
unique.sort()
for i in range(len(unique)-1):
    #print(unique[i+1]-unique[i])
    print(round((unique[i]/0.09417397)-0.06379))
    #print(round((unique[i]/0.12015787)-0.193581))
    #print(unique[i]/0.003883118)
    #print(unique[i]/0.3227902)
    #print(unique[i]/0.434137)











kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
class_0, class_1 = np.where(Y==0)[0], np.where(Y==1)[0]

'''for index, col in enumerate(header[:-1]):
    plt.figure(figsize=(6,4), dpi= 80)
    sns.distplot(X[:,index], kde=False, color="dodgerblue", label="Train", bins=30, **kwargs)
    sns.distplot(X_test[:,index], kde=False, color="orange", label="Test", bins=30, **kwargs)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    #plt.xticks(np.arange(-4, 5, 1))
    #plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.show()'''




'''for index, col in enumerate(header[:-1]):
    plt.figure(figsize=(6,4), dpi= 80)
    sns.lineplot([i for i in range(len(X[:1000]))], X[:1000,index], color="dodgerblue", label="class 0")
    plt.ylabel(col)
    #plt.xticks(np.arange(-4, 5, 1))
    #plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.show()'''





#X[:,0] = np.round((X[:,0]/0.09417397)-0.06379)
'''for index, col in enumerate(header[:-1]):
    plt.figure(figsize=(6,4), dpi= 80)
    sns.distplot(X[class_0,index], kde=False, color="dodgerblue", label="class 0", bins=30, **kwargs)
    sns.distplot(X[class_1,index], kde=False, color="orange", label="class 1", bins=30, **kwargs)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    #plt.xticks(np.arange(-4, 5, 1))
    #plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.show()'''

'''num_components = 3
#pca = KernelPCA(n_components=num_components, random_state=0, kernel='sigmoid')
tsne = TSNE(n_components=num_components, random_state=0)
X = tsne.fit_transform(X[:,:7])
#kmeans = KMeans(n_clusters=num_components, random_state=0)
#X = kmeans.fit_transform(X[:,:7])

for index in range(num_components):
    plt.figure(figsize=(6,4), dpi= 80)
    sns.distplot(X[class_0,index], kde=False, color="dodgerblue", label="class 0", bins=30, **kwargs)
    sns.distplot(X[class_1,index], kde=False, color="orange", label="class 1", bins=30, **kwargs)
    #plt.xlabel(col)
    plt.ylabel("Frequency")
    #plt.xticks(np.arange(-4, 5, 1))
    #plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.show()'''
