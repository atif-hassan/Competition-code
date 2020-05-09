import csv
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from gensim.models import FastText
import pickle
import numpy as np

# Clean review
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
    # 7. space join words
    return(words)



def train_word_vectors(train_path, test_path, output_path):

    # Final Corpus
    X = list()

    # Load training data reviews
    fp = open(train_path, encoding='utf-8')
    csvreader = csv.reader(fp)
    # Skip over header
    next(csvreader)
    for row in tqdm(csvreader, total=161297):
        X.append(review_to_words(row[3]))
    fp.close()

    # Load test data reviews
    fp = open(test_path, encoding='utf-8')
    csvreader = csv.reader(fp)
    # Skip over header
    next(csvreader)
    for row in tqdm(csvreader, total=54766):
        X.append(review_to_words(row[3]))
    fp.close()

    print(len(X))

    # Build and save fasttext model
    model = FastText(size=50, window=4, min_count=0, workers=4, sg=1)  # instantiate
    model.build_vocab(sentences=X)
    model.train(sentences=X, total_examples=len(X), epochs=30)  # train
    model.wv.save(output_path)

    # Let's check out some results
    word = "oral"
    results = model.similar_by_word(word)
    print(results[:10])

    print("\n\n")

    # Let's check out some results
    word = "drug"
    results = model.similar_by_word(word)
    print(results[:10])

    print("\n\n")

    # Let's check out some results
    word = "sex"
    results = model.similar_by_word(word)
    print(results[:10])

    print("\n\n")

    # Let's check out some results
    word = "surgery"
    results = model.similar_by_word(word)
    print(results[:10])

    print("\n\n")

    # Let's check out some results
    word = "and"
    results = model.similar_by_word(word)
    print(results[:10])

    print("\n\n")



def drug_disease_vectors(train_path, test_path, output_path_drugs, output_path_diseases):
    drugs, diseases = list(), list()
    
    # Load training data reviews
    fp = open(train_path, encoding='utf-8')
    csvreader = csv.reader(fp)
    # Skip over header
    next(csvreader)
    for row in tqdm(csvreader, total=161297):
        drugs.append(row[1])
        diseases.append(row[2])
    fp.close()

    # Load test data reviews
    fp = open(test_path, encoding='utf-8')
    csvreader = csv.reader(fp)
    # Skip over header
    next(csvreader)
    for row in tqdm(csvreader, total=54766):
        drugs.append(row[1])
        diseases.append(row[5])
    fp.close()

    # Generate indices for each drug and disease
    drugs_unique, diseases_unique = list(set(drugs)), list(set(diseases))
    drugs_unique.sort(), diseases_unique.sort()
    drugs_ind, diseases_ind = dict(), dict()
    for index, drug in enumerate(drugs_unique):
        drugs_ind[drug] = index
    for index, disease in enumerate(diseases_unique):
        diseases_ind[disease] = index
    print(len(drugs_unique), len(diseases_unique))

    # Generate co-occurrence matrix of size (num_drugs x num_diseases)
    drug_disease_matrix = np.zeros((len(drugs_unique), len(diseases_unique)), dtype=np.float32)
    for drug, disease in tqdm(zip(drugs, diseases)):
        drug_disease_matrix[drugs_ind[drug], diseases_ind[disease]] =1

    # Now generate vector dicts for drugs and diseases
    drugs_vecs, diseases_vecs = dict(), dict()
    for drug in drugs_unique:
        drugs_vecs[drug] = drug_disease_matrix[drugs_ind[drug], :]
    for disease in diseases_unique:
        diseases_vecs[disease] = drug_disease_matrix[:,diseases_ind[disease]]
    print(len(drugs_vecs), len(diseases_vecs))

    # Finally save the generated vector dictionaries
    fp = open(output_path_drugs, "wb")
    pickle.dump(drugs_vecs, fp)
    fp.close()
    fp = open(output_path_diseases, "wb")
    pickle.dump(diseases_vecs, fp)
    fp.close()
    


#train_word_vectors("../dataset/drugsComTrain_raw.csv", "../dataset/drugsComTest_raw.csv", "../dataset/fastText.bin")
drug_disease_vectors("../dataset/train.csv", "../dataset/test.csv", "../dataset/drug_vectors.bin", "../dataset/disease_vectors.bin")
