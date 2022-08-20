import pandas as pd
import numpy as np
import os

#for processing
import re
import nltk

## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing

from Recommendation_engine.Feature_eng import process_data, create_embedding
## model & processing libraries

from sklearn import feature_selection
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn import utils
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle

## DB accesses
import sqlite3 as sq

MODEL_PATH = "models/nlp"
MODEL_EMBEDDINGS_PATH = os.path.join(MODEL_PATH, 'similarity_embeddings')
CUISINE_CLASSES = ['brazilian','british','cajun_creole','chinese','filipino','french','greek','indian','irish','italian','jamaican','japanese','korean','mexican','moroccan','russian','southern_us','spanish','thai','vietnamese']
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(MODEL_EMBEDDINGS_PATH, exist_ok=True)

## Save to file in the current working directory
def save_pkl(file, pkl_filename):
    with open(pkl_filename, 'wb') as pkl_file:
        pickle.dump(file, pkl_file)

def compute_performances(predicted, predicted_prob, y_test):
    
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    balance_accuracy = metrics.balanced_accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob, 
                                multi_class="ovr")
    print("Balanced Accuracy:",  round(balance_accuracy,2))
    print("Accuracy:",  round(accuracy,2))
    print("Auc:", round(auc,2))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted))


def create_model_cuisine_predictions():
    ## Process data
    dataset = process_data()

    ## Create embeddings
    vectorizer = feature_extraction.text.TfidfVectorizer() #create_embeddings(dataset)

    ## Model
    classifier = LogisticRegressionCV(cv=3,
                                      random_state=42,
                                      max_iter=300,
                                      n_jobs=-1,
                                      verbose=1) #naive_bayes.MultinomialNB()

    ## pipeline
    model = pipeline.Pipeline([("vectorizer", vectorizer),  
                                ("classifier", classifier)])

    ## Split the dataset
    dataset_train, dataset_test = model_selection.train_test_split(dataset, test_size=0.3, random_state=42)

    ## Create embeddings
    X_train = dataset_train['ingredients_query']; X_test = dataset_test['ingredients_query']
    y_train = dataset_train['cuisine']; y_test = dataset_test['cuisine']; 

    ## train classifier
    model.fit(X_train, y_train)

    ## test
    predicted = model.predict(X_test)
    predicted_prob = model.predict_proba(X_test)

    ## Compute performance of the model
    compute_performances(predicted, predicted_prob, y_test)
    
    ## Save model and vectorizer to disk
    save_pkl(model, os.path.join(MODEL_PATH, "pickle_model.pkl"))

def d2v_embeddings(data):
    data = data['ingredients_query'].tolist()
    tagged_data = [TaggedDocument(words=row.split(), tags=[str(index)]) for index, row in enumerate(data)]

    max_epochs = 20
    vec_size = 50
    alpha = 0.025

    model_embedding = Doc2Vec(vector_size=vec_size,alpha=alpha, min_alpha=0.00025,min_count=1,dm =1)
  
    model_embedding.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model_embedding.train(tagged_data,
                    total_examples=model_embedding.corpus_count,
                    epochs=model_embedding.epochs)
        # decrease the learning rate
        model_embedding.alpha -= 0.0002
        # fix the learning rate, no decay
        model_embedding.min_alpha = model_embedding.alpha
    
    return model_embedding

def train_model_embeddings():
    db = sq.connect('recipes.db')
    cursor = db.cursor()
    
    for cuisine in CUISINE_CLASSES:
        sql_query = "SELECT title, instructions, ingredients, ingredients_query FROM main_recipes WHERE cuisine = ?"
        data = pd.read_sql(sql_query, db, params=(cuisine,))
        
        model_embedding = d2v_embeddings(data)
        save_pkl(model_embedding, os.path.join(MODEL_EMBEDDINGS_PATH, f'd2v_{cuisine}.pkl'))


