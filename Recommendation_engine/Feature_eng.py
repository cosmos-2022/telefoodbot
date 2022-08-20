import nltk
import pandas as pd
import re
from sklearn import feature_extraction, model_selection, pipeline, manifold, preprocessing,feature_selection
from Recommendation_engine.data import import_data
import numpy as np



stop_words = [ "advertisement", "advertisements",
                         "cup", "cups",
                         "tablespoon", "tablespoons", 
                         "teaspoon", "teaspoons", 
                         "ounce", "ounces",
                         "salt", 
                         "pepper", 
                         "pound", "pounds",]


nltk.download('wordnet')
nltk.download('stopwords')



def preprocess_text(text,flag_lemm =True,flag_stem = False,lst_stopwords=None):

    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    list_text = text.split()

    #remove stopwords

    if lst_stopwords is not None:

        list_text = [word for word in list_text if word not in lst_stopwords]

     ## Stemming (remove -ing, -ly, ...)


    if flag_stem == True:
        ps = nltk.stem.porter.PorterStemmer()
        list_text = [ps.stem(word) for word in list_text]
                
    ## Lemmatisation (convert the word into root word)
    if flag_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        list_text = [lem.lemmatize(word) for word in list_text]

    

    ## back to string from list
    text = " ".join(list_text)

    ## Remove digits
    text = ''.join([i for i in text if not i.isdigit()])

    ## remove mutliple space
    text = re.sub(' +', ' ', text)

    return text




def process_data():

    dataset = import_data()

    def processing(row):
        ls = row['ingredients']
        return ' '.join(ls)


    dataset['ingredients'] = dataset.apply(lambda x: processing(x), axis=1)
    dataset.dropna(inplace=True)
    dataset = dataset.drop(columns=['id']).reset_index(drop=True)


    stop_word_list = nltk.corpus.stopwords.words("english")

    stop_word_list.extend(stop_words)

    dataset["ingredients_query"] = dataset["ingredients"].apply(lambda x: 
          preprocess_text(x,flag_lemm =True,flag_stem = False,lst_stopwords=stop_word_list))

    return dataset



def create_embedding(dataset):
    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1,2))

    corpus = dataset["ingredients_query"]

    vectorizer.fit(corpus)

    embedded_ingredients = vectorizer.transform(corpus)
    dic_vocabulary = vectorizer.vocabulary_

    ## Chi squarred correlation embeddings reduction
    labels = dataset["cuisine"]
    names = vectorizer.get_feature_names()
    p_value_limit = 0.95
    dtf_features = pd.DataFrame()

    for cat in np.unique(labels):
        chi2, p = feature_selection.chi2(embedded_ingredients, labels==cat)
        dtf_features = dtf_features.append(pd.DataFrame(
                       {"feature":names, "score":1-p, "labels":cat}))
        dtf_features = dtf_features.sort_values(["labels","score"], 
                        ascending=[True,False])
        dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
    names = dtf_features["feature"].unique().tolist()

    ## Check the main ingredients
    for cat in np.unique(labels):
        print("# {}:".format(cat))
        print("  . selected features:",len(dtf_features[dtf_features["labels"]==cat]))
        print("  . top features:", ",".join(dtf_features[dtf_features["labels"]==cat]["feature"].values[:10]))
        print(" ")
    
    ## New embeddings
    vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=names)
    vectorizer.fit(corpus)
    embedded_ingredients = vectorizer.transform(corpus)
    dic_vocabulary = vectorizer.vocabulary_

    return vectorizer

def process_recipes(data):
    # list of stopwords
    stop_word_list = nltk.corpus.stopwords.words("english")

    # Extend list of stop words
    stop_word_list.extend(stop_words)

    data["ingredients_query"] = data["ingredients"].apply(lambda x: 
            preprocess_text(x,flag_lemm =True,flag_stem = False, 
            lst_stopwords=stop_word_list))
    return data

def get_tokenize_text(input_text):
    # list of stopwords
    stop_word_list = nltk.corpus.stopwords.words("english")

    # Extend list of stop words
    stop_word_list.extend(stop_words)

    return preprocess_text(input_text, flag_lemm =True,flag_stem = False, lst_stopwords=stop_word_list)





