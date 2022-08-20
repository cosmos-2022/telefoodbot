import sqlite3 as sq
import pandas as pd
import os

from Recommendation_engine.data import import_recipes
from Recommendation_engine.Feature_eng import process_recipes
from Recommendation_engine.inference import load_pkl

MODEL_PATH = 'models/nlp'

def create_and_populate_db():
    data = import_recipes()
    
    # Process the data
    data = process_recipes(data)
    
    # Predict cuisine from trained model
    model = load_pkl(os.path.join(MODEL_PATH, 'pickle_model.pkl'))
    data["cuisine"] = model.predict(data["ingredients_query"].tolist())
    
    db = sq.connect('recipes.db')
    #Verify dtypes
    for col in data.columns:
        data[col] = data[col].astype('str')

    print(' ------------------ Check data before populating the db ------------------')
    print(data.columns)
    print(data.head())
    print(data.shape)
    data.to_sql('main_recipes', db, if_exists='replace')