import os
import pandas as pd
from pandas.core.indexes.base import Index
Data_Cusian_Path = "data/cuisine_data/"
Data_reciepe_path = "data/recipes_data/"


def import_data():
    train = pd.read_json(os.path.join(Data_Cusian_Path, 'train.json'))
    test =  pd.read_json(os.path.join(Data_Cusian_Path, 'test.json'))
    return pd.concat([train,test],axis=0)


def import_recipes():
    data_path_ar = os.path.join(Data_reciepe_path, "recipes_raw_nosource_ar.json")
    data_path_epi = os.path.join(Data_reciepe_path, "recipes_raw_nosource_epi.json")
    data_path_fn = os.path.join(Data_reciepe_path, "recipes_raw_nosource_fn.json")

    data = pd.concat([pd.read_json(data_path_ar,orient='index'),pd.read_json(data_path_epi,orient='index'),pd.read_json(data_path_fn,orient='index')])

    data = data.reset_index()
    data = data.drop(columns=['picture_link', 'index'])
    return data


data = import_recipes()

print(data)


