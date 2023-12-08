import pickle
import os

# project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.ini')

# Using INI configuration file
from configparser import ConfigParser

config = ConfigParser()
config.read(CONFIG_PATH)
DB_PATH = str(config.get("PATHS", "DB_PATH"))
print(DB_PATH)
MODEL_PATH = str(config.get("PATHS", "MODEL_PATH"))
RANDOM_STATE = int(config.get("ML", "RANDOM_STATE"))

# # Doing the same with a YAML configuration file
# import yaml
#
# with open("config.yml", "r") as f:
#     config_yaml = yaml.load(f, Loader=yaml.SafeLoader)
#     DB_PATH = str(config_yaml['paths']['db_path'])
#     MODEL_PATH = str(config_yaml['paths']["model_path"])
#     RANDOM_STATE = int(config_yaml["ml"]["random_state"])

# SQLite requires the absolute path
# DB_PATH = os.path.abspath(DB_PATH)
DB_PATH = os.path.join(ROOT_DIR, os.path.normpath(DB_PATH))
print(DB_PATH)

def preprocess_data():
    print(f"Preprocessing data")
    num_features = ['vendor_id','passenger_count',	'pickup_longitude','pickup_latitude','	dropoff_longitude', '	dropoff_latitude']
    cat_features = ['pickup_datetime','store_and_fwd_flag']
    train_features = num_features + cat_features

    column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
    ('scaling', StandardScaler(), num_features)]
    )

    pipeline = Pipeline(steps=[
    ('ohe_and_scaling', column_transformer),
    ('regression', Ridge())
    ]) 
    return [pipeline,train_features]

def persist_model(model, path):
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Done")

def load_model(path):
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        model = pickle.load(file)
    print(f"Done")
    return model
