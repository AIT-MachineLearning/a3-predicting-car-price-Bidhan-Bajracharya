import mlflow
import os
import pickle

def save(filename: str, obj: object):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename: str) -> object:
    with open(filename, 'rb') as file:
        b = pickle.load(file)
    return b

model_name = os.environ.get('APP_MODEL_NAME')
model_version = os.environ.get('MODEL_VERSION')
def load_mlflow(stage='Staging'):
    cache_path = os.path.join("models", stage)
    if(os.path.exists(cache_path) == False):
        os.makedirs(cache_path)
    
    # check if we cache the model
    path = os.path.join(cache_path,model_name)
    if(os.path.exists( path ) == False):
        # This will keep loading the model again and again
        model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")
        save(filename=path, obj=model)

    model = load(path)
    return model