from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score

import pandas as pd
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from omegaconf.omegaconf import OmegaConf


def evaluate(ytrue, ypred):
    return(
        accuracy_score(ytrue, ypred),
        f1_score(ytrue, ypred, average="macro"),
        recall_score(ytrue, ypred, average="macro")
    )

def train(X, y, input_schema, output_schema, input_example_):
    
    # Importando parametros
    params = OmegaConf.load("config.yml")
    params = params['parameters']

    # Definindo o track

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    try:
        exp = mlflow.create_experiment(name="LogisticRegression")
        get_ = mlflow.get_experiment(exp)

    except mlflow.exceptions.MlflowException as e:
        get_ = mlflow.set_experiment(experiment_name="LogisticRegression")

    
    with mlflow.start_run(experiment_id=get_.experiment_id) as run:
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=40)
        model = LogisticRegression(**params)
        model.fit(xtrain, ytrain)
        
        (acc, f1, rec) = evaluate(ytest, model.predict(xtest))

        # log metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "f1": f1,
            "recall": rec
        })

        #log params



if __name__ == "__main__":
    data = load_wine(as_frame=True)['data']
    target = load_wine(as_frame=True)['target']

    X = data.values
    y = target.values

    train(X, y, input_schema, output_schema, input_example)