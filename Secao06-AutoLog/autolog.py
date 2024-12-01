import mlflow.models
import mlflow.models
import pandas as pd
from  sklearn.metrics import r2_score, mean_absolute_error
from  sklearn.linear_model import SGDRegressor
from  sklearn.model_selection import train_test_split


import mlflow
import mlflow.sklearn

import omegaconf
def print_(get_):
    return f"""
Name..............: {get_.name}
Id................: {get_.experiment_id}
Tag...............: {get_.tags}
LyfeCycle_Stage...: {get_.lifecycle_stage}
Artifact Location : {get_.artifact_location}
Timestamp Create..: {get_.creation_time}
"""

def train_(X, y):


    mlflow.set_tracking_uri('./meuMlflow')

    try:
        exp = mlflow.create_experiment(name="SGDregressor")
        get_ = mlflow.get_experiment(exp)

    except:
        get_ = mlflow.set_experiment(experiment_name="SGDregressor")

    # Informações
    print("URI : ", mlflow.get_tracking_uri())
    print(print_(get_))


    xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=32, test_size=0.75)
    
    # Salvando dados train e test
    pd.concat([xtrain, ytrain]).to_csv("dataset2_/train.csv")
    pd.concat([xtest, ytest]).to_csv("dataset2_/test.csv")

    # Pegando os hiperparametros do modelo
    config = omegaconf.OmegaConf.load(file_="configSGD.yml")

    with mlflow.start_run(experiment_id=get_.experiment_id):
        mlflow.autolog(
            log_input_examples=True,
        )
        sgd_ = SGDRegressor(**config.parameters)
        sgd_.fit(xtrain, ytrain)
        pred_ = sgd_.predict(xtrain)

        # registrando artefatos
        mlflow.log_artifacts(
            local_dir="./dataset2_/"
        )
       
   


if __name__ == "__main__":
    data = pd.read_csv("./dataset2_/wine-quality.csv", sep=';')

    X = data.drop(columns=["quality"])
    y = data["quality"]

    train_(X, y)
