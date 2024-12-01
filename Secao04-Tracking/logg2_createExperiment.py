import warnings
import argparse
import logging

# --------
import pandas as pd
import numpy as np

#----------
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

#--------- Importando mlflow
import mlflow
import mlflow.sklearn as mlsk

logging.basicConfig(
    level=logging.WARN
)
logger = logging.getLogger(__name__)

"""
e é uma biblioteca do Python que facilita a criação de interfaces de 
linha de comando (CLI) para seus scripts.  Ele permite que você defina 
quais argumentos (ou parâmetros) o script pode receber e os processa 
automaticamente para que você os utilize dentro do programa.
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "--alpha",
    type = float,
    required = False,
    default = 0.5
)
parser.add_argument(
    "--l1_ratio",
    type = float,
    required = False,
    default = 0.5
)
args = parser.parse_args()


# Metricas
def metricas(ytrue, ypred):
    rmse = root_mean_squared_error(ytrue, ypred)
    mae  = mean_absolute_error(ytrue, ypred)
    r2_  = r2_score(ytrue, ypred)

    return rmse, mae, r2_


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    url = (
        "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-red.csv"
    )

    try:
        data_ = pd.read_csv(url, sep=';')

    except Exception as e:
        logger.exception(
            "OBS: Algo deu de errado", 0
        )
    train, test = train_test_split(data_)

    xtrain, ytrain = (
        train.drop(columns=["quality"]),
        train["quality"]
    )

    xtest, ytest = (
        test.drop(columns=["quality"]),
        test["quality"]
    )

    alpha = args.alpha
    l1_ratio = args.l1_ratio
    # Definindo o URI
    mlflow.set_tracking_uri(
        uri="./mytracks"
    )
    print("Uri :", mlflow.get_tracking_uri())

    # Definindo um nome do experimento
    exp_id = mlflow.create_experiment(
        name="Elastic_",
        tags=dict(version_ = "1.0.0")
        )

    get_ = mlflow.get_experiment(experiment_id=exp_id)

    str_ = f"""
Name..............: {get_.name}
Id................: {get_.experiment_id}
Tag...............: {get_.tags}
LyfeCycle_Stage...: {get_.lifecycle_stage}
Artifact Location : {get_.artifact_location}
Timestamp Create..: {get_.creation_time}
"""
    print(str_)
    with mlflow.start_run(experiment_id=get_.experiment_id):
        elastic = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, random_state=32
        )
        elastic.fit(xtrain, ytrain)
    
        pred_ = elastic.predict(xtest)
    
        
        rmse, mae, r2 = metricas(ytest, pred_)
        
        print(elastic.get_params)
        print("RMSE :",rmse)
        print("MAE  :",mae)
        print("R2   :",r2)

        # Logando os paramatros do modelos
        mlflow.log_params(
            {"alpha":alpha, "l1_ratio":l1_ratio}
        )

        # Logando metricas
        mlflow.log_metrics(
            dict(
                root_mean_squared_error_ = rmse,
                mean_absolute_error_     = mae,
                r2_score_                = r2
            )
        )

        # Rastreando modelo treinado
        sign = mlflow.models.infer_signature(
            xtrain, pred_
        )
        mlsk.log_model(
            elastic, "modelo_elasticNet", signature=sign
        )
