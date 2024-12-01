import mlflow.models
import mlflow.models
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics  import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import omegaconf

import mlflow
import mlflow.sklearn



def train_(X, y):

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=40)
    mlflow.set_tracking_uri(
        uri="./mytracks"
    )
    print("Uri :", mlflow.get_tracking_uri())
    try:
        exp_ = mlflow.create_experiment(
            name = "GradientBoss",
            tags = dict(version_ = "1.0.0")
        )
        get_ = mlflow.get_experiment(experiment_id=exp_)
    except mlflow.exceptions.MlflowException as e:
        get_ = mlflow.set_experiment(experiment_name="GradientBoss")

    str_ = f"""
Name..............: {get_.name}
Id................: {get_.experiment_id}
Tag...............: {get_.tags}
LyfeCycle_Stage...: {get_.lifecycle_stage}
Artifact Location : {get_.artifact_location}
Timestamp Create..: {get_.creation_time}
"""
    print(str_)
    #mlflow.start_run(experiment_id=get_.experiment_id, run_name="run_1"):
    with mlflow.start_run(run_id="c0ac8f8bcb594410a03d9bfe33697368"):
        # Carregando a config.yml
        config = omegaconf.OmegaConf.load(file_="config.yml")
        
        boost_ = GradientBoostingRegressor(**config.parameters)
        boost_.fit(xtrain, ytrain)

        # Log Metrics
        mlflow.log_metrics(
            dict(
                mean_squared_error_ = mean_squared_error(ytrain, boost_.predict(xtrain)),
                r2_score_           = r2_score(ytrain, boost_.predict(xtrain))
            )
        )

        # Logando parametros do modelo
        mlflow.log_params(
            config.parameters
        )

        # Carreando modelo

        sign = mlflow.models.infer_signature(
            xtrain, boost_.predict(xtrain)
        )
        mlflow.sklearn.log_model(
             boost_,"GradientBoostingRegressor_2", signature=sign
        )
if __name__ == "__main__":
    X, y = fetch_california_housing(return_X_y=True)
    train_(X, y)
