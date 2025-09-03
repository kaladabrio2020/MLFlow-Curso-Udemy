import mlflow
import logging
import pandas as pd 
from   sklearn.datasets import fetch_california_housing
from   sklearn.model_selection import train_test_split
from   sklearn.ensemble import GradientBoostingRegressor
from   sklearn.metrics  import r2_score, mean_squared_error, mean_absolute_error
import pathlib

from omegaconf import OmegaConf
BASE_DIR = pathlib.Path(__file__).resolve().parent

def main():
    mlflow.set_tracking_uri(
        uri= BASE_DIR / "mytracks"
    )
    params = OmegaConf.load(BASE_DIR / "config.yml")

    params = params['parameters']
    params['random_state'] = 42

    print("Uri :", mlflow.get_tracking_uri())

    try:
        exp = mlflow.create_experiment(
            name = "GradientBoss",
            tags = dict(version_ = "1.0.0")
        )
        get_ = mlflow.get_experiment(experiment_id=exp)
    except mlflow.exceptions.MlflowException as e:
        get_ = mlflow.set_experiment(experiment_name="GradientBoss")

    with mlflow.start_run(experiment_id=get_.experiment_id):
        X, y = fetch_california_housing(return_X_y=True, as_frame=False)

        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=40)

        model = GradientBoostingRegressor(**params)
        model.fit(xtrain, ytrain)
        y_pred = model.predict(xtest)

        r2 = r2_score(ytest, y_pred)
        mae = mean_absolute_error(ytest, y_pred)
        mse = mean_squared_error(ytest, y_pred)


        mlflow.log_params(
            model.get_params()
        )
        mlflow.log_metrics(
            {
                "r2": r2,
                "mae": mae,
                "mse": mse
            }
        )
        sign = mlflow.models.infer_signature(
            xtrain,
            model.predict(xtrain)
        )
        mlflow.sklearn.log_model(model, name="model_gradientBoss", signature=sign)
        mlflow.log_artifact(BASE_DIR / "config.yml")
    

if __name__ == "__main__":
    main()
