from   sklearn.datasets import fetch_california_housing
from   sklearn.model_selection import train_test_split
from   sklearn.metrics  import r2_score, mean_squared_error, mean_absolute_error
import keras
import mlflow
import pathlib
from omegaconf.omegaconf import OmegaConf


from models_params import *
from plots import *  

def train(X, y):
    # Importando parametros

    params = OmegaConf.load("config.yml")
    params = params['parameters']
    # Criando esperimento
    mlflow.set_tracking_uri(uri="./mytracks")

    try:
        exp = mlflow.create_experiment(
            name = "RNA-Regressor-keras",
            tags = dict(version_ = "1.0.0")
        )
        get_ = mlflow.get_experiment(experiment_id=exp)
    
    except mlflow.exceptions.MlflowException:
        get_ = mlflow.set_experiment(experiment_name="RNA-Regressor-keras")

    # Inciando experimento
    with mlflow.start_run(experiment_id=get_.experiment_id) as run:

        # Divindo os dados de treino e teste
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=40)
        print(f"""
Train size = {xtrain.shape}
test size = {xtest.shape}
""")

        model = keras.models.Sequential()
        model.add(keras.layers.BatchNormalization())

        for i in params['hidden_layers']:
            model.add(keras.layers.Dense(
                units=i,
                kernel_initializer=kernal_initializer(type_=params['type_initializer_from_activation']),
            ))
            model.add(
                activations(type_=params['activation'])
            )

        model.add(keras.layers.Dense(1))

        metrics_ = metrics(type_=params['metrics'])
        losses_  = loss_(type_=params['loss'])
        optim    = optimizers(type_=params['optimizer'], lr=params['learning_rate'])
        
        model.compile(
            optimizer=optim, 
            loss=losses_, 
            metrics=metrics_, 
        )

        model.fit(
            xtrain, ytrain, 
            epochs=params['epochs'], 
            batch_size=params['batch_size'],
            validation_split=params['validation_split'],
            callbacks=[mlflow.keras.MlflowCallback(run)]
        )

        # Metricas
        y_pred = model.predict(xtest)
        plot_errors(ytest, y_pred)
        
        mlflow.log_artifact('plots/prediction_vs_real.png')
        mlflow.log_artifact('plots/residuos.png')
        mlflow.log_artifact('plots/hist_residuos.png')

        r2 = r2_score(ytest, y_pred)
        mse = mean_squared_error(ytest, y_pred)
        mae = mean_absolute_error(ytest, y_pred)
        
        # Logando parametros
        mlflow.log_params(params)
        # Logando metricas
        mlflow.log_metrics({"r2_score": r2, "erro_quadratico_medio": mse, "erro_absoluto": mae})

        signature = mlflow.models.infer_signature(xtrain, model.predict(xtrain))

        mlflow.keras.log_model(model, name="model_keras", signature=signature)

if __name__ == "__main__":
    X, y = fetch_california_housing(return_X_y=True)
    
    train(X, y)