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

def train(X_, input_schema, output_schema, input_example_):
    
    # Importando parametros
    params = OmegaConf.load("config.yml")
    params = params['parameters']

    # Definindo o track

    mlflow.set_tracking_uri(uri="./mytracks")

    try:
        exp = mlflow.create_experiment(name="LogisticRegression")
        get_ = mlflow.get_experiment(exp)

    except mlflow.exceptions.MlflowException as e:
        get_ = mlflow.set_experiment(experiment_name="LogisticRegression")

    train, test = train_test_split(X_, test_size=0.25, random_state=40)

    xtrain, xtest = train.drop("target", axis=1), test.drop("target", axis=1)
    ytrain, ytest = train["target"], test["target"]

    with mlflow.start_run(experiment_id=get_.experiment_id) as run:
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
        mlflow.log_params(model.get_params())

        signature = ModelSignature(
            inputs=input_schema,
            outputs=output_schema
        )
        #log model
        mlflow.sklearn.log_model(model, name="model_logistic_regression", signature=signature, input_example=input_example_)

        # uri
        model = f'runs:/{run.info.run_id}/model_logistic_regression'
        mlflow.evaluate(
            model=model,
            data=test,
            targets='target',
            model_type="classifier",
            evaluator_config={"label_list": [0,1,2]}

        )


        

def cases_(type_):
    if type_ == 'float64':
        return 'double'
    elif type_ == 'int64':
        return 'integer'
    else:
        return 'string'
    
if __name__ == "__main__":
    data = load_wine(as_frame=True)['data']
    target = load_wine(as_frame=True)['target']

    input_example = data.sample(n=6, random_state=40)
    input_example = input_example.to_dict(orient='list')
    input_schema = Schema([
        ColSpec(type=cases_(type_), name=name_)
        for type_, name_ in zip(data.dtypes, data.columns)
    ])
    output_schema = Schema([
        ColSpec(type=cases_(target.dtypes))
    ])

    X = data
    y = target
    X_ = pd.concat([X, y], axis=1)
    train(X_, input_schema, output_schema, input_example)