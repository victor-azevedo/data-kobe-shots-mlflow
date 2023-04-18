import os
import random
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import typer
from pycaret.classification import *
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             log_loss)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = random.randint(0, 100)
np.random.seed(SEED)


def create_train_lr_model(train_data, test_data, ):
    model_setup = setup(data=train_data, target='shot_made_flag',
                        test_data=test_data, session_id=SEED)
    add_metric('logloss', 'Log Loss', log_loss, greater_is_better=False)

    lr_model = create_model('lr', engine='sklearn')
    print(lr_model)

    register_metrics(test_data, lr_model)


def create_train_svm_model(train_data, test_data, ):
    model_setup = setup(data=train_data, target='shot_made_flag',
                        test_data=test_data, session_id=SEED)
    add_metric('logloss', 'Log Loss', log_loss, greater_is_better=False)

    svm_model = create_model('svm')
    print(svm_model)

    register_metrics(test_data, svm_model)


def create_train_ada_model(train_data, test_data, ):
    model_setup = setup(data=train_data, target='shot_made_flag',
                        test_data=test_data, session_id=SEED)
    add_metric('logloss', 'Log Loss', log_loss, greater_is_better=False)

    ada_model = create_model('ada')
    print(ada_model)

    register_metrics(test_data, ada_model)


def create_train_classification_model(train_data, test_data):
    model_setup = setup(data=train_data, target='shot_made_flag',
                        test_data=test_data, session_id=SEED)
    add_metric('logloss', 'Log Loss', log_loss, greater_is_better=False)

    best_model = compare_models()
    print(best_model)

    register_metrics(test_data, best_model)


def scaler_data(train_data, test_data):
    data = pd.concat([train_data, test_data], sort=False)
    x = data.drop('shot_made_flag', axis=1)

    x_train = train_data.drop('shot_made_flag', axis=1)
    x_test = test_data.drop('shot_made_flag', axis=1)

    scaler = StandardScaler()
    scaler.fit(x)

    train_data[['lat', 'lon', 'minutes_remaining', 'period',
                'playoffs', 'shot_distance']] = scaler.transform(x_train)
    test_data[['lat', 'lon', 'minutes_remaining', 'period',
               'playoffs', 'shot_distance']] = scaler.transform(x_test)

    return train_data, test_data


def register_metrics(test_data, model):
    x_test = test_data.drop('shot_made_flag', axis=1)
    y_test = test_data['shot_made_flag']

    y_pred = model.predict(x_test)
    f1_model = f1_score(y_test, y_pred)
    log_loss_model = log_loss(y_test, y_pred)
    accuracy_model = accuracy_score(y_test, y_pred)

    mlflow.log_metrics(
        {"f1": f1_model, "log_loss": log_loss_model, "accuracy": accuracy_model})
    print(classification_report(y_test, y_pred))


def create_experiment(experiment_name: str) -> int:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    return experiment_id


####################################################################
# Linha de Comando
####################################################################
app = typer.Typer()


@app.command()
def trainning(model: str = ''):

    experiment_id = create_experiment("Treinamento")
    with mlflow.start_run(run_name=f"data-preparation-{(datetime.now()).strftime('%Y-%m-%dT%H:%M:%S')}", experiment_id=experiment_id):
        mlflow.log_param('SEED', SEED)

        train_data = pd.read_parquet(path="Data/operalization/base_train")
        test_data = pd.read_parquet(path="Data/operalization/base_test")

        # train_data, test_data = scaler_data(train_data, test_data)
        mlflow.log_param('scaled', False)

        if (model == "lr"):
            print("Build lr")
            mlflow.log_param('Model', 'lr')

            create_train_lr_model(train_data, test_data)
        elif (model == "svm"):
            print("Build svm")
            mlflow.log_param('Model', 'svm')

            create_train_svm_model(train_data, test_data)
        elif (model == "ada"):
            print("Build ada")
            mlflow.log_param('Model', 'ada')

            create_train_ada_model(train_data, test_data)
        else:
            print("Build classification")
            mlflow.log_param('Model', 'best_PyCaret')

            create_train_classification_model(train_data, test_data)


if __name__ == "__main__":
    app()
