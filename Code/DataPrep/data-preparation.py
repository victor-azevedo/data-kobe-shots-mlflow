import os
from datetime import datetime

import mlflow
import pandas as pd
import typer
from sklearn.model_selection import train_test_split


def load_dataset(shot_type: int = 2) -> pd.DataFrame | None:
    if shot_type == 2:
        shot_type_str = "2PT Field Goal"
    elif shot_type == 3:
        shot_type_str = "3PT Field Goal"
    else:
        print("shot_type inválido")
        return

    data = pd.read_csv("Data/kobe_dataset.csv")
    data = data.loc[data['shot_type'] == shot_type_str]

    mlflow.log_param('shot_type', shot_type_str)
    mlflow.log_metrics({
        'data_shot_type_shape_0': data.shape[0],
        'data_shot_type_shape_1': data.shape[1]})

    print(f"Dimensão dados de {shot_type} pontos: {data.shape}")
    return data


def conform_data(data: pd.DataFrame, shot_type: int, selection: list = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']) -> pd.DataFrame:
    data_conformed = data.dropna()
    data_conformed = data_conformed[selection]

    mlflow.log_metrics({
        'data_conformed_shape_0': data_conformed.shape[0],
        'data_conformed_shape_1': data_conformed.shape[1]})

    print(f"Dimensão dados conformados: {data_conformed.shape}")

    if shot_type == 2:
        file_name = "data_filtered"
    else:
        file_name = "data_filtered_3pt"
    save_data_to_parquet(data_conformed, "processed", file_name)
    return data_conformed


def split_data(data: pd.DataFrame, test_size: int = 20) -> None:
    data_train, data_test = train_test_split(
        data, test_size=test_size/100, stratify=data['shot_made_flag'])

    mlflow.log_param('teste_percent', f"{test_size}%")

    mlflow.log_metrics({
        'base_train_size': data_train.shape[0],
        'base_test_size': data_test.shape[0]})

    save_data_to_parquet(
        data=data_train, sub_folder="operalization", filename="base_train")
    save_data_to_parquet(
        data=data_test, sub_folder="operalization", filename="base_test")


def save_data_to_parquet(data: pd.DataFrame, sub_folder: str, filename: str) -> None:
    if not os.path.exists(f"Data/{sub_folder}"):
        os.mkdir(f"Data/{sub_folder}")

    file = f"Data/{sub_folder}/{filename}"
    data.to_parquet(path=file)

    print(f"Arquivo salvo em: {file}")


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
def load_data(shot_type: int = 2):

    experiment_id = create_experiment("PreparacaoDados")
    with mlflow.start_run(run_name=f"data-preparation-{(datetime.now()).strftime('%Y-%m-%dT%H:%M:%S')}", experiment_id=experiment_id):
        if shot_type == 2 or shot_type == 3:
            data = load_dataset(shot_type)
        else:
            print("shot_type inválido")
            return

        data_conformed = conform_data(data=data, shot_type=shot_type)

        split_data(data=data_conformed, test_size=20)


if __name__ == "__main__":
    app()
