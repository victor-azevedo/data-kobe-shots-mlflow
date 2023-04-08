import os
from datetime import datetime

import mlflow
import pandas as pd
import typer


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


def conform_and_select_data(data: pd.DataFrame, selection: list = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']) -> pd.DataFrame:
    data_conformed = data.dropna()
    data_conformed = data_conformed[selection]

    mlflow.log_metrics({
        'data_conformed_shape_0': data_conformed.shape[0],
        'data_conformed_shape_1': data_conformed.shape[1]})

    print(f"Dimensão dados conformados: {data_conformed.shape}")
    return data_conformed


def save_data_to_parquet(data: pd.DataFrame, sub_folder: str, filename: str) -> None:
    if not os.path.exists(f"Data/{sub_folder}"):
        os.mkdir(f"Data/{sub_folder}")

    file = f"Data/{sub_folder}/{filename}"
    data.to_parquet(path=file)

    print(f"Arquivo salvo em: {file}")


def crete_experiment(experiment_name: str) -> int:
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

    experiment_id = crete_experiment("PreparacaoDados")
    with mlflow.start_run(run_name=f"data-preparation-{(datetime.now()).strftime('%Y-%m-%dT%H:%M:%S')}", experiment_id=experiment_id):
        if shot_type == 2 or shot_type == 3:
            data = load_dataset(shot_type)
        else:
            print("shot_type inválido")
            return
        data_conformed = conform_and_select_data(data)
        if shot_type == 2:
            file_name = "data_filtered"
        else:
            file_name = "data_filtered_3pt"

        save_data_to_parquet(data_conformed, "processed", file_name)


if __name__ == "__main__":
    app()
