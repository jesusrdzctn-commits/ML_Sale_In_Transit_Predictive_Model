import glob
import os
import pandas as pd

from setup import info_cols


def correct_str(data):
    """This function corrects string values."""
    data["Mixing Nombre"] = data["Mixing Nombre"].str.replace(" ", "")
    data["Mixing Nombre"] = data["Mixing Nombre"].str.replace("ó", "o")
    return data


def fill_na(data):
    "This function fills missing values in Y_Retorno."
    data["Y_Retorno"] = data["Y_Retorno"].fillna(value=0)
    return data


def format_data(data):
    """This function deals with dtypes."""
    data["Fecha de Corte"] = pd.to_datetime(
        data["Fecha de Corte"],
        yearfirst=False,
        dayfirst=True)
    data["Monitoreo"] = data["Monitoreo"].astype("int")
    return data


def get_clean_data(input_data_dir):
    """This function prepares data to be used."""
    data = read_data(input_data_dir)
    data = get_info_data(data, info_cols)
    data = correct_str(data)
    data = fill_na(data)
    data = format_data(data)
    write_results(data)
    return data


def get_info_data(data, info_cols):
    """This function drops useless columns and rows."""
    data.drop(
        columns=[col for col in data if col not in info_cols],
        inplace=True
        )
    data.drop(
        data[data["Mixing Nombre"] == "Maizoro"].index,
        inplace=True
        )
    data.drop(
        data[data["Mixing Nombre"] == "Vallejo"].index,
        inplace=True
        )
    data = data[~data.validacion.isin(["Fuera", "fuera"])]
    return data


def read_data(input_data_dir):
    """This function reads .csv file into Pandas DataFrame."""
    most_recent_time = 0
    for entry in os.scandir(input_data_dir):
        if entry.is_file():
            mod_time = entry.stat().st_mtime_ns
            if mod_time > most_recent_time:
                latest_file = entry.name
                most_recent_time = mod_time
    latest_file = os.path.join(input_data_dir, latest_file)
    data = pd.read_csv(
        latest_file,
        encoding='cp1252',
        low_memory=False)
    return data


def write_results(data):
    monitoring = data["Mes Monitoreo"].iloc[-1]
    with open(f"Resultados {monitoring}.txt", 'w') as f:
        f.write(f"---------- Resultados {monitoring} ----------")
