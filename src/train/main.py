import glob
import os
import pandas as pd
from datetime import datetime

from data_cleaning import *
from modeling import modeling
from smcs import *
from setup import env_vars


def get_current_month():
    """Obtiene el mes en curso en formato abreviado."""
    current_date = datetime.now()
    month_abbr = current_date.strftime('%b').lower()[:3]
    year = current_date.strftime('%y')
    
    # Mapeo para asegurar formato consistente
    month_map = {
        'jan': 'ene', 'feb': 'feb', 'mar': 'mar', 'apr': 'abr',
        'may': 'may', 'jun': 'jun', 'jul': 'jul', 'aug': 'ago',
        'sep': 'sep', 'oct': 'oct', 'nov': 'nov', 'dec': 'dic'
    }
    
    return f"{month_map[month_abbr]}-{year}"


def init_mode():
    """This function introduces the code."""
    print(f"\n      Training PMF Revenue Recognition Predictive Model      ")
    print(f"Reading Training Data from: {env_vars['input_data_dir']}")
    most_recent_time = 0
    for entry in os.scandir(env_vars['input_data_dir']):
        if entry.is_file():
            mod_time = entry.stat().st_mtime_ns
            if mod_time > most_recent_time:
                latest_file = entry.name
                most_recent_time = mod_time
    print(f"File name: {os.path.basename(latest_file)}")


def main():
    init_mode()
    
    # Obtener mes en curso
    current_month = get_current_month()
    print(f"Mes en curso: {current_month}")
    
    # Crear estructura de carpetas: ModelosEntrenados/mes-año/
    base_models_path = 'ModelosEntrenados'
    current_month_path = os.path.join(base_models_path, current_month)
    
    # Crear carpetas si no existen
    if not os.path.exists(base_models_path):
        os.makedirs(base_models_path)
        print(f"Carpeta base creada: {os.path.abspath(base_models_path)}")
    
    if not os.path.exists(current_month_path):
        os.makedirs(current_month_path)
        print(f"Carpeta del mes creada: {os.path.abspath(current_month_path)}")
    else:
        print(f"Usando carpeta existente: {os.path.abspath(current_month_path)}")
    
    # Establecer variables globales para que las use BaseModel
    import base
    base.GLOBAL_MODELS_PATH = current_month_path
    base.GLOBAL_CURRENT_MONTH = current_month
    
    # Crear archivo de log de auditoría en la carpeta del mes
    log_filename = f"log_entrenamiento_{current_month}.txt"
    log_path = os.path.join(current_month_path, log_filename)
    
    # Escribir encabezado del log
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"=== LOG DE ENTRENAMIENTO - {current_month} ===\n")
        f.write(f"Fecha y hora de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Carpeta de modelos: {os.path.abspath(current_month_path)}\n")
        f.write("=" * 50 + "\n\n")
    
    print(f"Log de auditoría creado: {os.path.abspath(log_path)}")
    
    data = get_clean_data(env_vars['input_data_dir'])
    mixings = data["Mixing Nombre"].unique()
    
    # Escribir información del dataset en el log
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"Dataset cargado: {len(data)} registros\n")
        f.write(f"Mixings encontrados: {len(mixings)}\n")
        f.write(f"Mixings: {', '.join(mixings)}\n\n")
    
    for mixing in mixings:
        print(f"Training {mixing} Model")
        
        # Registrar inicio del entrenamiento en el log
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando entrenamiento: {mixing}\n")
        
        mixing_data = data.drop(
            data[data['Mixing Nombre'] != mixing].index
            )
        smc_modeling = globals()[mixing]()
        modeling(mixing_data, smc_modeling)
        
        # Registrar fin del entrenamiento en el log
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Finalizado entrenamiento: {mixing}\n\n")
    
    # Escribir resumen final en el log
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write(f"ENTRENAMIENTO COMPLETADO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de modelos entrenados: {len(mixings)}\n")
        f.write("=" * 50 + "\n")
    
    print(f"Process completed. Log guardado en: {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
