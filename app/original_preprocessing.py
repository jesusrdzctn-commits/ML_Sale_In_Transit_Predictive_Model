"""
Replicación exacta del preprocessing del sistema original
"""

import pandas as pd
import os
import glob

def get_latest_file():
    """Obtener el archivo más reciente (simulado para la GUI)"""
    return "input_data.csv"  # Se pasará desde la GUI

def clean_data(data_name, monitoring):
    """Limpiar datos como en el original"""
    print(f"📖 Leyendo archivo de datos: {data_name}")
    input_data = pd.read_csv(data_name, low_memory=False)
    print(f"✅ Datos cargados: {len(input_data)} filas, {len(input_data.columns)} columnas")
    
    print(f"🔍 Filtrando por mes de monitoreo: {monitoring}")
    print(f"📊 Valores únicos en 'Mes Monitoreo': {sorted(input_data['Mes Monitoreo'].unique())}")
    
    # Filtrar por mes de monitoreo
    before_filter = len(input_data)
    input_data.drop(
        input_data[input_data['Mes Monitoreo'] != monitoring].index,
        inplace=True)
    after_filter = len(input_data)
    print(f"✅ Filtrado completado: {before_filter} → {after_filter} filas")
    
    if after_filter == 0:
        print(f"⚠️  ADVERTENCIA: No se encontraron datos para el mes '{monitoring}'")
        return pd.DataFrame()
    
    return input_data

def join_data(folder, pred, monitoring):
    """Unir datos como en el original"""
    print(f"🔍 Buscando archivos CSV en: {folder}")
    print(f"📂 Ruta absoluta de búsqueda: {os.path.abspath(folder)}")
    
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    print(f"📋 Archivos encontrados con glob: {all_files}")
    
    if not all_files:
        print(f"❌ No se encontraron archivos CSV en la carpeta")
        print(f"📁 Contenido de la carpeta: {os.listdir(folder) if os.path.exists(folder) else 'Carpeta no existe'}")
        raise ValueError(f"No CSV files found in folder: {folder}")
    
    print(f"✅ Found {len(all_files)} files to process")
    
    all_data = []
    
    for i, file in enumerate(all_files, 1):
        print(f"\n📖 Leyendo archivo {i}/{len(all_files)}: {os.path.basename(file)}")
        try:
            df = pd.read_csv(file, encoding="cp1252")
            print(f"   ✅ Archivo leído exitosamente")
            print(f"   📊 Filas: {len(df)}, Columnas: {len(df.columns)}")
            all_data.append(df)
        except Exception as e:
            print(f"   ❌ Error leyendo archivo: {str(e)}")
            continue
    
    if all_data:
        print(f"\n🔗 Uniendo {len(all_data)} archivos...")
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Crear nombre del archivo final
        if pred == "doble":
            output_file = os.path.join(folder, f"Predicciones Dobles {monitoring}.csv")
        else:
            output_file = os.path.join(folder, f"Predicciones Simples {monitoring}.csv")
        
        print(f"💾 Guardando archivo final: {output_file}")
        combined_data.to_csv(output_file, index=False, encoding="cp1252")
        print(f"✅ Archivo final creado exitosamente")
        print(f"📊 Total de registros en archivo final: {len(combined_data)}")
    else:
        print("❌ No se pudieron procesar archivos para unir")
