import glob
import os
import pandas as pd


input_path = 'input-data'


def get_latest_file():
    print(f"🔍 Buscando archivo más reciente en: {input_path}")
    print(f"📂 Ruta absoluta: {os.path.abspath(input_path)}")
    
    if not os.path.exists(input_path):
        print(f"❌ La carpeta {input_path} no existe!")
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    files_found = []
    most_recent_time = 0
    latest_file = None
    
    for entry in os.scandir(input_path):
        if entry.is_file():
            mod_time = entry.stat().st_mtime_ns
            files_found.append((entry.name, mod_time))
            if mod_time > most_recent_time:
                latest_file = entry.name
                most_recent_time = mod_time
    
    print(f"📋 Archivos encontrados en {input_path}: {len(files_found)}")
    for name, time in files_found:
        print(f"   - {name} (modificado: {time})")
    
    if latest_file:
        full_path = os.path.join(input_path, latest_file)
        print(f"✅ Archivo más reciente seleccionado: {latest_file}")
        print(f"📂 Ruta completa: {full_path}")
        return full_path
    else:
        print(f"❌ No se encontraron archivos en {input_path}")
        raise FileNotFoundError(f"No files found in {input_path}")


def clean_data(data_name, monitoring):
    print(f"📖 Leyendo archivo de datos: {data_name}")
    input_data = pd.read_csv(data_name, low_memory=False)
    print(f"✅ Datos cargados: {len(input_data)} filas, {len(input_data.columns)} columnas")
    print(f"📋 Columnas originales: {list(input_data.columns)}")
    
    #print(input_data) #AGREGUÉ LÍNEA
    print(f"🔍 Filtrando por mes de monitoreo: {monitoring}")
    print(f"📊 Valores únicos en 'Mes Monitoreo': {sorted(input_data['Mes Monitoreo'].unique())}")
    print(f"🔍 Buscando coincidencias exactas con '{monitoring}'...")
    
    # Verificar si hay coincidencias exactas
    exact_matches = input_data[input_data['Mes Monitoreo'] == monitoring]
    print(f"📈 Coincidencias exactas encontradas: {len(exact_matches)}")
    
    # Verificar coincidencias parciales
    partial_matches = input_data[input_data['Mes Monitoreo'].str.contains(monitoring, case=False, na=False)]
    print(f"📈 Coincidencias parciales (case-insensitive): {len(partial_matches)}")
    if len(partial_matches) > 0:
        print(f"📋 Valores con coincidencias parciales: {sorted(partial_matches['Mes Monitoreo'].unique())}")
    
    before_filter = len(input_data)
    input_data.drop(
        input_data[input_data['Mes Monitoreo'] != monitoring].index,
        inplace=True)
    after_filter = len(input_data)
    print(f"✅ Filtrado completado: {before_filter} → {after_filter} filas")
    
    if after_filter == 0:
        print(f"⚠️  ADVERTENCIA: No se encontraron datos para el mes '{monitoring}'")
        print(f"💡 Sugerencias:")
        print(f"   - Verificar el formato del mes en el archivo")
        print(f"   - Verificar si hay espacios extra o caracteres especiales")
        print(f"   - Verificar si el caso (mayúsculas/minúsculas) es correcto")
    
    print(f"🔧 Procesando columnas...")
    input_data["Horario"] = input_data["Horario.1"].astype("int")
    input_data["Monitoreo"] = input_data["Monitoreo"].astype("int")
    #input_data["Y_Retorno"].fillna(value=0, inplace=True)
    input_data["Y_Retorno"] = input_data["Y_Retorno"].fillna(value=0)
    print(f"✅ Columnas procesadas")
    
    print(f"📋 Seleccionando columnas específicas...")
    input_data = input_data[[
        "Mixing Nombre",
        "Mes Monitoreo",
        "Fecha de Corte",
        "FactA",
        "Y_Retorno",
        "Horario",
        "Monitoreo",
        "Facturación A",
        "Retornado A",
        "validacion"
        ]].copy(deep=False)
    print(f"✅ Columnas seleccionadas: {list(input_data.columns)}")
    
    print(f"🧹 Limpiando datos de validación...")
    before_validation = len(input_data)
    input_data["validacion"] = input_data["validacion"].str.replace(
        "Pasa",
        "pasa"
        )
    input_data.drop(
        input_data[input_data["validacion"] != "pasa"].index,
        inplace=True
        )
    after_validation = len(input_data)
    print(f"✅ Validación completada: {before_validation} → {after_validation} filas")
    
    print(f"🔧 Limpiando nombres de mixing...")
    input_data["Mixing Nombre"] = input_data["Mixing Nombre"].str.replace(
        " ",
        ""
        )
    input_data["Mixing Nombre"] = input_data["Mixing Nombre"].str.replace(
        "ó",
        "o"
        )
    print(f"✅ Nombres de mixing limpiados")
    
    print(f"🔍 Filtrando por monitoreo = 7...")
    before_monitoring = len(input_data)
    input_data.drop(
        input_data[input_data["Monitoreo"] != 7].index,
        inplace=True
        )
    after_monitoring = len(input_data)
    print(f"✅ Filtrado por monitoreo completado: {before_monitoring} → {after_monitoring} filas")
    
    print(f"📅 Procesando fechas...")
    input_data["Fecha de Corte"] = pd.to_datetime(
        input_data["Fecha de Corte"],
        yearfirst=False,
        dayfirst=True
        )
    print(f"✅ Fechas procesadas")
    
    print(f"💾 Guardando datos procesados...")
    input_data.to_csv("input_data.csv", encoding='cp1252')
    print(f"✅ Datos guardados en input_data.csv")
    
    print(f"📊 Resumen final de datos:")
    print(f"   - Filas totales: {len(input_data)}")
    print(f"   - Mixings únicos: {input_data['Mixing Nombre'].nunique()}")
    print(f"   - Mixings disponibles: {list(input_data['Mixing Nombre'].unique())}")
    
    return input_data


def join_data(folder, pred, monitoring):
    print(f"🔍 Buscando archivos CSV en: {folder}")
    print(f"📂 Ruta absoluta de búsqueda: {os.path.abspath(folder)}")
    
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    print(f"📋 Archivos encontrados con glob: {all_files}")
    
    '''df_from_each_mixing = (
        pd.read_csv(f, encoding="cp1252") for f in all_files
        )
    
    concatenated_df = pd.concat(df_from_each_mixing, ignore_index=True)'''
#--
    if not all_files:
        print(f"❌ No se encontraron archivos CSV en la carpeta")
        print(f"🔍 Verificando contenido de la carpeta:")
        if os.path.exists(folder):
            all_items = os.listdir(folder)
            print(f"   - Total items en carpeta: {len(all_items)}")
            print(f"   - Items encontrados: {all_items}")
            csv_items = [item for item in all_items if item.endswith('.csv')]
            print(f"   - Items con extensión .csv: {csv_items}")
        else:
            print(f"   - ❌ La carpeta {folder} no existe!")
        raise ValueError(f"No CSV files found in folder: {folder}")
   
    print(f"✅ Found {len(all_files)} files to process")
    print(f"📄 Archivos a procesar:")
    for i, file in enumerate(all_files, 1):
        print(f"   {i}. {file}")
   
    dataframes = []
    for i, file in enumerate(all_files, 1):
        print(f"\n📖 Leyendo archivo {i}/{len(all_files)}: {os.path.basename(file)}")
        try:
            df = pd.read_csv(file, encoding="cp1252")
            print(f"   ✅ Archivo leído exitosamente")
            print(f"   📊 Filas: {len(df)}, Columnas: {len(df.columns)}")
            print(f"   📋 Columnas: {list(df.columns)}")
            
            if not df.empty:
                dataframes.append(df)
                print(f"   ✅ DataFrame agregado a la lista")
            else:
                print(f"   ⚠️  Warning: Empty dataframe from file {file}")
        except Exception as e:
            print(f"   ❌ Error reading file {file}: {str(e)}")
            print(f"   🔍 Tipo de error: {type(e).__name__}")
            continue
   
    print(f"\n📈 Resumen de lectura:")
    print(f"   - Archivos procesados exitosamente: {len(dataframes)}/{len(all_files)}")
    
    if not dataframes:
        print(f"❌ No hay dataframes válidos para concatenar")
        raise ValueError("No valid dataframes to concatenate")
   
    print(f"🔗 Concatenando {len(dataframes)} dataframes...")
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    print(f"✅ DataFrame concatenado creado: {len(concatenated_df)} filas, {len(concatenated_df.columns)} columnas")
  #--  
    print(f"🧹 Limpiando datos...")
    concatenated_df.drop(
        concatenated_df[concatenated_df["Horario"] == 10].index,
        inplace=True
    )
    concatenated_df.drop(
        ["Y_Retorno", "Monitoreo", "Retornado A", "Facturación A"],
        axis=1,
        inplace=True
        )
    print(f"✅ Datos limpiados: {len(concatenated_df)} filas restantes")
    
    if pred == "doble":
        print(f"🎯 Procesando predicciones dobles...")
        first_pred = concatenated_df.drop(
            concatenated_df[concatenated_df["Horario"] == 21].index
            )
        first_pred.drop(["Y_Retorno_Pred_Doble"], axis=1, inplace=True)
        first_pred.columns = [
            "Mixing Nombre",
            "Fecha de Corte",
            "Horario",
            "Mes Monitoreo",
            "Predicción"
            ]
        second_pred = concatenated_df.drop(
            concatenated_df[concatenated_df["Horario"] == 16].index
            )
        second_pred.drop(["Y_Retorno_Pred"], axis=1, inplace=True)
        second_pred.columns = [
            "Mixing Nombre",
            "Fecha de Corte",
            "Horario",
            "Mes Monitoreo",
            "Predicción"
            ]
        preds = pd.concat([first_pred, second_pred], ignore_index=True)
        preds.sort_values(by=["Mixing Nombre", "Horario"], inplace=True)
        
        output_file = "Predicciones Dobles " + monitoring + ".csv"
        preds.to_csv(output_file, index=False, encoding="cp1252")
        print(f"✅ Archivo de predicciones dobles guardado: {output_file}")
        print(f"📊 Total predicciones: {len(preds)}")
    else:
        print(f"🎯 Procesando predicciones simples...")
        simple_pred = concatenated_df[concatenated_df["Horario"] == 21].copy()
        print(f"📋 Columnas antes de renombrar: {list(simple_pred.columns)}")
        required_columns = ["Mixing Nombre", "Fecha de Corte", "Horario", "Mes Monitoreo", "Y_Retorno_Pred"]
        simple_pred = simple_pred[required_columns].copy()
        
        simple_pred.columns = [
            "Mixing Nombre",
            "Fecha de Corte",
            "Horario",
            "Mes Monitoreo",
            "Predicción"
        ]
        
        output_file = "Predicciones Simples " + monitoring + ".csv"
        simple_pred.to_csv(output_file, index=False, encoding="cp1252")
        print(f"✅ Archivo de predicciones simples guardado: {output_file}")
        print(f"📊 Total predicciones: {len(simple_pred)}")
