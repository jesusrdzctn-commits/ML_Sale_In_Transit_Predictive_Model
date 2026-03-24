import os
import sys

from mixings import *
from preprocessing import *
from robust_model_wrapper import RobustModelWrapper #NEW


def main():
    dataset = get_latest_file()
    monitoring = sys.argv[1] #input("Mes:")
    prediction = sys.argv[2] #input("dobles: ")
    print("="*50)
    print("INICIO DEL PROCESO")
    print("="*50)
    print("Leyendo base de datos " + dataset)
    print("Monitoreo " + monitoring)
    print("Tipo de predicción: " + prediction)
    
    if prediction == "doble":
        folder = "Predicciones Dobles " + monitoring
        check = os.path.isdir(folder)
        if check:
            print(f"✓ Carpeta existente: {folder}")
        else:
            os.mkdir(folder)
            print(f"✓ Carpeta creada: {folder}")
    elif prediction == "mediodia":
        folder = "Predicciones Mediodia " + monitoring
        check = os.path.isdir(folder)
        if check:
            print(f"✓ Carpeta existente: {folder}")
        else:
            os.mkdir(folder)
            print(f"✓ Carpeta creada: {folder}")
    else:
        folder = "Predicciones Simples " + monitoring
        check = os.path.isdir(folder)
        if check:
            print(f"✓ Carpeta existente: {folder}")
        else:
            os.mkdir(folder)
            print(f"✓ Carpeta creada: {folder}")
    
    print(f"📁 Carpeta de destino: {folder}")
    print(f"📂 Ruta absoluta: {os.path.abspath(folder)}")
            
    input_data = clean_data(dataset, monitoring)
    mixins = input_data["Mixing Nombre"].unique()
    print(f"🔍 Mixings encontrados: {len(mixins)}")
    print(f"📋 Lista de mixings: {list(mixins)}")
    
    files_created = 0
    for i, mixin in enumerate(mixins, 1):
        print(f"\n--- Procesando Mixing {i}/{len(mixins)} ---")
        print(f"🎯 Generando Predicciones de SMC: {mixin}")
        mixin_data = input_data[input_data["Mixing Nombre"] == mixin]
        print(f"📊 Datos para {mixin}: {len(mixin_data)} registros")
        
        try:
            #print(globals()) NEW
            #input("globals") NEW 
            model = globals()[mixin](mixin, mixin_data, folder)
            print(f"✅ Modelo creado para {mixin}")
            model.run_model(prediction)
            print(f"✅ Predicciones generadas para {mixin}")
            files_created += 1
        except Exception as e:
            print(f"❌ ERROR procesando {mixin}: {str(e)}")
            print(f"🔍 Tipo de error: {type(e).__name__}")
    
    print(f"\n📈 Resumen de procesamiento:")
    print(f"   - Mixings procesados exitosamente: {files_created}/{len(mixins)}")
    
    # Verificar archivos creados antes de join_data
    print(f"\n🔍 Verificando archivos en carpeta antes de unir:")
    if os.path.exists(folder):
        files_in_folder = os.listdir(folder)
        csv_files = [f for f in files_in_folder if f.endswith('.csv')]
        print(f"   - Total archivos en carpeta: {len(files_in_folder)}")
        print(f"   - Archivos CSV encontrados: {len(csv_files)}")
        if csv_files:
            print(f"   - Archivos CSV: {csv_files}")
        else:
            print(f"   - ⚠️  No hay archivos CSV en la carpeta!")
    else:
        print(f"   - ❌ La carpeta {folder} no existe!")
    
    print("\n" + "="*50)
    print("UNIENDO ARCHIVOS")
    print("="*50)
    print("Creando archivo final")
    join_data(folder, prediction, monitoring)
    print("✅ Proceso terminado exitosamente.")


if __name__ == "__main__":
    main()
