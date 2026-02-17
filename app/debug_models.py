"""
Script de Debug para Diagnosticar Problemas en los Modelos Avanzados
"""

import pandas as pd
import numpy as np
from advanced_predictor import AdvancedPredictor

def debug_single_mixing():
    """Debug de un solo mixing para identificar problemas"""
    
    # Cargar datos de prueba
    input_file = "C:/Users/81155875/Documents/Sale_In_Transit_Predictive_Model/src/test/input-data/Base Acumulado Sep-25.csv"
    monitoring = "sep-25"
    
    print("🔍 DEBUGGING MODELOS AVANZADOS")
    print("="*50)
    
    try:
        # Cargar datos
        data = pd.read_csv(input_file, low_memory=False)
        print(f"✅ Datos cargados: {len(data)} filas")
        
        # Filtrar por mes
        data = data[data['Mes Monitoreo'] == monitoring]
        print(f"✅ Datos filtrados: {len(data)} filas")
        
        # Obtener mixings
        mixings = data["Mixing Nombre"].unique()
        print(f"✅ Mixings encontrados: {list(mixings)}")
        
        # Probar con el primer mixing
        test_mixing = mixings[0]
        print(f"\n🎯 Probando con: {test_mixing}")
        
        # Crear predictor
        predictor = AdvancedPredictor(test_mixing, print)
        
        # Obtener datos del mixing
        mixing_data = data[data["Mixing Nombre"] == test_mixing].copy()
        print(f"📊 Datos del mixing: {len(mixing_data)} registros")
        
        # Verificar columnas
        print(f"📋 Columnas disponibles: {list(mixing_data.columns)}")
        
        # Crear variables
        print("\n🔧 Creando variables...")
        data_with_vars = predictor.create_variables(mixing_data)
        print(f"✅ Variables creadas: {len(data_with_vars)} registros")
        print(f"📋 Columnas después de crear variables: {list(data_with_vars.columns)}")
        
        # Verificar predictores
        available_predictors = [p for p in predictor.predictors if p in data_with_vars.columns]
        print(f"🎯 Predictores disponibles: {available_predictors}")
        
        # Crear pesos temporales
        print("\n⚖️ Creando pesos temporales...")
        w_array = predictor.create_temporal_weights(data_with_vars)
        print(f"✅ Pesos creados: {len(w_array)} valores")
        print(f"📊 Distribución de pesos: {np.unique(w_array, return_counts=True)}")
        
        # Entrenar modelo
        print("\n🤖 Entrenando modelo...")
        success = predictor.train_models(data_with_vars)
        
        if success:
            print(f"✅ Modelo entrenado exitosamente!")
            print(f"📊 Mejor modelo: {predictor.best_model_name}")
            print(f"📊 R² score: {predictor.r2_score:.4f}")
            
            # Probar predicción
            print("\n🔮 Probando predicción...")
            pred_simple = predictor.predict_simple(mixing_data)
            pred_16h, pred_21h = predictor.predict_double(mixing_data)
            
            print(f"✅ Predicción simple: {pred_simple:.4f}")
            print(f"✅ Predicción 16h: {pred_16h:.4f}")
            print(f"✅ Predicción 21h: {pred_21h:.4f}")
            
        else:
            print("❌ Error entrenando modelo")
            
    except Exception as e:
        print(f"❌ Error general: {str(e)}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_single_mixing()
