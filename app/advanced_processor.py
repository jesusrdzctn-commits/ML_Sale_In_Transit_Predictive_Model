"""
Procesador Avanzado que Usa el Sistema de Entrenamiento con Mayor Peso a los Últimos 12 Meses
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from advanced_predictor import AdvancedPredictor

class AdvancedProcessor:
    """Procesador avanzado que replica la lógica del sistema de entrenamiento"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.predictors = {}
    
    def log(self, message):
        if self.logger:
            self.logger(message)
        else:
            print(message)
    
    def clean_data(self, input_file, monitoring_month):
        """Limpiar datos como en el original"""
        self.log(f"📖 Leyendo archivo: {input_file}")
        
        try:
            data = pd.read_csv(input_file, low_memory=False)
            self.log(f"✅ Datos cargados: {len(data)} filas")
            
            # Filtrar por mes
            original_count = len(data)
            data = data[data['Mes Monitoreo'] == monitoring_month]
            filtered_count = len(data)
            
            self.log(f"🔍 Filtrado por '{monitoring_month}': {original_count} → {filtered_count} filas")
            
            if filtered_count == 0:
                self.log(f"❌ No se encontraron datos para el mes '{monitoring_month}'")
                return pd.DataFrame()
            
            return data
            
        except Exception as e:
            self.log(f"❌ Error leyendo archivo: {str(e)}")
            return pd.DataFrame()
    
    def train_all_models(self, input_data):
        """Entrenar modelos para todos los mixings con mayor peso a los últimos 12 meses"""
        self.log("🎯 Entrenando modelos avanzados con mayor peso a los últimos 12 meses...")
        
        mixings = input_data["Mixing Nombre"].unique()
        trained_count = 0
        
        for mixing in mixings:
            self.log(f"\n--- Entrenando Modelo para {mixing} ---")
            
            try:
                # Crear predictor
                predictor = AdvancedPredictor(mixing, self.log)
                
                # Obtener datos del mixing
                mixing_data = input_data[input_data["Mixing Nombre"] == mixing].copy()
                
                # Entrenar modelo
                if predictor.train_models(mixing_data):
                    self.predictors[mixing] = predictor
                    trained_count += 1
                    self.log(f"✅ Modelo entrenado exitosamente para {mixing}")
                else:
                    self.log(f"❌ No se pudo entrenar modelo para {mixing}")
                
            except Exception as e:
                self.log(f"❌ Error entrenando {mixing}: {str(e)}")
                self.log(f"   Tipo de error: {type(e).__name__}")
                import traceback
                self.log(f"   Traceback: {traceback.format_exc()}")
        
        self.log(f"\n📈 Resumen de entrenamiento:")
        self.log(f"   - Modelos entrenados exitosamente: {trained_count}/{len(mixings)}")
        
        return trained_count > 0
    
    def process_mixing(self, mixing_name, mixing_data, output_folder, prediction_type):
        """Procesar un mixing usando el modelo entrenado"""
        self.log(f"🎯 Procesando: {mixing_name}")
        
        try:
            # Obtener predictor entrenado
            predictor = self.predictors.get(mixing_name)
            
            if predictor is None:
                self.log(f"❌ No hay modelo entrenado para {mixing_name}")
                return False
            
            # Preparar datos de salida
            output_data = mixing_data.copy()
            
            if prediction_type == "simple":
                # Predicción simple
                pred_value = predictor.predict_simple(mixing_data)
                output_data['Y_Retorno_Pred'] = pred_value
                
                # Mostrar información del modelo
                self.log(f"📊 {mixing_name.upper()}: {predictor.best_model_name} (R²: {predictor.r2_score:.4f})")
                
            else:
                # Predicción doble
                pred_16h, pred_21h = predictor.predict_double(mixing_data)
                output_data['Y_Retorno_Pred_16h'] = pred_16h
                output_data['Y_Retorno_Pred_21h'] = pred_21h
                
                # Mostrar información del modelo
                self.log(f"📊 {mixing_name.upper()}: {predictor.best_model_name} (R²: {predictor.r2_score:.4f})")
            
            # Guardar archivo individual
            self._save_individual_file(output_data, mixing_name, output_folder, prediction_type)
            
            self.log(f"✅ {mixing_name} procesado exitosamente")
            return True
            
        except Exception as e:
            self.log(f"❌ Error procesando {mixing_name}: {str(e)}")
            return False
    
    def _save_individual_file(self, data, mixing_name, output_folder, prediction_type):
        """Guardar archivo individual por mixing"""
        output_file = os.path.join(output_folder, f"predicciones_{mixing_name.lower()}.csv")
        
        if prediction_type == "simple":
            columns = [
                "Mixing Nombre", "Fecha de Corte", "Y_Retorno", "Monitoreo", "Horario",
                "Facturación A", "Retornado A", "Mes Monitoreo", "Y_Retorno_Pred"
            ]
        else:
            columns = [
                "Mixing Nombre", "Fecha de Corte", "Y_Retorno", "Monitoreo", "Horario",
                "Facturación A", "Retornado A", "Mes Monitoreo", "Y_Retorno_Pred_16h", "Y_Retorno_Pred_21h"
            ]
        
        # Asegurar que todas las columnas existan
        for col in columns:
            if col not in data.columns:
                if col == "Retornado A":
                    data[col] = 0
                elif col in ["Y_Retorno_Pred", "Y_Retorno_Pred_16h", "Y_Retorno_Pred_21h"]:
                    data[col] = 0
                else:
                    data[col] = data.iloc[0].get(col, 0) if len(data) > 0 else 0
        
        # Guardar archivo individual
        data[columns].to_csv(output_file, index=False, encoding="cp1252")
    
    def join_data(self, folder, prediction_type, monitoring_month):
        """Unir datos y crear archivo consolidado como en el original"""
        self.log("🔗 Uniendo archivos de predicciones...")
        
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        
        if not csv_files:
            self.log("❌ No se encontraron archivos CSV para unir")
            return
        
        all_predictions = []
        
        for file in csv_files:
            try:
                df = pd.read_csv(file, encoding="cp1252")
                self.log(f"✅ Procesado: {os.path.basename(file)}")
                
                # Obtener el primer registro (solo necesitamos uno por SMC)
                if len(df) > 0:
                    first_row = df.iloc[0]
                    
                    # Extraer predicciones para el archivo consolidado
                    if prediction_type == "simple":
                        # Para predicción simple, solo horario 21h
                        if pd.notna(first_row.get('Y_Retorno_Pred', np.nan)):
                            all_predictions.append({
                                'Mixing Nombre': first_row['Mixing Nombre'],
                                'Fecha de Corte': '2024-01-01',  # Fecha fija como en el original
                                'Horario': 21,
                                'Mes Monitoreo': first_row['Mes Monitoreo'],
                                'Predicción': first_row['Y_Retorno_Pred']
                            })
                    else:
                        # Para predicción doble, horarios 16h y 21h
                        if pd.notna(first_row.get('Y_Retorno_Pred_16h', np.nan)):
                            all_predictions.append({
                                'Mixing Nombre': first_row['Mixing Nombre'],
                                'Fecha de Corte': '2024-01-01',  # Fecha fija como en el original
                                'Horario': 16,
                                'Mes Monitoreo': first_row['Mes Monitoreo'],
                                'Predicción': first_row['Y_Retorno_Pred_16h']
                            })
                        
                        if pd.notna(first_row.get('Y_Retorno_Pred_21h', np.nan)):
                            all_predictions.append({
                                'Mixing Nombre': first_row['Mixing Nombre'],
                                'Fecha de Corte': '2024-01-01',  # Fecha fija como en el original
                                'Horario': 21,
                                'Mes Monitoreo': first_row['Mes Monitoreo'],
                                'Predicción': first_row['Y_Retorno_Pred_21h']
                            })
                
            except Exception as e:
                self.log(f"❌ Error procesando {file}: {str(e)}")
        
        if all_predictions:
            # Crear DataFrame consolidado
            consolidated_df = pd.DataFrame(all_predictions)
            
            # Ordenar por Mixing Nombre y Horario
            consolidated_df = consolidated_df.sort_values(['Mixing Nombre', 'Horario'])
            
            # Crear nombre del archivo final
            if prediction_type == "doble":
                output_file = os.path.join(folder, f"Predicciones Dobles {monitoring_month}.csv")
            else:
                output_file = os.path.join(folder, f"Predicciones Simples {monitoring_month}.csv")
            
            # Guardar archivo consolidado
            consolidated_df.to_csv(output_file, index=False, encoding="cp1252")
            self.log(f"✅ Archivo final creado: {output_file}")
            self.log(f"📊 Total de predicciones: {len(consolidated_df)}")
        else:
            self.log("❌ No se pudieron procesar predicciones para unir")
