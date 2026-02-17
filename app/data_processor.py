"""
Procesador de Datos Robusto
Replica exactamente la lógica de los sistemas originales
"""

import pandas as pd
import numpy as np
import os
import glob
from predictive_models import create_predictive_model

class DataProcessor:
    """Procesador principal de datos"""
    
    def __init__(self, logger=None):
        self.logger = logger
        
        # Configuración de mixings (basada en los sistemas originales)
        self.mixings_config = {
            'azcapotzalco': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion'],
                'filters': {'Y_Retorno': lambda x: x >= -50}
            },
            'celaya': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'guadalajara': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'merida': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'monterrey': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'nexxus': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'nexxuscap': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'obregon': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion', 'Facturacion_Monitoreo'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'plantaobregon': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'porteo': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion', 'Monitoreo_Horario'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'ptamerida': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion', 'Monitoreo_Horario'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'sanmartin': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'tijuana': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'centerobregon': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'smcpuebla': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'centersaltillo': {
                'predictors': ['Monitoreo', 'Horario', 'FactA', 'Facturacion Anterior', 'Retornado Anterior', 'Monitoreo_Facturacion'],
                'filters': {'Y_Retorno': lambda x: x <= 100}
            },
            'vallejo': {
                'predictors': [],
                'filters': {}
            }
        }
    
    def log(self, message):
        """Función de logging"""
        if self.logger:
            self.logger(message)
        else:
            print(message)
    
    def clean_data(self, input_file, monitoring_month):
        """Limpiar y filtrar datos de entrada"""
        self.log(f"📖 Leyendo archivo: {input_file}")
        
        try:
            data = pd.read_csv(input_file, low_memory=False)
            self.log(f"✅ Datos cargados: {len(data)} filas")
            
            # Filtrar por mes de monitoreo
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
    
    def create_variables(self, data, mixing_name):
        """Crear variables específicas para cada mixing (como en el sistema original)"""
        data = data.copy()
        
        # Variables comunes
        data = self._previous_invoicing(data)
        data = self._previous_y(data)
        data = self._monitoring_times_invoicing(data)
        
        # Variables específicas por mixing
        try:
            if mixing_name.lower() in ['obregon', 'ptamerida', 'porteo']:
                data['FactA'] = pd.to_numeric(data['FactA'], errors='coerce').fillna(0)
                data['Monitoreo'] = pd.to_numeric(data['Monitoreo'], errors='coerce').fillna(0)
                data['Facturacion_Monitoreo'] = data['FactA'] * data['Monitoreo']
            
            if mixing_name.lower() in ['ptamerida', 'porteo']:
                data['Monitoreo'] = pd.to_numeric(data['Monitoreo'], errors='coerce').fillna(0)
                data['Horario'] = pd.to_numeric(data['Horario'], errors='coerce').fillna(0)
                data['Monitoreo_Horario'] = data['Monitoreo'] * data['Horario']
        except Exception as e:
            self.log(f"❌ Error creando variables específicas para {mixing_name}: {str(e)}")
            if mixing_name.lower() in ['obregon', 'ptamerida', 'porteo']:
                data['Facturacion_Monitoreo'] = 0
            if mixing_name.lower() in ['ptamerida', 'porteo']:
                data['Monitoreo_Horario'] = 0
        
        return data
    
    def _previous_invoicing(self, data):
        """Crear variable de facturación anterior"""
        try:
            data = data.sort_values('Fecha de Corte')
            data['FactA'] = pd.to_numeric(data['FactA'], errors='coerce').fillna(0)
            data['Facturacion Anterior'] = data['FactA'].shift(1).fillna(0)
        except Exception as e:
            self.log(f"❌ Error en _previous_invoicing: {str(e)}")
            data['Facturacion Anterior'] = 0
        return data
    
    def _previous_y(self, data):
        """Crear variable de retorno anterior"""
        try:
            data = data.sort_values('Fecha de Corte')
            data['Y_Retorno'] = pd.to_numeric(data['Y_Retorno'], errors='coerce').fillna(0)
            data['Retornado Anterior'] = data['Y_Retorno'].shift(1).fillna(0)
        except Exception as e:
            self.log(f"❌ Error en _previous_y: {str(e)}")
            data['Retornado Anterior'] = 0
        return data
    
    def _monitoring_times_invoicing(self, data):
        """Crear variable de interacción monitoreo-facturación"""
        try:
            # Asegurar que las columnas sean numéricas
            data['Monitoreo'] = pd.to_numeric(data['Monitoreo'], errors='coerce').fillna(0)
            data['FactA'] = pd.to_numeric(data['FactA'], errors='coerce').fillna(0)
            data['Monitoreo_Facturacion'] = data['Monitoreo'] * data['FactA']
        except Exception as e:
            self.log(f"❌ Error en _monitoring_times_invoicing: {str(e)}")
            data['Monitoreo_Facturacion'] = 0
        return data
    
    def process_mixing(self, mixing_name, mixing_data, output_folder, prediction_type):
        """Procesar un mixing específico"""
        self.log(f"🎯 Procesando: {mixing_name}")
        
        try:
            self.log(f"🔍 Datos originales: {len(mixing_data)} filas")
            self.log(f"🔍 Columnas disponibles: {list(mixing_data.columns)}")
            
            # Verificar tipos de datos problemáticos
            for col in ['Monitoreo', 'FactA', 'Horario', 'Y_Retorno']:
                if col in mixing_data.columns:
                    sample_val = mixing_data[col].iloc[0] if len(mixing_data) > 0 else None
                    self.log(f"🔍 {col}: tipo={type(sample_val)}, valor={sample_val}")
            
            # Crear variables
            self.log(f"🔍 Creando variables para {mixing_name}...")
            processed_data = self.create_variables(mixing_data, mixing_name)
            self.log(f"✅ Variables creadas exitosamente")
            
            # Caso especial para Vallejo
            if mixing_name.lower() == 'vallejo':
                return self._process_vallejo(processed_data, output_folder, prediction_type)
            
            # Crear predictor
            predictor = create_predictive_model(mixing_name)
            
            # Agregar datos históricos
            for _, row in processed_data.iterrows():
                if not pd.isna(row.get('Y_Retorno', np.nan)):
                    predictor.add_historical_data(row.to_dict())
            
            # Entrenar modelo
            if predictor.train_model():
                model_info = predictor.get_model_info()
                if model_info:
                    self.log(f"📊 {mixing_name.upper()}: {model_info['model_name']} (R²: {model_info['r2_score']:.4f})")
            
            # Hacer predicciones
            if prediction_type == "simple":
                predictions = self._predict_simple(processed_data, predictor)
                processed_data['Y_Retorno_Pred'] = predictions
            else:
                predictions_16h, predictions_21h = self._predict_double(processed_data, predictor)
                processed_data['Y_Retorno_Pred_16h'] = predictions_16h
                processed_data['Y_Retorno_Pred_21h'] = predictions_21h
            
            # Guardar archivo
            self._save_predictions(processed_data, mixing_name, output_folder, prediction_type)
            
            self.log(f"✅ {mixing_name} procesado exitosamente")
            return True
            
        except Exception as e:
            self.log(f"❌ Error procesando {mixing_name}: {str(e)}")
            return False
    
    def _predict_simple(self, data, predictor):
        """Predicción simple (solo 21h)"""
        predictions = []
        for _, row in data.iterrows():
            features = row.to_dict()
            features['Horario'] = 21  # Asegurar horario 21h
            pred = predictor.predict(features)
            predictions.append(pred)
        return predictions
    
    def _predict_double(self, data, predictor):
        """Predicción doble (16h y 21h)"""
        predictions_16h = []
        predictions_21h = []
        
        for _, row in data.iterrows():
            features = row.to_dict()
            
            # Predicción 16h
            features_16h = features.copy()
            features_16h['Horario'] = 16
            pred_16h = predictor.predict(features_16h)
            predictions_16h.append(pred_16h)
            
            # Predicción 21h
            features_21h = features.copy()
            features_21h['Horario'] = 21
            pred_21h = predictor.predict(features_21h)
            predictions_21h.append(pred_21h)
        
        return predictions_16h, predictions_21h
    
    def _process_vallejo(self, data, output_folder, prediction_type):
        """Procesar Vallejo (caso especial)"""
        self.log(f"📊 VALLEJO: vallejo_special (R²: 1.0000)")
        
        # Vallejo simplemente copia el valor actual de Y_Retorno
        if prediction_type == "simple":
            data['Y_Retorno_Pred'] = data['Y_Retorno']
        else:
            data['Y_Retorno_Pred_16h'] = data['Y_Retorno']
            data['Y_Retorno_Pred_21h'] = data['Y_Retorno']
        
        self._save_predictions(data, 'vallejo', output_folder, prediction_type)
        return True
    
    def _save_predictions(self, data, mixing_name, output_folder, prediction_type):
        """Guardar predicciones en archivo CSV"""
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
                data[col] = 0
        
        data[columns].to_csv(output_file, index=False, encoding="cp1252")
    
    def join_data(self, folder, prediction_type, monitoring_month):
        """Unir todos los archivos de predicciones"""
        self.log("🔗 Uniendo archivos de predicciones...")
        
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        
        if not csv_files:
            self.log("❌ No se encontraron archivos CSV para unir")
            return
        
        all_data = []
        
        for file in csv_files:
            try:
                df = pd.read_csv(file, encoding="cp1252")
                all_data.append(df)
                self.log(f"✅ Procesado: {os.path.basename(file)}")
            except Exception as e:
                self.log(f"❌ Error procesando {file}: {str(e)}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            output_file = os.path.join(folder, f"Predicciones {prediction_type.capitalize()} {monitoring_month}.csv")
            combined_data.to_csv(output_file, index=False, encoding="cp1252")
            self.log(f"✅ Archivo unificado creado: {output_file}")
