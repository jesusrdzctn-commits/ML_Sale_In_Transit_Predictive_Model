"""
Sistema Avanzado de Predicción que Replica la Lógica del Sistema de Entrenamiento
Con mayor peso a los últimos 12 meses y competencia de modelos
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class AdvancedPredictor:
    """Predictor avanzado que replica la lógica del sistema de entrenamiento"""
    
    def __init__(self, mixing_name, logger=None):
        self.mixing_name = mixing_name.lower()
        self.logger = logger
        self.model = None
        self.best_model_name = None
        self.r2_score = 0.0
        
        # Predictores específicos por SMC (basados en smcs.py)
        self.predictors_map = {
            'azcapotzalco': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'celaya': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'guadalajara': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'merida': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'monterrey': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'nexxus': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'nexxuscap': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'obregon': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'plantaobregon': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'porteo': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'ptamerida': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'sanmartin': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'tijuana': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'vallejo': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'centerobregon': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'smcpuebla': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"],
            'centersaltillo': ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"]
        }
        
        self.predictors = self.predictors_map.get(self.mixing_name, ["Monitoreo", "Horario.1", "FactA", "Fact Anterior", "Retornado Anterior", "Monitoreo_Facturacion"])
    
    def log(self, message):
        if self.logger:
            self.logger(message)
        else:
            print(message)
    
    def create_variables(self, data):
        """Crear variables como en el sistema original"""
        data = data.copy()
        
        # Asegurar tipos numéricos
        for col in ['Monitoreo', 'FactA', 'Y_Retorno']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # Crear variables como en el original
        data = self._previous_invoicing(data)
        data = self._previous_y(data)
        data = self._monitoring_times_invoicing(data)
        
        # Crear Horario.1 (dummy para horario)
        if 'Horario' in data.columns:
            data['Horario.1'] = (data['Horario'] == 21).astype(int)
        else:
            data['Horario.1'] = 0
        
        return data
    
    def _previous_invoicing(self, data):
        """Facturación anterior como en el original"""
        data["Fact Anterior"] = 0
        for i in range(1, data.shape[0]):
            if abs(data.loc[i, "Monitoreo"] - data.loc[i-1, "Monitoreo"]) > 2:
                data.loc[i, "Fact Anterior"] = 0
            else:
                data.loc[i, "Fact Anterior"] = data.loc[i-1, "FactA"]
        return data
    
    def _previous_y(self, data):
        """Retorno anterior como en el original"""
        data["Retornado Anterior"] = 0
        for i in range(1, data.shape[0]):
            if abs(data.loc[i, "Monitoreo"] - data.loc[i-1, "Monitoreo"]) > 2:
                data.loc[i, "Retornado Anterior"] = 0
            else:
                data.loc[i, "Retornado Anterior"] = data.loc[i-1, "Y_Retorno"]
        return data
    
    def _monitoring_times_invoicing(self, data):
        """Crear variable de interacción como en el original"""
        data["Monitoreo_Facturacion"] = data["Monitoreo"] * data["FactA"]
        return data
    
    def create_temporal_weights(self, data):
        """Crear pesos temporales: últimos 12 meses peso 1.0, anteriores peso 0.1"""
        try:
            # Mapeo de meses como en el original
            Mes_Map = {
                "ene": "01", "feb": "02", "mar": "03", "abr": "04", "may": "05", "jun": "06",
                "jul": "07", "ago": "08", "sep": "09", "oct": "10", "nov": "11", "dic": "12"
            }
            
            Year_Map = {
                "20": "2020", "21": "2021", "22": "2022", "23": "2023", 
                "24": "2024", "25": "2025", "26": "2026", "27": "2027"
            }
            
            def convert_to_datetime(mes_monitoreo):
                try:
                    mes_abbr, year_abbr = mes_monitoreo.split('-')
                    month = Mes_Map.get(mes_abbr, "01")
                    year = Year_Map.get(year_abbr, "2024")
                    return pd.to_datetime(f"{year}-{month}-01")
                except:
                    return pd.to_datetime('2024-01-01')
            
            # Crear columna de soporte con formato datetime
            data['Mes Monitoreo Soporte'] = data['Mes Monitoreo'].apply(convert_to_datetime)
            
            # Determinar el último mes
            max_mes_soporte = data["Mes Monitoreo Soporte"].max()
            
            # Crear array de pesos: últimos 12 meses peso 1.0, anteriores peso 0.1
            w_array = np.where(
                data["Mes Monitoreo Soporte"] >= max_mes_soporte - pd.DateOffset(months=12),
                1.0,
                0.1
            )
            
            return w_array
            
        except Exception as e:
            self.log(f"⚠️ Error creando pesos temporales para {self.mixing_name}: {str(e)}")
            # Retornar pesos uniformes como fallback
            return np.ones(len(data))
    
    def train_models(self, data):
        """Entrenar múltiples modelos y seleccionar el mejor"""
        try:
            if len(data) < 10:
                self.log(f"⚠️ Datos insuficientes para {self.mixing_name}: {len(data)} registros")
                return False
            
            # Crear variables
            data = self.create_variables(data)
            self.log(f"🔧 Variables creadas para {self.mixing_name}: {len(data)} registros")
            
            # Crear pesos temporales
            w_array = self.create_temporal_weights(data)
            self.log(f"⚖️ Pesos temporales creados para {self.mixing_name}")
            
            # Filtrar datos según el SMC (como en el original)
            data = self._filter_data_by_smc(data)
            
            if len(data) < 5:
                self.log(f"⚠️ Datos insuficientes después del filtrado para {self.mixing_name}: {len(data)} registros")
                return False
            
            # Preparar X e Y
            available_predictors = [p for p in self.predictors if p in data.columns]
            if not available_predictors:
                self.log(f"❌ No hay predictores disponibles para {self.mixing_name}")
                self.log(f"   Predictores esperados: {self.predictors}")
                self.log(f"   Columnas disponibles: {list(data.columns)}")
                return False
            
            X = data[available_predictors]
            y = data["Y_Retorno"]
            
            # Verificar que no haya valores NaN
            if X.isnull().any().any():
                self.log(f"⚠️ Valores NaN encontrados en X para {self.mixing_name}, llenando con 0")
                X = X.fillna(0)
            
            if y.isnull().any():
                self.log(f"⚠️ Valores NaN encontrados en y para {self.mixing_name}, llenando con 0")
                y = y.fillna(0)
            
            # Partición train/test
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, w_array, test_size=0.3, random_state=0
            )
            
            self.log(f"📊 Datos preparados para {self.mixing_name}: X_train={X_train.shape}, y_train={y_train.shape}")
            
        except Exception as e:
            self.log(f"❌ Error en preparación de datos para {self.mixing_name}: {str(e)}")
            return False
        
        # Definir modelos para competir (configuraciones más robustas)
        modelos = {
            'GradientBoosting': GradientBoostingRegressor(random_state=0, n_estimators=50, max_depth=3),
            'RandomForest': RandomForestRegressor(random_state=0, n_estimators=50, max_depth=5),
            'Ridge': Ridge(alpha=1.0),
            'SVR': SVR(C=1.0, epsilon=0.1, kernel='rbf')
        }
        
        resultados = {}
        
        for nombre, modelo in modelos.items():
            try:
                # Entrenar modelo
                if hasattr(modelo, 'fit') and 'sample_weight' in modelo.fit.__code__.co_varnames:
                    modelo.fit(X_train, y_train, sample_weight=w_train)
                else:
                    modelo.fit(X_train, y_train)
                
                # Predecir
                y_pred = modelo.predict(X_test)
                
                # Verificar que las predicciones sean válidas
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    self.log(f"⚠️ Predicciones inválidas para {nombre} en {self.mixing_name}")
                    continue
                
                # Calcular métricas
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Verificar que las métricas sean válidas
                if np.isnan(r2) or np.isinf(r2):
                    self.log(f"⚠️ Métricas inválidas para {nombre} en {self.mixing_name}")
                    continue
                
                resultados[nombre] = {
                    'modelo': modelo,
                    'r2': r2,
                    'mse': mse,
                    'mae': mae
                }
                
                self.log(f"✅ {nombre} entrenado para {self.mixing_name}: R²={r2:.4f}")
                
            except Exception as e:
                self.log(f"❌ Error entrenando {nombre} para {self.mixing_name}: {str(e)}")
                self.log(f"   Tipo de error: {type(e).__name__}")
                continue
        
        if not resultados:
            self.log(f"❌ No se pudo entrenar ningún modelo para {self.mixing_name}")
            return False
        
        # Seleccionar el mejor modelo basándose en R²
        mejor_modelo = max(
            resultados,
            key=lambda x: (resultados[x]['r2'], -resultados[x]['mse'], -resultados[x]['mae'])
        )
        
        self.model = resultados[mejor_modelo]['modelo']
        self.best_model_name = mejor_modelo
        self.r2_score = resultados[mejor_modelo]['r2']
        
        # Entrenar el mejor modelo con todo el dataset
        try:
            if hasattr(self.model, 'fit') and 'sample_weight' in self.model.fit.__code__.co_varnames:
                self.model.fit(X, y, sample_weight=w_array)
            else:
                self.model.fit(X, y)
            
            # Verificar que el modelo final funcione
            test_pred = self.model.predict(X.iloc[[0]])
            if np.any(np.isnan(test_pred)) or np.any(np.isinf(test_pred)):
                self.log(f"⚠️ Modelo final produce predicciones inválidas para {self.mixing_name}")
                return False
                
        except Exception as e:
            self.log(f"❌ Error en entrenamiento final para {self.mixing_name}: {str(e)}")
            return False
        
        # Log de resultados
        self.log(f"📊 {self.mixing_name.upper()}: {mejor_modelo} (R²: {self.r2_score:.4f})")
        for nombre, metricas in resultados.items():
            self.log(f"   - {nombre}: R²={metricas['r2']:.4f}, MSE={metricas['mse']:.4f}")
        
        return True
    
    def _filter_data_by_smc(self, data):
        """Filtrar datos según reglas específicas del SMC"""
        original_len = len(data)
        
        if self.mixing_name == 'azcapotzalco':
            data = data[data["Y_Retorno"] >= -50]
        elif self.mixing_name == 'celaya':
            data = data[data["Y_Retorno"] <= 100]
        elif self.mixing_name == 'vallejo':
            # Vallejo no usa modelos predictivos
            return data
        
        filtered_len = len(data)
        if filtered_len < original_len:
            self.log(f"🔍 Filtrado {self.mixing_name}: {original_len} → {filtered_len} registros")
        
        return data
    
    def predict_simple(self, data):
        """Predicción simple para horario 21h"""
        if self.mixing_name == 'vallejo':
            return data['Y_Retorno'].iloc[0] if len(data) > 0 else 0
        
        if self.model is None:
            self.log(f"⚠️ Modelo no entrenado para {self.mixing_name}, usando valor por defecto")
            return 50.0
        
        try:
            # Crear variables para predicción
            pred_data = self.create_variables(data)
            
            # Preparar predictores
            available_predictors = [p for p in self.predictors if p in pred_data.columns]
            if not available_predictors:
                return 50.0
            
            X_pred = pred_data[available_predictors].iloc[[0]]  # Solo el primer registro
            
            # Predecir
            prediction = self.model.predict(X_pred)[0]
            
            # Asegurar rango válido
            return max(0, min(100, prediction))
            
        except Exception as e:
            self.log(f"❌ Error en predicción simple para {self.mixing_name}: {str(e)}")
            return 50.0
    
    def predict_double(self, data):
        """Predicción doble: 16h y 21h"""
        if self.mixing_name == 'vallejo':
            base_value = data['Y_Retorno'].iloc[0] if len(data) > 0 else 0
            return base_value, base_value
        
        base_prediction = self.predict_simple(data)
        
        # Para predicción doble, ajustar según el horario
        # 16h: predicción base
        pred_16h = base_prediction
        
        # 21h: predicción base con ajuste (como en el sistema original)
        pred_21h = base_prediction * 1.08  # 8% más para 21h
        
        return pred_16h, pred_21h
