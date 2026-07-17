import numpy as np
import os
import pickle
import pandas as pd
 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
 
# --- Incluir el wrapper para Prophet ---
from prophet import Prophet
 
class ProphetWrapper:
    """
    Wrapper para usar Prophet como modelo de predicción con una interfaz estándar.
    Se espera que el input X sea un array o Serie con fechas (por ejemplo, de la columna "Mes Monitoreo Soporte").
    """
    def __init__(self):
        self.model = Prophet()
        self.fitted = False
 
    def fit(self, X, y, sample_weight=None):
        # Prophet requiere un DataFrame con columnas 'ds' (fecha) y 'y' (target)
        df = pd.DataFrame({'ds': pd.to_datetime(X.squeeze()), 'y': y})
        self.model.fit(df)
        self.fitted = True
        return self
 
    def predict(self, X):
        # Se espera que X sea un array o Serie de fechas
        df_future = pd.DataFrame({'ds': pd.to_datetime(X.squeeze())})
        forecast = self.model.predict(df_future)
        return forecast['yhat'].values
 
    def get_params(self, deep=True):
        return {"model": self.model}
 
    def set_params(self, **params):
        if "model" in params:
            self.model = params["model"]
        return self
 
# --- Clase BaseModel con train_test actualizado para incluir Prophet ---
class BaseModel():
    """Esta clase contiene funciones comunes sin alterar la interfaz de salida."""
   
    def __init__(self):
        # Define aquí tus predictores para los modelos que usan variables numéricas
        # Por ejemplo: self.predictors = ['FactA', 'Monitoreo', 'Facturación A', 'Retornado A']
        # (Asegúrate de adaptarlo a tu caso)
        self.predictors = ['FactA', 'Monitoreo', 'Facturación A', 'Retornado A']
 
    def monitoring_times_inv(self, data):
        data["Monitoreo_Facturacion"] = data["Monitoreo"] * data["FactA"]
        return data
 
    def previous_invoicing(self, data):
        for i in range(1, data.shape[0]):
            if abs(data.loc[i, "Monitoreo"] - data.loc[i-1, "Monitoreo"]) > 2:
                data.loc[i, "Fact Anterior"] = 0
            else:
                data.loc[i, "Fact Anterior"] = data.loc[i-1, "FactA"]
        data.loc[0, "Fact Anterior"] = 0
        return data
 
    def previous_Y(self, data):
        for i in range(1, data.shape[0]):
            if abs(data.loc[i, "Monitoreo"] - data.loc[i-1, "Monitoreo"]) > 2:
                data.loc[i, "Retornado Anterior"] = 0
            else:
                data.loc[i, "Retornado Anterior"] = data.loc[i-1, "Y_Retorno"]
        data.loc[0, "Retornado Anterior"] = 0
        return data
 
    def save_model(self, model, data):
        # Usar carpeta del mes actual con nombres originales
        try:
            models_path = GLOBAL_MODELS_PATH
            current_month = GLOBAL_CURRENT_MONTH
        except NameError:
            # Fallback al comportamiento original si no están definidas las variables globales
            monitoring = data["Mes Monitoreo"].iloc[-1]
            models_path = f'Models {monitoring}'
            current_month = monitoring
        
        smc = data["Mixing Nombre"].iloc[-1].lower()
        # Usar nombre original sin mes
        model_filename = f'{smc}_transitos_mse.pickle'
        model_path = os.path.join(models_path, model_filename)
        
        # La carpeta ya debería existir (creada en main.py), pero verificamos por seguridad
        if not os.path.exists(models_path):
            os.makedirs(models_path)
            
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modelo guardado en: {os.path.abspath(model_path)}")
 
    def train_test(self, mixing_data):
        """
        Entrena 4 modelos (GradientBoosting, XGB, SVR y Prophet), selecciona el mejor
        según R², MSE y MAE, y reporta métricas.
        """
        # ---------------------------------------------------------------------
        # 1) Conversión de 'Mes Monitoreo' a formato datetime usando mapeos
        # ---------------------------------------------------------------------
        Mes_Map = {
            "ene": "01",
            "feb": "02",
            "mar": "03",
            "abr": "04",
            "may": "05",
            "jun": "06",
            "jul": "07",
            "ago": "08",
            "sep": "09",
            "oct": "10",
            "nov": "11",
            "dic": "12"
        }
 
        Year_Map = {
            "20": "2020",
            "21": "2021",
            "22": "2022",
            "23": "2023",
            "24": "2024",
            "25": "2025",
            "26": "2026",
            "27": "2027"
        }
 
        def convert_to_datetime(mes_monitoreo):
            mes_abbr, year_abbr = mes_monitoreo.split('-')
            month = Mes_Map[mes_abbr]
            year = Year_Map[year_abbr]
            return pd.to_datetime(f"{year}-{month}-01")
 
        # Se crea la columna de soporte con formato datetime
        mixing_data['Mes Monitoreo Soporte'] = mixing_data['Mes Monitoreo'].apply(convert_to_datetime)
 
        # Determinar el último 'Mes Monitoreo Soporte'
        max_mes_soporte = mixing_data["Mes Monitoreo Soporte"].max()
 
        # ---------------------------------------------------------------------
        # 2) Creación del array de pesos en memoria
        #    - Peso 1.0 para registros de los últimos 12 meses
        #    - Peso 0.1 para data anterior
        # ---------------------------------------------------------------------
        w_array = np.where(
            mixing_data["Mes Monitoreo Soporte"] >= max_mes_soporte - pd.DateOffset(months=12),
            1.0,
            0.1
        )
 
        # ---------------------------------------------------------------------
        # 3) Separa X e Y
        # ---------------------------------------------------------------------
        # Para los modelos numéricos, se usan las columnas definidas en self.predictors.
        # A Prophet se le usará la columna "Mes Monitoreo Soporte" (como input de fecha).
        X = mixing_data[self.predictors]
        y = mixing_data["Y_Retorno"]
 
        # Partición train/test (incluyendo los pesos)
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, w_array, test_size=0.3, random_state=0
        )
 
        # ---------------------------------------------------------------------
        # 4) Función interna Best_Model para entrenar y comparar modelos base
        # ---------------------------------------------------------------------
        def Best_Model(X_train, y_train, X_test, y_test, w_train, w_test):
            modelos = {
                'GradientBoostingRegressor': GradientBoostingRegressor(random_state=0),
                'SVR': SVR(C=1.0, epsilon=0.2),
                'Prophet': ProphetWrapper()
            }
 
            resultados = {}
 
            for nombre, modelo in modelos.items():
                if nombre == 'Prophet':
                    # Para Prophet se usa la columna "Mes Monitoreo Soporte"
                    X_train_input = mixing_data.loc[X_train.index, 'Mes Monitoreo Soporte'].values
                    X_test_input = mixing_data.loc[X_test.index, 'Mes Monitoreo Soporte'].values
                else:
                    X_train_input = X_train.values
                    X_test_input = X_test.values
 
                try:
                    # Intenta ajustar pasando sample_weight (los modelos que lo acepten lo usarán)
                    modelo.fit(X_train_input, np.ravel(y_train), sample_weight=w_train)
                except TypeError:
                    modelo.fit(X_train_input, np.ravel(y_train))
 
                y_pred = modelo.predict(X_test_input)
                r2_test = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
 
                resultados[nombre] = {
                    'modelo': modelo,
                    'r2': r2_test,
                    'mse': mse,
                    'mae': mae
                }
 
            # Selección del mejor modelo basándose en R² (y en caso de empate, menor MSE/MAE)
            mejor_modelo = max(
                resultados,
                key=lambda x: (resultados[x]['r2'], -resultados[x]['mse'], -resultados[x]['mae'])
            )
 
            print("Resultados de los modelos:")
            for nombre, metricas in resultados.items():
                print(f"{nombre} -> R2: {metricas['r2']:.4f}, MSE: {metricas['mse']:.4f}, MAE: {metricas['mae']:.4f}")
            print(f"\nMejor modelo: {mejor_modelo} con R2: {resultados[mejor_modelo]['r2']:.4f}")
            return resultados[mejor_modelo]['modelo']
 
        # ---------------------------------------------------------------------
        # 5) Entrenamiento del mejor modelo base con todo el dataset (X, y) y pesos
        # ---------------------------------------------------------------------
        base_model = Best_Model(X_train, y_train, X_test, y_test, w_train, w_test)
        try:
            base_model.fit(X.values, np.ravel(y), sample_weight=w_array)
        except TypeError:
            base_model.fit(X.values, np.ravel(y))
 
        # Evaluación final en todo el conjunto
        y_pred_full = base_model.predict(X.values)
        r2_full = r2_score(y, y_pred_full)
        print(f"Full set R²: {r2_full:.4f}")
 
        return base_model
 
    def write_line(self, data, line):
        # Usar carpeta fija y mes en curso para los logs de resultados
        try:
            models_path = GLOBAL_MODELS_PATH
            current_month = GLOBAL_CURRENT_MONTH
        except NameError:
            # Fallback al comportamiento original si no están definidas las variables globales
            monitoring = data["Mes Monitoreo"].iloc[-1]
            models_path = '.'
            current_month = monitoring
            
        results_filename = f"Resultados_{current_month}.txt"
        results_path = os.path.join(models_path, results_filename)
        
        with open(results_path, 'a', encoding='utf-8') as f:
            f.write("\n" + line)