"""
Replicación exacta de las clases de mixings del sistema original
"""

import pandas as pd
from original_base import BaseModel, BaseMixin

class Azcapotzalco(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

class Celaya(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

class Guadalajara(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

class Merida(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

class Monterrey(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

class Nexxus(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

class NexxusCAP(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

class Obregon(BaseModel):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion",
            "Facturacion_Monitoreo"
        ]
    
    def create_variables(self):
        """Crear variables específicas para Obregon"""
        self.previous_invoicing()
        self.previous_y(self.data, "Y_Retorno", "Retornado Anterior")
        self.monitoring_times_invoicing()
        # Crear variable específica Facturacion_Monitoreo
        self.data["FactA"] = pd.to_numeric(self.data["FactA"], errors='coerce').fillna(0)
        self.data["Monitoreo"] = pd.to_numeric(self.data["Monitoreo"], errors='coerce').fillna(0)
        self.data["Facturacion_Monitoreo"] = self.data["FactA"] * self.data["Monitoreo"]
    
    def create_variables_doble(self):
        """Crear variables para predicción doble"""
        self.previous_invoicing()
        self.monitoring_times_invoicing()
        # Crear variable específica Facturacion_Monitoreo
        self.data["FactA"] = pd.to_numeric(self.data["FactA"], errors='coerce').fillna(0)
        self.data["Monitoreo"] = pd.to_numeric(self.data["Monitoreo"], errors='coerce').fillna(0)
        self.data["Facturacion_Monitoreo"] = self.data["FactA"] * self.data["Monitoreo"]

class PlantaObregon(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

class Porteo(BaseModel):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion",
            "Monitoreo_Horario"
        ]
    
    def create_variables(self):
        """Crear variables específicas para Porteo"""
        self.previous_invoicing()
        self.previous_y(self.data, "Y_Retorno", "Retornado Anterior")
        self.monitoring_times_invoicing()
        # Crear variable específica Monitoreo_Horario
        self.data["Monitoreo"] = pd.to_numeric(self.data["Monitoreo"], errors='coerce').fillna(0)
        self.data["Horario"] = pd.to_numeric(self.data["Horario"], errors='coerce').fillna(0)
        self.data["Monitoreo_Horario"] = self.data["Monitoreo"] * self.data["Horario"]
    
    def create_variables_doble(self):
        """Crear variables para predicción doble"""
        self.previous_invoicing()
        self.monitoring_times_invoicing()
        # Crear variable específica Monitoreo_Horario
        self.data["Monitoreo"] = pd.to_numeric(self.data["Monitoreo"], errors='coerce').fillna(0)
        self.data["Horario"] = pd.to_numeric(self.data["Horario"], errors='coerce').fillna(0)
        self.data["Monitoreo_Horario"] = self.data["Monitoreo"] * self.data["Horario"]

class PtaMerida(BaseModel):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion",
            "Monitoreo_Horario"
        ]
    
    def create_variables(self):
        """Crear variables específicas para PtaMerida"""
        self.previous_invoicing()
        self.previous_y(self.data, "Y_Retorno", "Retornado Anterior")
        self.monitoring_times_invoicing()
        # Crear variable específica Monitoreo_Horario
        self.data["Monitoreo"] = pd.to_numeric(self.data["Monitoreo"], errors='coerce').fillna(0)
        self.data["Horario"] = pd.to_numeric(self.data["Horario"], errors='coerce').fillna(0)
        self.data["Monitoreo_Horario"] = self.data["Monitoreo"] * self.data["Horario"]
    
    def create_variables_doble(self):
        """Crear variables para predicción doble"""
        self.previous_invoicing()
        self.monitoring_times_invoicing()
        # Crear variable específica Monitoreo_Horario
        self.data["Monitoreo"] = pd.to_numeric(self.data["Monitoreo"], errors='coerce').fillna(0)
        self.data["Horario"] = pd.to_numeric(self.data["Horario"], errors='coerce').fillna(0)
        self.data["Monitoreo_Horario"] = self.data["Monitoreo"] * self.data["Horario"]

class SanMartin(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

class Tijuana(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

class CenterObregon(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

class SMCPuebla(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

class CenterSaltillo(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

class Vallejo:
    """Implementación especial para Vallejo que no usa modelos predictivos"""
    def __init__(self, mixin_name, mixin_data, folder):
        self.mixin = mixin_name.lower()
        self.data = mixin_data.copy()
        self.folder = folder
    
    def create_row(self, time, pred):
        """Crear fila con predicción basada en Y_Retorno actual"""
        self.data.reset_index(drop=True, inplace=True)
        row = {
            "Mixing Nombre": [self.data.at[0, "Mixing Nombre"]],
            "Mes Monitoreo": [self.data.at[0, "Mes Monitoreo"]],
            "Fecha de Corte": [self.data.at[0, "Fecha de Corte"]],
            "FactA": [self.data.at[0, "FactA"]],
            "Horario": [time],
            "Monitoreo": [7],
            "Facturación A": [self.data.at[0, "Facturación A"]],
            pred: [self.data.at[0, "Y_Retorno"]]
        }
        new_row = pd.DataFrame(row)
        self.data = pd.concat([self.data, new_row], ignore_index=True)
    
    def run_model(self, prediction):
        """Ejecutar modelo especial para Vallejo"""
        if self.data.shape[0] < 2:
            # Caso de predicción doble
            self.create_row(16, "Y_Retorno_Pred")
            self.create_row(21, "Y_Retorno_Pred_Doble")
            
            columns_to_save = [
                "Mixing Nombre", "Fecha de Corte", "Y_Retorno", "Monitoreo", "Horario",
                "Facturación A", "Retornado A", "Mes Monitoreo", "Y_Retorno_Pred", "Y_Retorno_Pred_Doble"
            ]
        else:
            # Caso de predicción simple
            self.create_row(21, "Y_Retorno_Pred")
            
            columns_to_save = [
                "Mixing Nombre", "Fecha de Corte", "Y_Retorno", "Monitoreo", "Horario",
                "Facturación A", "Retornado A", "Mes Monitoreo", "Y_Retorno_Pred"
            ]
        
        # Guardar archivo
        output_file = f"{self.folder}/predicciones_{self.mixin}.csv"
        self.data.to_csv(output_file, columns=columns_to_save, index=False, encoding="cp1252")

# Función para obtener la clase del mixing
def get_mixing_class(mixing_name):
    """Obtener la clase correspondiente al mixing"""
    class_mapping = {
        'azcapotzalco': Azcapotzalco,
        'celaya': Celaya,
        'guadalajara': Guadalajara,
        'merida': Merida,
        'monterrey': Monterrey,
        'nexxus': Nexxus,
        'nexxuscap': NexxusCAP,
        'nexxus cap': NexxusCAP,
        'obregon': Obregon,
        'plantaobregon': PlantaObregon,
        'planta obregon': PlantaObregon,
        'porteo': Porteo,
        'ptamerida': PtaMerida,
        'pta merida': PtaMerida,
        'sanmartin': SanMartin,
        'san martin': SanMartin,
        'tijuana': Tijuana,
        'centerobregon': CenterObregon,
        'center obregon': CenterObregon,
        'smcpuebla': SMCPuebla,
        'smc puebla': SMCPuebla,
        'centersaltillo': CenterSaltillo,
        'center saltillo': CenterSaltillo,
        'vallejo': Vallejo
    }
    
    return class_mapping.get(mixing_name.lower())
