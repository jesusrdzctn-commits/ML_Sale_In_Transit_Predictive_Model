from base import BaseModel


class Azcapotzalco(BaseModel):
    """This class inherits from BaseModel and trains Azcapotzalco Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
            ]

    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        mixing_data.drop(
            mixing_data[mixing_data["Y_Retorno"] < -50].index,
            inplace=True
        )
        return mixing_data


class Celaya(BaseModel):
    """This class inherits from BaseModel and trains Celaya Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
            ]

    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        mixing_data.drop(
            mixing_data[mixing_data["Y_Retorno"] > 100].index,
            inplace=True
        )
        return mixing_data


class Guadalajara(BaseModel):
    """This class inherits from BaseModel and trains Guadalajara Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
            ]

    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        return mixing_data


class Merida(BaseModel):
    """This class inherits from BaseModel and trains Merida Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
            ]

    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        mixing_data.drop(
            mixing_data[mixing_data["Y_Retorno"] < -50].index,
            inplace=True
        )
        return mixing_data


class Monterrey(BaseModel):
    """This class inherits from BaseModel and trains Monterrey Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
            ]

    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        mixing_data.drop(
            mixing_data[mixing_data["Y_Retorno"] > 100].index,
            inplace=True
        )
        return mixing_data


class Nexxus(BaseModel):
    """This class inherits from BaseModel and trains Nexxus Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        mixing_data.drop(
            mixing_data[mixing_data["Y_Retorno"] < -100].index,
            inplace=True
        )
        return mixing_data


class NexxusCAP(BaseModel):
    """This class inherits from BaseModel and trains Nexxus Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        return mixing_data


class Obregon(BaseModel):
    """This class inherits from BaseModel and trains Obregon Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion",
            "Facturacion_Horario"
            ]

    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        mixing_data = self.invoicing_over_time(mixing_data)
        mixing_data.drop(
            mixing_data[mixing_data["Y_Retorno"] > 100].index,
            inplace=True
        )
        return mixing_data

    def invoicing_over_time(self, data):
        data["Facturacion_Horario"] = data["FactA"] / data["Horario.1"]
        return data

#Adicionar-New models for Revenue Recognition
class CenterSaltillo(BaseModel):
    """This class inherits from BaseModel and trains Center Saltillo Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion",
            "Facturacion_Monitoreo"
        ]
 
    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        mixing_data = self.invoicing_over_time(mixing_data)
        return mixing_data
 
    def invoicing_over_time(self, data):
        data["Facturacion_Monitoreo"] = data["FactA"] / data["Horario.1"]
        return data
#Adicionar-New models for Revenue Recognition

#Adicionar-New models for Revenue Recognition
class SMCPuebla(BaseModel):
    """This class inherits from BaseModel and trains PueblaSMC Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion",
            "Facturacion_Monitoreo"
        ]
 
    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        mixing_data = self.invoicing_over_time(mixing_data)
        return mixing_data
 
    def invoicing_over_time(self, data):
        data["Facturacion_Monitoreo"] = data["FactA"] / data["Horario.1"]
        return data
#Adicionar-New models for Revenue Recognition

#Adicionar-New models for Revenue Recognition
class CenterObregon(BaseModel):
    """This class inherits from BaseModel and trains ObregonSMC Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion",
            "Facturacion_Monitoreo"
        ]
 
    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        mixing_data = self.invoicing_over_time(mixing_data)
        return mixing_data
 
    def invoicing_over_time(self, data):
        data["Facturacion_Monitoreo"] = data["FactA"] / data["Horario.1"]
        return data
#Adicionar-New models for Revenue Recognition

class PlantaObregon(BaseModel):
    """This class inherits from BaseModel and trains Planta Obregon Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
        ]

    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        return mixing_data


class Porteo(BaseModel):
    """This class inherits from BaseModel and trains Porteo Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion",
            "Facturacion_Monitoreo",
            "Facturacion_Horario",
            "Horario_Monitoreo"
        ]

    def create_vars(self, mixing_data):
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        mixing_data = self.invoicing_over_mon(mixing_data)
        mixing_data = self.invoicing_over_time(mixing_data)
        mixing_data = self.time_times_monitoring(mixing_data)
        mixing_data.drop(
            mixing_data[mixing_data["Y_Retorno"] < -100].index,
            inplace=True
        )
        mixing_data.drop(
            mixing_data[mixing_data["Y_Retorno"] > 100].index,
            inplace=True
        )
        return mixing_data

    def invoicing_over_mon(self, data):
        data["Facturacion_Monitoreo"] = data["FactA"] / data["Monitoreo"]
        return data

    def invoicing_over_time(self, data):
        data["Facturacion_Horario"] = data["FactA"] / data["Horario.1"]
        return data

    def time_times_monitoring(self, data):
        data["Horario_Monitoreo"] = data["Horario.1"] * data["Monitoreo"]
        return data


class PtaMerida(BaseModel):
    """This class inherits from BaseModel and trains Pta Merida Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Retornado Anterior",
            "Monitoreo_Horario",
            "Monitoreo_Facturacion",
            "Facturacion2"
            ]

    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_time(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        mixing_data = self.squared_inv(mixing_data)
        mixing_data.drop(
            mixing_data[mixing_data["Y_Retorno"] > 100].index,
            inplace=True
        )
        return mixing_data

    def squared_inv(self, data):
        data["Facturacion2"] = data["FactA"] ** 2
        return data

    def monitoring_times_time(self, data):
        data["Monitoreo_Horario"] = data["Monitoreo"] * data["Horario.1"]
        return data


class SanMartin(BaseModel):
    """This class inherits from BaseModel and trains San Martin Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
            ]

    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        return mixing_data


class Tijuana(BaseModel):
    """This class inherits from BaseModel and trains Tijuana Model."""
    def __init__(self, ):
        self.predictors = [
            "Monitoreo",
            "Horario.1",
            "FactA",
            "Fact Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"
            ]

    def create_vars(self, mixing_data):
        """This method creates customized input data"""
        mixing_data = self.previous_invoicing(mixing_data)
        mixing_data = self.previous_Y(mixing_data)
        mixing_data = self.monitoring_times_inv(mixing_data)
        return mixing_data
