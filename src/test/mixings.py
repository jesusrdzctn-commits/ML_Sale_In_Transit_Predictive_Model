from base import *


class Azcapotzalco(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"]


class Celaya(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"]


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


class Maizoro(BaseMixin):
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
            "Monitoreo_Facturacion"]


class Monterrey(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"]


class Nexxus(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"]


class NexxusCAP(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"]


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

    def create_variables(self, ):
        self.previous_invoicing()
        self.previous_y(self.data, "Y_Retorno", "Retornado Anterior")
        self.monitoring_times_invoicing()
        self.invoicing_over_monitoring()

    def create_variables_doble(self, ):
        self.previous_invoicing()
        self.monitoring_times_invoicing()
        self.invoicing_over_monitoring()

    def invoicing_over_monitoring(self, ):
        self.data["Facturacion_Monitoreo"] = self.data[
            "FactA"
            ] / self.data[
                "Horario"
                ]

##Adicionar-New
class MixingHenco(BaseMixin):
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
        self.previous_invoicing()
        self.previous_y(self.data, "Y_Retorno", "Retornado Anterior")
        self.monitoring_times_invoicing()
        self.invoicing_over_monitoring()
 
    def create_variables_doble(self):
        self.previous_invoicing()
        self.monitoring_times_invoicing()
        self.invoicing_over_monitoring()
 
    def invoicing_over_monitoring(self):
        self.data["Facturacion_Monitoreo"] = self.data["FactA"] / self.data["Horario"]

class SMCPuebla(BaseMixin):
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
        self.previous_invoicing()
        self.previous_y(self.data, "Y_Retorno", "Retornado Anterior")
        self.monitoring_times_invoicing()
        self.invoicing_over_monitoring()
 
    def create_variables_doble(self):
        self.previous_invoicing()
        self.monitoring_times_invoicing()
        self.invoicing_over_monitoring()
 
    def invoicing_over_monitoring(self):
        self.data["Facturacion_Monitoreo"] = self.data["FactA"] / self.data["Horario"]


class CenterSaltillo(BaseMixin):
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
        self.previous_invoicing()
        self.previous_y(self.data, "Y_Retorno", "Retornado Anterior")
        self.monitoring_times_invoicing()
        self.invoicing_over_monitoring()
 
    def create_variables_doble(self):
        self.previous_invoicing()
        self.monitoring_times_invoicing()
        self.invoicing_over_monitoring()
 
    def invoicing_over_monitoring(self):
        self.data["Facturacion_Monitoreo"] = self.data["FactA"] / self.data["Horario"]
#Adicionar-New


class PlantaObregon(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"]


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
            "Facturacion_Monitoreo",
            "Facturacion_Horario",
            "Horario_Monitoreo"
        ]

    def create_variables(self, ):
        self.previous_invoicing()
        self.previous_y(self.data, "Y_Retorno", "Retornado Anterior")
        self.monitoring_times_invoicing()
        self.invoicing_over_monitoring()
        self.invoicing_over_time()
        self.time_times_monitoring()

    def create_variables_doble(self, ):
        self.previous_invoicing()
        self.monitoring_times_invoicing()
        self.invoicing_over_monitoring()
        self.invoicing_over_time()
        self.time_times_monitoring()

    def invoicing_over_monitoring(self, ):
        self.data["Facturacion_Monitoreo"] = self.data[
            "FactA"
            ] / self.data[
                "Monitoreo"
                ]

    def invoicing_over_time(self, ):
        self.data["Facturacion_Horario"] = self.data[
            "FactA"
            ] / self.data[
                "Horario"
                ]

    def time_times_monitoring(self, ):
        self.data["Horario_Monitoreo"] = self.data[
            "Horario"
            ] * self.data[
                "Monitoreo"]


class PtaMerida(BaseModel):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Retornado Anterior",
            "Monitoreo_Horario",
            "Monitoreo_Facturacion",
            "Facturacion2"
        ]

    def create_variables(self, ):
        self.previous_y(self.data, "Y_Retorno", "Retornado Anterior")
        self.time_times_monitoring()
        self.monitoring_times_invoicing()
        self.square_invoicing()

    def create_variables_doble(self, ):
        self.time_times_monitoring()
        self.monitoring_times_invoicing()
        self.square_invoicing()

    def square_invoicing(self, ):
        self.data["Facturacion2"] = self.data["FactA"] ** 2

    def time_times_monitoring(self, ):
        self.data["Monitoreo_Horario"] = self.data[
            "Monitoreo"
            ] * self.data[
                "Horario"
                ]


class SanMartin(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"]


class Tijuana(BaseMixin):
    def __init__(self, mixin_name, mixin_data, folder):
        super().__init__(mixin_name, mixin_data, folder)
        self.predictors = [
            "Monitoreo",
            "Horario",
            "FactA",
            "Facturacion Anterior",
            "Retornado Anterior",
            "Monitoreo_Facturacion"]


class Vallejo():
    def __init__(self, mixin_name, mixin_data, folder):
        self.mixin = mixin_name
        self.data = mixin_data
        self.folder = folder

    def create_row(self, time, pred):
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
        self.data = pd.concat(
            [self.data, pd.DataFrame(row)],
            ignore_index=True
            )

    def run_model(self, prediction):
        if self.data.shape[0] < 2:
            self.create_row(16, "Y_Retorno_Pred")
            self.create_row(21, "Y_Retorno_Pred_Doble")
            self.data.to_csv(
                self.folder + "/predicciones_" + self.mixin + ".csv",
                columns=[
                    "Mixing Nombre",
                    "Fecha de Corte",
                    "Y_Retorno",
                    "Monitoreo",
                    "Horario",
                    "Facturación A",
                    "Retornado A",
                    "Mes Monitoreo",
                    "Y_Retorno_Pred",
                    "Y_Retorno_Pred_Doble"], index=False, encoding="cp1252")
        else:
            self.create_row(21, "Y_Retorno_Pred")
            self.data.to_csv(
                self.folder + "/predicciones_" + self.mixin + ".csv",
                columns=[
                    "Mixing Nombre",
                    "Fecha de Corte",
                    "Y_Retorno",
                    "Monitoreo",
                    "Horario",
                    "Facturación A",
                    "Retornado A",
                    "Mes Monitoreo",
                    "Y_Retorno_Pred"
                    ], index=False, encoding="cp1252")
