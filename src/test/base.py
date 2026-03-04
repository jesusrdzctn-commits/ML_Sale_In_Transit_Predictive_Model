import numpy as np
import pandas as pd
import pickle
from robust_model_wrapper import RobustModelWrapper  # IMPORTACIÓN AGREGADA PARA PICKLE

class BaseModel():
    def __init__(self, mixin_name, mixin_data, folder):
        self.columns = [
            "Mixing Nombre",
            "Fecha de Corte",
            "Y_Retorno",
            "Monitoreo",
            "FactA",
            "Horario",
            "Facturación A",
            "Retornado A",
            "Mes Monitoreo"]
        self.data = mixin_data
        self.folder = folder
        self.invoicing = []
        self.mixin = mixin_name.lower()
        self.models = {}
        self.target = "Y_Retorno"
        self.X = []
        self.X_doble = []
        self.X_noon = []
        self.y_pred = []
        self.y_double_pred = []
        self.y_noon_pred = []
 
    def correct_predictions(self, pred):
        for ind in self.data.index:
            if self.data.loc[ind, "Monitoreo"] == 7 and self.data.loc[ind, "Horario"] == 16:
                if ind > 0 and ind-1 in self.data.index:
                    if self.data.loc[ind, pred] < self.data.loc[ind-1, "Y_Retorno"]:
                        self.data.loc[ind, pred] = self.data.loc[ind-1, "Y_Retorno"]
            elif self.data.loc[ind, "Monitoreo"] == 7 and self.data.loc[ind, "Horario"] == 21:
                if ind > 1 and ind-2 in self.data.index:
                    if self.data.loc[ind, pred] < self.data.loc[ind-2, "Y_Retorno"]:
                        self.data.loc[ind, pred] = self.data.loc[ind-2, "Y_Retorno"]
            elif self.data.loc[ind, "Monitoreo"] == 7 and self.data.loc[ind, "Horario"] == 12:
                if ind > 0 and ind-1 in self.data.index:
                    if self.data.loc[ind, pred] < self.data.loc[ind-1, "Y_Retorno"]:
                        self.data.loc[ind, pred] = self.data.loc[ind-1, "Y_Retorno"]
            else:
                pass
 
    def create_row(self, time):
        self.data.reset_index(drop=True, inplace=True)
        row = {
            "Mixing Nombre": [self.data.at[0, "Mixing Nombre"]],
            "Mes Monitoreo": [self.data.at[0, "Mes Monitoreo"]],
            "Fecha de Corte": [self.data.at[0, "Fecha de Corte"]],
            "FactA": [self.data.at[0, "FactA"]],
            "Horario": [time],
            "Monitoreo": [7],
            "Facturación A": [self.data.at[0, "Facturación A"]]
        }
        self.data = pd.concat(
            [self.data, pd.DataFrame(row)],
            ignore_index=True
        )
 
    def load_models(self, ):
        models = ["mse"]
        for model in models:
            with open("new_models/" + self.mixin + "_transitos_" + model + ".pickle", 'rb') as f:
                self.models[model] = pickle.load(f)
 
    def monitoring_times_invoicing(self, ):
        self.data["Monitoreo_Facturacion"] = self.data["Monitoreo"] * self.data["FactA"]
 
    def predict(self, prediction):
        self.load_models()
        if prediction == "first":
            self.y_pred = self.models["mse"].predict(self.X.values)
            self.y_pred = pd.Series(np.where(
                self.y_pred > 100,
                100,
                self.y_pred), name="Y_Retorno_Pred")
            self.data = pd.concat([self.data, self.y_pred], axis=1)
            self.data.to_csv("Auxilio predict.csv")
        else:
            self.y_pred = self.models["mse"].predict(self.X_doble.values)
            self.y_pred = pd.Series(np.where(
                self.y_pred > 100,
                100,
                self.y_pred), name="Y_Retorno_Pred_Doble")
            self.data = pd.concat([self.data, self.y_pred], axis=1)
 
    def previous_invoicing(self, ):
        for i in range(1, self.data.shape[0]):
            if abs(self.data.loc[i, "Monitoreo"] - self.data.loc[i-1, "Monitoreo"]) > 2:
                self.data.loc[i, "Facturacion Anterior"] = 0
            else:
                self.data.loc[i, "Facturacion Anterior"] = self.data.loc[i-1, "FactA"]
        self.data.loc[0, "Facturacion Anterior"] = 0
 
    def previous_y(self, data, variable_name, column):
        for i in range(1, data.shape[0]):
            if abs(data.loc[i, "Monitoreo"] - data.loc[i-1, "Monitoreo"]) > 2:
                data.loc[i, column] = 0
            else:
                data.loc[i, column] = data.loc[i-1, variable_name]
        data.loc[0, column] = 0
 
    def save_predictions(self, pred):
        if pred == "double":
            if (self.data['Y_Retorno'] == 0).any():
                self.data['Y_Retorno_Pred'] = self.data['Y_Retorno_Pred'].apply(lambda x: 0 if pd.notna(x) else x)
                self.data['Y_Retorno_Pred_Doble'] = self.data['Y_Retorno_Pred_Doble'].apply(lambda x: 0 if pd.notna(x) else x)
            self.data.to_csv(
                self.folder + "/predicciones_" + self.mixin + ".csv",
                columns=["Mixing Nombre",
                         "Fecha de Corte",
                         "Y_Retorno",
                         "Monitoreo",
                         "Horario",
                         "Facturación A",
                         "Retornado A",
                         "Mes Monitoreo",
                         "Y_Retorno_Pred",
                         "Y_Retorno_Pred_Doble"],
                index=False, encoding="cp1252")
        elif pred == "mediodia":
            if (self.data['Y_Retorno'] == 0).any():
                self.data['Y_Retorno_Pred'] = self.data['Y_Retorno_Pred'].apply(
                    lambda x: 0 if pd.notna(x) else x
                )
            columns_to_save = [
                "Mixing Nombre",
                "Fecha de Corte",
                "Y_Retorno",
                "Monitoreo",
                "Horario",
                "Facturación A",
                "Retornado A",
                "Mes Monitoreo",
                "Y_Retorno_Pred"
            ]
            existing_columns = [col for col in columns_to_save if col in self.data.columns]
            self.data.to_csv(
                self.folder + "/predicciones_" + self.mixin + ".csv",
                columns=existing_columns,
                index=False,
                encoding="cp1252"
            )
        else:
            if (self.data['Y_Retorno'] == 0).any():
                self.data['Y_Retorno_Pred'] = self.data['Y_Retorno_Pred'].apply(lambda x: 0 if pd.notna(x) else x)
            columns_to_save = [
                "Mixing Nombre",
                "Fecha de Corte",
                "Y_Retorno",
                "Monitoreo",
                "Horario",
                "Facturación A",
                "Retornado A",
                "Mes Monitoreo",
                "Y_Retorno_Pred"
            ]
            existing_columns = [col for col in columns_to_save if col in self.data.columns]
            self.data.to_csv(
                self.folder + "/predicciones_" + self.mixin + ".csv",
                columns=existing_columns,
                index=False,
                encoding="cp1252")
 
    def prepare_data(self, time):
        self.data = self.data[self.columns].copy(deep=False)
        self.create_row(time)
        self.create_variables()
        self.X = self.data[self.predictors]
 
    def prepare_data_double(self, ):
        self.create_row(21)
        self.create_variables_doble()
        self.previous_y(self.data, "Y_Retorno_Pred", "Retornado Anterior Pred")
        self.X_doble = self.data[self.predictors].copy(deep=False)
        self.X_doble.loc[2, "Retornado Anterior"] = self.data.loc[2, "Retornado Anterior Pred"]
 
    def run_model(self, prediction):
        if prediction == "doble":
            self.prepare_data(16)
            self.predict("first")
            self.correct_predictions("Y_Retorno_Pred")
            self.prepare_data_double()
            self.predict("second")
            self.correct_predictions("Y_Retorno_Pred_Doble")
            self.save_predictions("double")
        elif prediction == "mediodia":
            self.prepare_data(12)
            self.predict("first")
            self.correct_predictions("Y_Retorno_Pred")
            self.save_predictions("mediodia")
        else:
            self.prepare_data(21)
            self.predict("first")
            self.correct_predictions("Y_Retorno_Pred")
            self.save_predictions("simple")
 
    def correct_predictions(self, pred):
        for ind in self.data.index:
            if self.data.loc[ind, "Monitoreo"] == 7 and self.data.loc[ind, "Horario"] == 16:
                if ind > 0 and ind-1 in self.data.index:
                    if self.data.loc[ind, pred] < self.data.loc[ind-1, "Y_Retorno"]:
                        self.data.loc[ind, pred] = self.data.loc[ind-1, "Y_Retorno"]
            elif self.data.loc[ind, "Monitoreo"] == 7 and self.data.loc[ind, "Horario"] == 21:
                if ind > 1 and ind-2 in self.data.index:
                    if self.data.loc[ind, pred] < self.data.loc[ind-2, "Y_Retorno"]:
                        self.data.loc[ind, pred] = self.data.loc[ind-2, "Y_Retorno"]
            elif self.data.loc[ind, "Monitoreo"] == 7 and self.data.loc[ind, "Horario"] == 12:
                if ind > 0 and ind-1 in self.data.index:
                    if self.data.loc[ind, pred] < self.data.loc[ind-1, "Y_Retorno"]:
                        self.data.loc[ind, pred] = self.data.loc[ind-1, "Y_Retorno"]
            else:
                pass
 
    def monitoring_times_invoicing(self, ):
        self.data["Monitoreo_Facturacion"] = self.data["Monitoreo"] * self.data["FactA"]
 
    def create_variables(self, ):
        self.previous_invoicing()
        self.previous_y(self.data, "Y_Retorno", "Retornado Anterior")
        self.monitoring_times_invoicing()
 
    def create_variables_doble(self, ):
        self.previous_invoicing()
        self.monitoring_times_invoicing()
 
class BaseMixin(BaseModel):
    def create_variables(self, ):
        self.previous_invoicing()
        self.previous_y(self.data, "Y_Retorno", "Retornado Anterior")
        self.monitoring_times_invoicing()
 
    def create_variables_doble(self, ):
        self.previous_invoicing()
        self.monitoring_times_invoicing()