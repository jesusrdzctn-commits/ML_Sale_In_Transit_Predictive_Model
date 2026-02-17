import numpy as np
from sklearn.linear_model import HuberRegressor
import pickle
import warnings
 
class RobustModelWrapper:
    """
    Este wrapper combina un modelo base y un modelo de corrección de residuales.
    Primero se ajusta el modelo base; luego se entrena un HuberRegressor sobre los
    residuales (diferencia entre la predicción base y el valor real). En producción,
    la predicción final es la suma de ambas predicciones.
    """
    def __init__(self, base_model, residual_model=None):
        self.base_model = base_model
        if residual_model is None:
            self.residual_model = HuberRegressor()
        else:
            self.residual_model = residual_model
    
    @classmethod
    def from_pickle(cls, pickle_path):
        """
        Carga un modelo pickle con manejo de incompatibilidades de versiones
        """
        try:
            with open(pickle_path, 'rb') as f:
                model = pickle.load(f)
            return cls(model)
        except (AttributeError, ModuleNotFoundError, ImportError) as e:
            print(f"⚠️  Error de compatibilidad al cargar {pickle_path}: {str(e)}")
            # Intentar cargar con supresión de warnings y manejo de errores
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(pickle_path, 'rb') as f:
                        model = pickle.load(f)
                return cls(model)
            except Exception as e2:
                print(f"❌ No se pudo cargar el modelo: {str(e2)}")
                raise e2
    
    def fit(self, X, y, sample_weight=None):
        # Entrena el modelo base
        if sample_weight is not None:
            self.base_model.fit(X, y, sample_weight=sample_weight)
        else:
            self.base_model.fit(X, y)
        # Calcula los residuales
        base_preds = self.base_model.predict(X)
        residuals = y - base_preds
        # Entrena el modelo de residuales (HuberRegressor por robustez)
        self.residual_model.fit(X, residuals)
        return self
 
    def predict(self, X):
        try:
            base_preds = self.base_model.predict(X)
            residual_correction = self.residual_model.predict(X)
            return base_preds + residual_correction
        except Exception as e:
            # Si falla la predicción con residuales, usar solo el modelo base
            print(f"⚠️  Error en predicción con residuales, usando modelo base: {str(e)}")
            return self.base_model.predict(X)
 
    def get_params(self, deep=True):
        return {"base_model": self.base_model, "residual_model": self.residual_model}
 
    def set_params(self, **params):
        if "base_model" in params:
            self.base_model = params["base_model"]
        if "residual_model" in params:
            self.residual_model = params["residual_model"]
        return self