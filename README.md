# Sale-in-Transit Return Predictive Model — PMF Controllership

## Overview

This project predicts the **return percentage of sale in transit** (`Y_Retorno`) for PepsiCo Mexico distribution centers ("mixings") on the 1st of each month. Controllership uses the prediction to recognize revenue before the actual returns are fully reported.

The codebase is split into **two independent pipelines** that live in separate folders (which is why some file names, such as `main.py` and `base.py`, appear twice):

| Pipeline | Files | Cadence |
|---|---|---|
| **Training** | `main.py`, `setup.py`, `data_cleaning.py`, `modeling.py`, `smcs.py`, `base.py` (training version) | Every 3 months |
| **Prediction** | `main.py`, `preprocessing.py`, `mixings.py`, `base.py` (prediction version), `robust_model_wrapper.py` | Every month except December |

One model is trained **per mixing center**; there is no single global model. Each center gets its own class, its own feature set, its own outlier rules, and its own serialized model file.

## Data

Input is a CSV export (Windows-1252 encoding). Both pipelines pick the **most recently modified file** in their input folder (`training-data/` for training, `input-data/` for prediction) rather than a file selected by name. Relevant columns (`setup.py: info_cols`):

- `Mixing Nombre` — distribution center; normalized by removing spaces and replacing "ó" with "o" so values match Python class names (e.g. `Obregón` → `Obregon`).
- `Mes Monitoreo` — monitoring month label in `mmm-aa` format (e.g. `ago-26`).
- `Fecha de Corte` — cutoff date, parsed day-first.
- `Monitoreo` — monitoring sequence number within the cycle (prediction is generated at `Monitoreo == 7`).
- `Horario` / `Horario.1` — hour of the intraday cut (12, 16, 21).
- `FactA`, `Facturación A`, `Retornado A` — invoicing and returned amounts.
- `Y_Retorno` — **target**: return percentage. Missing values are filled with 0 (missing is treated as "no return reported").
- `validacion` — data-quality flag from the business.

## Training pipeline

Run from the training folder after placing the latest extract in `training-data/`:

```
python main.py
```

### 1. Cleaning (`data_cleaning.py`)

`get_clean_data` reads the newest CSV, keeps only `info_cols`, drops the `Maizoro` and `Vallejo` centers (excluded from ML training), removes rows whose `validacion` is `"Fuera"`/`"fuera"`, normalizes center names, fills `Y_Retorno` NaNs with 0, and casts dtypes (`Fecha de Corte` to datetime day-first, `Monitoreo` to int).

### 2. Per-center orchestration (`main.py`, `modeling.py`)

`main.py` creates a versioned output folder `ModelosEntrenados/<mmm-aa>/`, writes an audit log (`log_entrenamiento_<mmm-aa>.txt`) recording dataset size and start/end timestamps per center, then loops over each unique `Mixing Nombre`, instantiating the matching class from `smcs.py` via `globals()[mixing]()` — the cleaned center name must therefore exactly match a class name. `modeling.py` sorts each center's data by `Fecha de Corte` and `Horario.1`, builds features, trains, and saves the model.

### 3. Feature engineering (`smcs.py`, training `base.py`)

Shared engineered variables:

- `Fact Anterior` — previous row's `FactA` (lag 1), reset to 0 when the jump in `Monitoreo` exceeds 2 (i.e., a new monitoring cycle begins, so the lag would cross month boundaries).
- `Retornado Anterior` — previous row's `Y_Retorno`, with the same cycle-reset rule.
- `Monitoreo_Facturacion` — interaction `Monitoreo * FactA`.

Center-specific extras: `Facturacion_Horario = FactA / Horario.1` (Obregon, Porteo), `Facturacion_Monitoreo` (defined as `FactA / Horario.1` for CenterSaltillo, SMCPuebla, MixingHenco, but `FactA / Monitoreo` for Porteo — note the naming inconsistency), `Horario_Monitoreo = Horario.1 * Monitoreo` (Porteo), and `Facturacion2 = FactA**2` plus `Monitoreo_Horario` (PtaMerida).

### 4. Outlier removal (per center, in `create_vars`)

Outliers are filtered on the **target** with hard business thresholds, tuned per center:

| Rule | Centers |
|---|---|
| Drop `Y_Retorno < -50` | Azcapotzalco, Merida |
| Drop `Y_Retorno > 100` | Celaya, Monterrey, Obregon, PtaMerida |
| Drop `Y_Retorno < -100` | Nexxus |
| Drop both `< -100` and `> 100` | Porteo |
| No filter | Guadalajara, NexxusCAP, SanMartin, Tijuana, PlantaObregon, CenterSaltillo, SMCPuebla, MixingHenco |

Values above 100% are physically impossible (more returned than sold in transit); moderately negative values can occur through accounting adjustments and are kept where the center's history shows they are legitimate.

### 5. Model training and selection (training `base.py: train_test`)

For each center:

1. `Mes Monitoreo` is converted to a datetime support column via month/year maps (maps currently cover years 2020–2027).
2. **Recency weights** are built: samples from the last 12 months get weight `1.0`; older samples get `0.1`. Old data still informs the fit but recent behavior dominates.
3. A 70/30 `train_test_split` (fixed `random_state=0`) partitions X, y, and weights.
4. Three candidate models are trained and compared on the test split: `GradientBoostingRegressor(random_state=0)`, `SVR(C=1.0, epsilon=0.2)`, and a `ProphetWrapper` (Prophet fed only the monitoring-month date). `sample_weight` is passed when the estimator accepts it (a `TypeError` fallback fits without weights; the Prophet wrapper accepts the argument but ignores it). *Note: the docstring mentions four models and `XGBRegressor` is imported, but XGBoost is not currently in the candidate dictionary.*
5. The winner is selected by highest test **R²**, breaking ties by lower MSE, then lower MAE.
6. The winning model is **refit on the full dataset** with the recency weights, and the full-set R² printed for reference is therefore in-sample.
7. The model is pickled as `<center>_transitos_mse.pickle` inside `ModelosEntrenados/<mmm-aa>/`.

**Deployment note:** the prediction pipeline loads pickles from a `new_models/` folder. Promoting the freshly trained pickles from `ModelosEntrenados/<mmm-aa>/` into `new_models/` is a manual step.

## Prediction pipeline

Run from the prediction folder after placing the latest extract in `input-data/`:

```
python main.py <mmm-aa> <mode>
```

where `<mmm-aa>` is the monitoring month (e.g. `ago-26`) and `<mode>` is `simple` (21:00 cut only), `doble` (16:00 and 21:00 cuts), or `mediodia` (12:00 cut).

### Flow

1. **Cleaning (`preprocessing.py: clean_data`)** — filters the newest input CSV to the requested `Mes Monitoreo`, keeps only rows with `validacion == "pasa"` (after normalizing "Pasa"), keeps only `Monitoreo == 7` (the final monitoring, when the prediction is due), normalizes center names, fills target NaNs with 0, and writes the working set to `input_data.csv`. Verbose diagnostics are printed at every step for traceability.
2. **Per-center prediction (`mixings.py`, prediction `base.py`)** — each center class appends a **synthetic row** for the cut to be predicted (`Horario` 12/16/21, `Monitoreo = 7`), rebuilds the same engineered features used in training, loads its pickle from `new_models/`, and predicts.
3. **Post-processing guards** (prediction `base.py`):
   - Predictions above 100 are **clipped to 100** (a return percentage cannot exceed 100).
   - `correct_predictions` enforces **monotonicity**: the predicted return at a cut cannot be lower than the last actually observed `Y_Retorno`, because returns only accumulate during the day.
   - In `doble` mode, the 16:00 prediction is fed as `Retornado Anterior` into the 21:00 row (chained prediction).
   - **Zero rule:** if any observed `Y_Retorno` for the center that month equals 0, all predictions for the center are forced to 0.
4. **Special case — Vallejo:** excluded from ML; its class simply carries the currently observed `Y_Retorno` forward as the prediction (naive model).
5. **Consolidation (`preprocessing.py: join_data`)** — per-center CSVs are concatenated, `Horario == 10` rows and raw actuals columns dropped, and the final deliverable written as `Predicciones Simples|Dobles|Mediodia <mmm-aa>.csv`.

`robust_model_wrapper.py` provides `RobustModelWrapper`, which stacks a `HuberRegressor` trained on the base model's residuals (final prediction = base + residual correction, with a fallback to base-only if the correction fails) and a version-tolerant `from_pickle` loader. It is imported in the prediction `main.py` so that pickles containing wrapped models can be deserialized.

## Retraining cadence — why every 3 months?

Return behavior drifts: routes, product mix, client portfolios, and plant logistics change over time (**concept drift**), so a model frozen for a year would slowly go stale. Retraining quarterly refreshes the models often enough to track drift, while avoiding the operational cost and month-to-month instability of retraining on every run. The 12-month **recency weighting** complements this: even between retrains, each fit is dominated by the most recent year of behavior. Predictions are generated monthly except **December**, when year-end closing calendars distort the process. Each training run is versioned (`ModelosEntrenados/<mmm-aa>/` plus an audit log), so any month's predictions can be traced back to the exact model vintage that produced them.

## Getting started

Recommended Python 3.9.x. Install dependencies:

```
pip install -r requirements_stable.txt
```

Key dependencies: pandas, numpy, scikit-learn, prophet/cmdstanpy (versions kept PyInstaller-compatible).

## Known caveats (audit notes)

1. `train_test`'s docstring promises four models (including XGB) but only three are trained.
2. Model selection relies on a single fixed 70/30 split (no cross-validation), and the reported full-set R² after refit is in-sample.
3. Input file selection is by file modification time, not by validated file name.
4. The training filter keeps `validacion` not in {"Fuera","fuera"} while prediction keeps only `"pasa"` — subtly different populations.
5. `prepare_data_double` assigns `X_doble.loc[2, "Retornado Anterior"]` by hard-coded positional index.
6. Trained pickles must be manually promoted to `new_models/`; there is no automated hand-off.
7. `Facturacion_Monitoreo` means `FactA/Horario` in three centers but `FactA/Monitoreo` in Porteo.
8. The `Maizoro` center has a prediction class but is excluded from training data.