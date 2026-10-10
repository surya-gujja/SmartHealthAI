
# SmartHealthAI – IoMT Health Risk Prediction

MSc Thesis, Blekinge Institute of Technology (2026)
Authors: Sai Nikhil Sakhetapuram, Surya Gujja
Supervisor: Dr. Huseyin Kusetogullari

LSTM + attention ensemble (with XGBoost, LightGBM, Random Forest) for
classifying patient health status as Normal or At-Risk.

## Files
| File | Purpose |
|---|---|
| `EDA_preprocessing.ipynb` | Exploratory data analysis |
| `baseline models.py` | Baseline models (XGBoost, LightGBM, Random Forest) |
| `v2-2 train.py` | Trains and saves the ensemble models |
| `v2-1predict.py` | Loads saved models and predicts patient risk |
| `final version 3 (1).py` | Full pipeline with plots and SHAP/LIME |
| `healthcare_iot_target_dataset.csv` | Dataset (Kaggle: Healthcare IoT Data) |

## Results
| Model | Accuracy |
|---|---|
| Random Forest (baseline) | 65.0% |
| Proposed ensemble | 97.5% |

## How to run
Install: `pip install numpy pandas scikit-learn torch xgboost lightgbm imbalanced-learn matplotlib seaborn shap lime`

**Note:** Update the file paths inside each script to point to the dataset on your machine before running.
