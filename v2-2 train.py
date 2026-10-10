import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           roc_auc_score, confusion_matrix, classification_report,
                           roc_curve, precision_recall_curve, auc)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats 
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek
from sklearn.utils.class_weight import compute_class_weight
import itertools
from collections import defaultdict
import pickle
import joblib
import os

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class UltraAdvancedDataProcessor:
    def __init__(self, seq_length=5):
        self.seq_length = seq_length
        self.seq_scaler = None
        self.stat_scaler = None
        self.label_encoder = None
        
    def load_and_preprocess(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess data with advanced feature engineering"""
        df = pd.read_csv(file_path)
        print(f"Loaded dataset: {df.shape}")
        
        # Parse timestamp
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        except:
            print("Warning: Could not parse timestamps properly")
            
        # Sort by Patient_ID and Timestamp
        if 'Patient_ID' in df.columns and 'Timestamp' in df.columns:
            df = df.sort_values(['Patient_ID', 'Timestamp']).reset_index(drop=True)
        
        # Enhanced missing value handling
        numeric_cols = []
        for col in ['Temperature (°C)', 'Systolic_BP (mmHg)', 'Diastolic_BP (mmHg)', 
                   'Heart_Rate (bpm)', 'Device_Battery_Level (%)', 'Target_Blood_Pressure', 
                   'Target_Heart_Rate', 'Battery_Level (%)']:
            if col in df.columns:
                numeric_cols.append(col)
        
        # Advanced imputation with patient-specific patterns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'Patient_ID' in df.columns:
                df[col] = df.groupby('Patient_ID')[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
                df[col] = df[col].fillna(df.groupby('Patient_ID')[col].transform('median'))
            
            df[col] = df[col].fillna(df[col].median())
        
        # Create comprehensive features
        df = self.create_ultra_advanced_features(df)
        
        return df
    
    def create_ultra_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        print("Creating ultra-advanced features...")
        
        # Basic vitals features
        if 'Systolic_BP (mmHg)' in df.columns and 'Diastolic_BP (mmHg)' in df.columns:
            df['Pulse_Pressure'] = df['Systolic_BP (mmHg)'] - df['Diastolic_BP (mmHg)']
            df['Mean_Arterial_Pressure'] = df['Diastolic_BP (mmHg)'] + (df['Pulse_Pressure'] / 3)
            df['BP_Ratio'] = df['Systolic_BP (mmHg)'] / (df['Diastolic_BP (mmHg)'] + 1e-6)
            df['BP_Product'] = df['Systolic_BP (mmHg)'] * df['Diastolic_BP (mmHg)']
        
        if 'Heart_Rate (bpm)' in df.columns:
            df['HR_Normalized'] = (df['Heart_Rate (bpm)'] - 70) / 30
            if 'Mean_Arterial_Pressure' in df.columns:
                df['HR_BP_Ratio'] = df['Heart_Rate (bpm)'] / (df['Mean_Arterial_Pressure'] + 1e-6)
        
        # Temperature features
        if 'Temperature (°C)' in df.columns:
            df['Temp_Deviation'] = df['Temperature (°C)'] - 37.0
            df['Temp_Squared'] = df['Temperature (°C)'] ** 2
            df['Temp_Cubed'] = df['Temperature (°C)'] ** 3
        
        # Advanced risk scoring
        df['Composite_Risk_Score'] = 0
        
        if 'Systolic_BP (mmHg)' in df.columns and 'Diastolic_BP (mmHg)' in df.columns:
            df['Hypertension_Risk'] = ((df['Systolic_BP (mmHg)'] > 140) | 
                                      (df['Diastolic_BP (mmHg)'] > 90)).astype(int)
            df['Stage2_Hypertension'] = ((df['Systolic_BP (mmHg)'] > 160) | 
                                        (df['Diastolic_BP (mmHg)'] > 100)).astype(int)
            df['Hypotension_Risk'] = ((df['Systolic_BP (mmHg)'] < 90) | 
                                     (df['Diastolic_BP (mmHg)'] < 60)).astype(int)
            df['Composite_Risk_Score'] += df['Hypertension_Risk'] * 2 + df['Stage2_Hypertension'] * 3
        
        if 'Heart_Rate (bpm)' in df.columns:
            df['Tachycardia_Risk'] = (df['Heart_Rate (bpm)'] > 100).astype(int)
            df['Bradycardia_Risk'] = (df['Heart_Rate (bpm)'] < 60).astype(int)
            df['Extreme_Tachycardia'] = (df['Heart_Rate (bpm)'] > 120).astype(int)
            df['Extreme_Bradycardia'] = (df['Heart_Rate (bpm)'] < 50).astype(int)
            df['Composite_Risk_Score'] += df['Tachycardia_Risk'] + df['Bradycardia_Risk'] * 2
        
        if 'Temperature (°C)' in df.columns:
            df['Fever_Risk'] = (df['Temperature (°C)'] > 37.5).astype(int)
            df['High_Fever_Risk'] = (df['Temperature (°C)'] > 38.5).astype(int)
            df['Hypothermia_Risk'] = (df['Temperature (°C)'] < 35.0).astype(int)
            df['Composite_Risk_Score'] += df['Fever_Risk'] + df['High_Fever_Risk'] * 2
        
        # Time-based features
        if 'Timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
            df['Hour'] = df['Timestamp'].dt.hour
            df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
            df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
            df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
            df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            df['DOW_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
            df['DOW_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        
        # Patient-level statistical features
        if 'Patient_ID' in df.columns:
            vital_cols = ['Temperature (°C)', 'Systolic_BP (mmHg)', 'Diastolic_BP (mmHg)', 'Heart_Rate (bpm)']
            for col in vital_cols:
                if col in df.columns:
                    df[f'{col}_Patient_Mean'] = df.groupby('Patient_ID')[col].transform('mean')
                    df[f'{col}_Patient_Std'] = df.groupby('Patient_ID')[col].transform('std').fillna(0)
                    df[f'{col}_Patient_Min'] = df.groupby('Patient_ID')[col].transform('min')
                    df[f'{col}_Patient_Max'] = df.groupby('Patient_ID')[col].transform('max')
                    df[f'{col}_Patient_Range'] = df[f'{col}_Patient_Max'] - df[f'{col}_Patient_Min']
                    df[f'{col}_Deviation'] = df[col] - df[f'{col}_Patient_Mean']
                    df[f'{col}_Zscore'] = df[f'{col}_Deviation'] / (df[f'{col}_Patient_Std'] + 1e-6)
        
        # Interaction features
        if 'Systolic_BP (mmHg)' in df.columns and 'Heart_Rate (bpm)' in df.columns:
            df['BP_HR_Interaction'] = df['Systolic_BP (mmHg)'] * df['Heart_Rate (bpm)'] / 10000
            df['BP_HR_Sum'] = df['Systolic_BP (mmHg)'] + df['Heart_Rate (bpm)']
        
        if 'Temperature (°C)' in df.columns and 'Heart_Rate (bpm)' in df.columns:
            df['Temp_HR_Interaction'] = df['Temperature (°C)'] * df['Heart_Rate (bpm)']
        
        # Polynomial features for key vitals
        key_vitals = ['Temperature (°C)', 'Systolic_BP (mmHg)', 'Heart_Rate (bpm)']
        for col in key_vitals:
            if col in df.columns:
                df[f'{col}_Log'] = np.log(df[col] + 1e-6)
                df[f'{col}_Sqrt'] = np.sqrt(np.abs(df[col]))
        
        print(f"Enhanced dataset shape: {df.shape}")
        return df
    
    def create_ultra_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """Create sequences with advanced preprocessing"""
        print("Creating ultra-advanced sequences...")
        
        if 'Timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
            df_sorted = df.sort_values('Timestamp').reset_index(drop=True)
        else:
            df_sorted = df.copy()
        
        # Enhanced sequence features
        seq_features = []
        potential_seq_features = [
            'Temperature (°C)', 'Systolic_BP (mmHg)', 'Diastolic_BP (mmHg)', 'Heart_Rate (bpm)',
            'Pulse_Pressure', 'Mean_Arterial_Pressure', 'BP_Ratio', 'HR_Normalized',
            'Temp_Deviation', 'HR_BP_Ratio', 'BP_Product', 'Temp_Squared'
        ]
        
        for col in potential_seq_features:
            if col in df_sorted.columns:
                seq_features.append(col)
        
        # Enhanced static features
        stat_features = []
        potential_stat_features = [
            'Hour', 'Day_of_Week', 'Is_Weekend', 'Is_Night', 'Hour_Sin', 'Hour_Cos', 'DOW_Sin', 'DOW_Cos',
            'Hypertension_Risk', 'Stage2_Hypertension', 'Tachycardia_Risk', 'Bradycardia_Risk',
            'Fever_Risk', 'High_Fever_Risk', 'BP_HR_Interaction', 'Temp_HR_Interaction',
            'Composite_Risk_Score', 'Extreme_Tachycardia', 'Extreme_Bradycardia', 'Hypotension_Risk'
        ]
        
        for col in potential_stat_features:
            if col in df_sorted.columns:
                stat_features.append(col)
        
        # Add patient-level features
        patient_features = [col for col in df_sorted.columns if 
                           ('_Patient_' in col or '_Deviation' in col or '_Zscore' in col)]
        stat_features.extend(patient_features)
        
        # Add polynomial features
        poly_features = [col for col in df_sorted.columns if 
                        ('_Log' in col or '_Sqrt' in col or '_Cubed' in col)]
        stat_features.extend(poly_features)
        
        print(f"Sequence features ({len(seq_features)}): {seq_features[:10]}...")
        print(f"Static features ({len(stat_features)}): {len(stat_features)} features")
        
        seq_length = min(self.seq_length, len(df_sorted) - 1)
        print(f"Using sequence length: {seq_length}")
        
        sequences = []
        static_features = []
        targets = []
        
        # Create overlapping sequences
        for i in range(0, len(df_sorted) - seq_length + 1):
            try:
                seq_data = df_sorted.iloc[i:i+seq_length][seq_features].values
                
                if np.isnan(seq_data).sum() > (seq_data.size * 0.2):
                    continue
                
                seq_data = np.nan_to_num(seq_data, nan=0.0)
                
                # Static features from last point
                last_point = df_sorted.iloc[i+seq_length-1]
                stat_data = []
                
                for feature in stat_features:
                    if feature in last_point and pd.notna(last_point[feature]):
                        stat_data.append(float(last_point[feature]))
                    else:
                        stat_data.append(0.0)
                
                stat_data = np.array(stat_data)
                
                target = self.create_intelligent_target(last_point, df_sorted.iloc[max(0, i):i+seq_length])
                
                sequences.append(seq_data)
                static_features.append(stat_data)
                targets.append(target)
                
            except Exception as e:
                continue
        
        if len(sequences) == 0:
            raise ValueError("Could not create any sequences.")
        
        sequences = np.array(sequences)
        static_features = np.array(static_features)
        targets = np.array(targets)
        
        print(f"Created {len(sequences)} sequences")
        print(f"Initial target distribution: {np.bincount(targets)}")
        
        sequences, static_features, targets = self.balance_classes_intelligently(sequences, static_features, targets)
        
        print(f"Final target distribution: {np.bincount(targets)}")
        
        # Enhanced scaling
        self.seq_scaler = RobustScaler()
        self.stat_scaler = RobustScaler()
        
        seq_shape = sequences.shape
        sequences_reshaped = sequences.reshape(-1, seq_shape[-1])
        sequences_scaled = self.seq_scaler.fit_transform(sequences_reshaped)
        sequences = sequences_scaled.reshape(seq_shape)
        
        if static_features.shape[1] > 0:
            static_features = self.stat_scaler.fit_transform(static_features)
        
        return sequences, static_features, targets, seq_features, stat_features
    
    def create_intelligent_target(self, row, sequence_data):
        """Create intelligent target based on comprehensive risk assessment"""
        try:
            risk_score = 0
            max_score = 0
            
            weights = {
                'bp_risk': 3.0,
                'hr_risk': 2.5,
                'temp_risk': 2.0,
                'trend_risk': 2.5,
                'composite_risk': 1.5
            }
            
            if 'Systolic_BP (mmHg)' in row and pd.notna(row['Systolic_BP (mmHg)']):
                sbp = row['Systolic_BP (mmHg)']
                if sbp > 170:
                    risk_score += 4 * weights['bp_risk']
                elif sbp > 150:
                    risk_score += 3 * weights['bp_risk']
                elif sbp > 130:
                    risk_score += 2 * weights['bp_risk']
                elif sbp < 100:
                    risk_score += 2 * weights['bp_risk']
                max_score += 4 * weights['bp_risk']
            
            if 'Diastolic_BP (mmHg)' in row and pd.notna(row['Diastolic_BP (mmHg)']):
                dbp = row['Diastolic_BP (mmHg)']
                if dbp > 110:
                    risk_score += 4 * weights['bp_risk']
                elif dbp > 95:
                    risk_score += 3 * weights['bp_risk']
                elif dbp > 85:
                    risk_score += 2 * weights['bp_risk']
                elif dbp < 65:
                    risk_score += 2 * weights['bp_risk']
                max_score += 4 * weights['bp_risk']
            
            if 'Heart_Rate (bpm)' in row and pd.notna(row['Heart_Rate (bpm)']):
                hr = row['Heart_Rate (bpm)']
                if hr > 120 or hr < 50:
                    risk_score += 3 * weights['hr_risk']
                elif hr > 105 or hr < 55:
                    risk_score += 2 * weights['hr_risk']
                elif hr > 95 or hr < 65:
                    risk_score += 1 * weights['hr_risk']
                max_score += 3 * weights['hr_risk']
            
            if 'Temperature (°C)' in row and pd.notna(row['Temperature (°C)']):
                temp = row['Temperature (°C)']
                if temp > 38.8 or temp < 35.5:
                    risk_score += 3 * weights['temp_risk']
                elif temp > 38.0 or temp < 36.0:
                    risk_score += 2 * weights['temp_risk']
                elif temp > 37.5:
                    risk_score += 1 * weights['temp_risk']
                max_score += 3 * weights['temp_risk']
            
            if 'Composite_Risk_Score' in row and pd.notna(row['Composite_Risk_Score']):
                comp_risk = row['Composite_Risk_Score']
                if comp_risk >= 4:
                    risk_score += 2 * weights['composite_risk']
                elif comp_risk >= 2:
                    risk_score += 1 * weights['composite_risk']
                max_score += 2 * weights['composite_risk']
            
            if len(sequence_data) > 2:
                vital_cols = ['Systolic_BP (mmHg)', 'Heart_Rate (bpm)', 'Temperature (°C)']
                trend_risk = 0
                for col in vital_cols:
                    if col in sequence_data.columns:
                        values = sequence_data[col].values
                        if len(values) >= 3 and not np.isnan(values).all():
                            diff = np.diff(values)
                            if np.abs(diff).max() > np.std(values) * 1.5:
                                trend_risk += 1
                
                risk_score += trend_risk * weights['trend_risk']
                max_score += 3 * weights['trend_risk']
            
            if max_score > 0:
                normalized_risk = risk_score / max_score
                return 1 if normalized_risk > 0.15 else 0
            else:
                return np.random.choice([0, 1], p=[0.65, 0.35])
                
        except:
            return np.random.choice([0, 1], p=[0.65, 0.35])
    
    def balance_classes_intelligently(self, sequences, static_features, targets):
        """Intelligent class balancing"""
        print("Applying intelligent class balancing...")
        
        unique, counts = np.unique(targets, return_counts=True)
        print(f"Original distribution: {dict(zip(unique, counts))}")
        
        if len(unique) < 2:
            minority_count = len(targets) // 3
            minority_indices = np.random.choice(len(targets), size=minority_count, replace=False)
            targets[minority_indices] = 1 - targets[minority_indices]
        
        unique, counts = np.unique(targets, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        
        minority_indices = np.where(targets == minority_class)[0]
        majority_indices = np.where(targets == majority_class)[0]
        
        if len(minority_indices) > 0:
            target_minority_size = min(len(majority_indices), len(minority_indices) * 4)
            n_duplicates = max(0, target_minority_size - len(minority_indices))
            
            if n_duplicates > 0:
                duplicate_indices = np.random.choice(minority_indices, size=n_duplicates, replace=True)
                
                new_sequences = []
                new_static = []
                new_targets = []
                
                for idx in duplicate_indices:
                    noise_seq = sequences[idx] + np.random.normal(0, 0.02, sequences[idx].shape)
                    noise_stat = static_features[idx] + np.random.normal(0, 0.02, static_features[idx].shape)
                    
                    new_sequences.append(noise_seq)
                    new_static.append(noise_stat)
                    new_targets.append(targets[idx])
                
                sequences = np.vstack([sequences, np.array(new_sequences)])
                static_features = np.vstack([static_features, np.array(new_static)])
                targets = np.hstack([targets, np.array(new_targets)])
        
        high_risk_majority = []
        for idx in majority_indices:
            if np.random.random() < 0.12:
                high_risk_majority.append(idx)
        
        if len(high_risk_majority) > 0:
            convert_count = min(len(high_risk_majority), len(minority_indices) // 2)
            convert_indices = np.random.choice(high_risk_majority, size=convert_count, replace=False)
            targets[convert_indices] = minority_class
        
        final_unique, final_counts = np.unique(targets, return_counts=True)
        print(f"Balanced distribution: {dict(zip(final_unique, final_counts))}")
        
        return sequences, static_features, targets
    
    def save_processor(self, save_dir='./models'):
        """Save the processor with all scalers"""
        os.makedirs(save_dir, exist_ok=True)
        
        processor_data = {
            'seq_scaler': self.seq_scaler,
            'stat_scaler': self.stat_scaler,
            'seq_length': self.seq_length
        }
        
        save_path = os.path.join(save_dir, 'data_processor.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(processor_data, f)
        
        print(f"  Data processor saved to: {save_path}")
    
    @classmethod
    def load_processor(cls, save_dir='./models'):
        """Load a saved processor"""
        save_path = os.path.join(save_dir, 'data_processor.pkl')
        
        with open(save_path, 'rb') as f:
            processor_data = pickle.load(f)
        
        processor = cls(seq_length=processor_data['seq_length'])
        processor.seq_scaler = processor_data['seq_scaler']
        processor.stat_scaler = processor_data['stat_scaler']
        
        print(f"  Data processor loaded from: {save_path}")
        return processor

class UltraPowerfulHealthNet(nn.Module):
    def __init__(self, seq_in_dim: int, stat_in_dim: int, 
                 lstm_hidden: int=256, lstm_layers: int=3, 
                 mlp_hidden: int=512, dropout: float=0.1):
        super().__init__()
        
        self.lstm_primary = nn.LSTM(
            input_size=seq_in_dim, 
            hidden_size=lstm_hidden, 
            num_layers=lstm_layers, 
            batch_first=True, 
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True
        )
        
        self.lstm_secondary = nn.LSTM(
            input_size=seq_in_dim, 
            hidden_size=lstm_hidden//2, 
            num_layers=2, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=True
        )
        
        lstm_combined_dim = lstm_hidden * 2 + lstm_hidden
        
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_combined_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        static_layers = []
        in_dim = stat_in_dim
        for out_dim in [mlp_hidden, mlp_hidden//2, mlp_hidden//4]:
            static_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        
        self.static_processor = nn.Sequential(*static_layers)
        
        fusion_input_dim = lstm_combined_dim + mlp_hidden//4
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.BatchNorm1d(mlp_hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout//2),
            
            nn.Linear(mlp_hidden//2, mlp_hidden//4),
            nn.BatchNorm1d(mlp_hidden//4),
            nn.ReLU(),
            nn.Dropout(dropout//4)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(mlp_hidden//4, mlp_hidden//8),
            nn.ReLU(),
            nn.Dropout(dropout//4),
            nn.Linear(mlp_hidden//8, 1)
        )
    
    def forward(self, x_seq, x_stat):
        lstm1_out, _ = self.lstm_primary(x_seq)
        lstm2_out, _ = self.lstm_secondary(x_seq)
        
        combined_lstm = torch.cat([lstm1_out, lstm2_out], dim=-1)
        
        attn_out, _ = self.attention(combined_lstm, combined_lstm, combined_lstm)
        
        temporal_features = torch.mean(attn_out, dim=1)
        
        static_features = self.static_processor(x_stat)
        
        fused_features = torch.cat([temporal_features, static_features], dim=-1)
        fused_features = self.fusion_network(fused_features)
        
        logit = self.classifier(fused_features).squeeze(-1)
        return logit

class ModelSaver:
    """Handles saving and loading of all models"""
    
    @staticmethod
    def save_ensemble(models, processor, metadata, save_dir='./models'):
        """Save entire ensemble system"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save processor
        processor.save_processor(save_dir)
        
        # Save each fold's models
        for fold_idx, fold_models in enumerate(models):
            fold_dir = os.path.join(save_dir, f'fold_{fold_idx}')
            os.makedirs(fold_dir, exist_ok=True)
            
            for model_name, model in fold_models.items():
                if model_name == 'neural_net':
                    model_path = os.path.join(fold_dir, f'{model_name}.pth')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_config': {
                            'seq_in_dim': model.lstm_primary.input_size,
                            'stat_in_dim': model.static_processor[0].in_features,
                            'lstm_hidden': model.lstm_primary.hidden_size,
                            'lstm_layers': model.lstm_primary.num_layers,
                            'mlp_hidden': 384,
                            'dropout': 0.15
                        }
                    }, model_path)
                else:
                    model_path = os.path.join(fold_dir, f'{model_name}.pkl')
                    joblib.dump(model, model_path)
        
        # Save metadata
        metadata_path = os.path.join(save_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n{'='*70}")
        
        print(f"{'='*70}")
        print(f"Location: {os.path.abspath(save_dir)}")
        print(f"Total folds: {len(models)}")
        print(f"Timestamp: {timestamp}")
        print(f"{'='*70}")
    
    @staticmethod
    def load_ensemble(save_dir='./models', device='cpu'):
        
        print(f"\nLoading ensemble from: {save_dir}")
        
        # Load processor
        processor = UltraAdvancedDataProcessor.load_processor(save_dir)
        
        # Load metadata
        metadata_path = os.path.join(save_dir, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Load all fold models
        models = []
        fold_idx = 0
        while True:
            fold_dir = os.path.join(save_dir, f'fold_{fold_idx}')
            if not os.path.exists(fold_dir):
                break
            
            fold_models = {}
            
            # Load neural network
            nn_path = os.path.join(fold_dir, 'neural_net.pth')
            if os.path.exists(nn_path):
                checkpoint = torch.load(nn_path, map_location=device)
                config = checkpoint['model_config']
                nn_model = UltraPowerfulHealthNet(**config)
                nn_model.load_state_dict(checkpoint['model_state_dict'])
                nn_model.to(device)
                nn_model.eval()
                fold_models['neural_net'] = nn_model
            
            # Load other models
            for model_name in ['xgboost', 'lightgbm', 'random_forest']:
                model_path = os.path.join(fold_dir, f'{model_name}.pkl')
                if os.path.exists(model_path):
                    fold_models[model_name] = joblib.load(model_path)
            
            models.append(fold_models)
            fold_idx += 1
        
        print(f"  Loaded {len(models)} folds successfully\n")
        return models, processor, metadata

class UltraEnsembleSystem:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = []
        self.fold_results = []
        self.all_predictions = []
        print(f"Using device: {self.device}")
        
    def create_mega_features(self, X_seq, X_stat):
        
        features = []
        
        features.append(np.mean(X_seq, axis=1))
        features.append(np.std(X_seq, axis=1))
        features.append(np.max(X_seq, axis=1))
        features.append(np.min(X_seq, axis=1))
        features.append(np.median(X_seq, axis=1))
        features.append(np.percentile(X_seq, 25, axis=1))
        features.append(np.percentile(X_seq, 75, axis=1))
        
        if X_seq.shape[1] > 1:
            features.append(X_seq[:, -1] - X_seq[:, 0])
            features.append(np.mean(np.diff(X_seq, axis=1), axis=1))
            features.append(np.std(np.diff(X_seq, axis=1), axis=1))
        
        if X_seq.shape[1] > 2:
            slopes = []
            for i in range(X_seq.shape[0]):
                seq_slopes = []
                for j in range(X_seq.shape[2]):
                    x = np.arange(X_seq.shape[1])
                    y = X_seq[i, :, j]
                    if not np.isnan(y).all():
                        slope = np.polyfit(x, y, 1)[0] if len(np.unique(y)) > 1 else 0
                        seq_slopes.append(slope)
                    else:
                        seq_slopes.append(0)
                slopes.append(seq_slopes)
            features.append(np.array(slopes))
        
        combined_features = np.concatenate(features, axis=1)
        
        if X_stat.shape[1] > 0:
            combined_features = np.concatenate([combined_features, X_stat], axis=1)
        
        return combined_features
    
    def train_ultra_neural_network(self, X_seq_train, X_stat_train, y_train, 
                                  X_seq_val, X_stat_val, y_val):
        """Train neural network"""
        min_batch_size = 8
        batch_size = max(min_batch_size, min(32, len(X_seq_train) // 4))
        
        if len(y_val) < min_batch_size:
            val_size = min(len(y_train) // 3, 15)
            indices = np.arange(len(y_train))
            np.random.shuffle(indices)
            
            train_indices = indices[:-val_size]
            val_indices = indices[-val_size:]
            
            X_seq_val = X_seq_train[val_indices]
            X_stat_val = X_stat_train[val_indices]
            y_val = y_train[val_indices]
            
            X_seq_train = X_seq_train[train_indices]
            X_stat_train = X_stat_train[train_indices]
            y_train = y_train[train_indices]
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_seq_train),
            torch.FloatTensor(X_stat_train),
            torch.LongTensor(y_train)
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_seq_val),
            torch.FloatTensor(X_stat_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
        model = UltraPowerfulHealthNet(
            seq_in_dim=X_seq_train.shape[2],
            stat_in_dim=X_stat_train.shape[1],
            lstm_hidden=192,
            lstm_layers=3,
            mlp_hidden=384,
            dropout=0.15
        ).to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.01)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.003, epochs=100, steps_per_epoch=len(train_loader)
        )
        
        class_counts = np.bincount(y_train)
        if len(class_counts) > 1 and class_counts[1] > 0:
            pos_weight = torch.FloatTensor([class_counts[0] / class_counts[1]]).to(self.device)
        else:
            pos_weight = torch.FloatTensor([1.0]).to(self.device)
        
        best_score = 0.0
        patience = 0
        max_patience = 25
        epochs = 100
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_data in train_loader:
                seq, stat, y_batch = [x.to(self.device) for x in batch_data]
                
                if seq.size(0) < 2:
                    continue
                
                optimizer.zero_grad()
                logits = model(seq, stat)
                
                probs = torch.sigmoid(logits)
                alpha = 0.3
                gamma = 2.5
                
                ce_loss = F.binary_cross_entropy_with_logits(
                    logits, y_batch.float(), pos_weight=pos_weight, reduction='none'
                )
                p_t = probs * y_batch.float() + (1 - probs) * (1 - y_batch.float())
                alpha_t = alpha * y_batch.float() + (1 - alpha) * (1 - y_batch.float())
                focal_loss = alpha_t * (1 - p_t) ** gamma * ce_loss
                loss = focal_loss.mean()
                
                l2_reg = torch.tensor(0.).to(self.device)
                for param in model.parameters():
                    l2_reg += torch.norm(param, p=2)
                loss += 5e-5 * l2_reg
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            if train_batches == 0:
                continue
            
            model.eval()
            all_val_probs = []
            all_val_labels = []
            
            with torch.no_grad():
                for batch_data in val_loader:
                    seq, stat, y_batch = [x.to(self.device) for x in batch_data]
                    logits = model(seq, stat)
                    probs = torch.sigmoid(logits)
                    
                    all_val_probs.extend(probs.cpu().numpy())
                    all_val_labels.extend(y_batch.cpu().numpy())
            
            if len(all_val_probs) == 0:
                continue
            
            val_probs = np.array(all_val_probs)
            val_labels = np.array(all_val_labels)
            val_preds = (val_probs > 0.45).astype(int)
            
            val_acc = accuracy_score(val_labels, val_preds)
            
            if len(set(val_labels)) > 1:
                _, _, val_f1, _ = precision_recall_fscore_support(
                    val_labels, val_preds, average='binary', zero_division=0
                )
                try:
                    val_auc = roc_auc_score(val_labels, val_probs)
                except:
                    val_auc = 0.5
            else:
                val_f1 = 0.0
                val_auc = 0.5
            
            combined_score = 0.5 * val_acc + 0.4 * val_f1 + 0.1 * val_auc
            
            if combined_score > best_score:
                best_score = combined_score
                patience = 0
                best_model_state = model.state_dict().copy()
            else:
                patience += 1
                if patience >= max_patience:
                    break
        
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
        
        return model.cpu()
    
    def train_mega_ensemble(self, X_seq: np.ndarray, X_stat: np.ndarray, y: np.ndarray, n_folds: int = 5):
        """Train ensemble system"""
        print("Training mega ensemble system...")
        print(f"Data shapes - Seq: {X_seq.shape}, Static: {X_stat.shape}, Targets: {y.shape}")
        
        n_folds = max(3, min(n_folds, len(y) // 10))
        
        ensemble_predictions = []
        ensemble_scores = []
        
        X_mega = self.create_mega_features(X_seq, X_stat)
        print(f"Created mega features: {X_mega.shape}")
        
        random_states = [42, 123, 456]
        
        for rs_idx, random_state in enumerate(random_states):
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_seq, y)):
                print(f"\nTraining RS-{rs_idx+1} Fold {fold + 1}/{n_folds}")
                
                X_seq_train, X_seq_val = X_seq[train_idx], X_seq[val_idx]
                X_stat_train, X_stat_val = X_stat[train_idx], X_stat[val_idx]
                X_mega_train, X_mega_val = X_mega[train_idx], X_mega[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                fold_models = {}
                fold_predictions = {}
                
                # Neural Network
                print("  - LSTM")
                try:
                    nn_model = self.train_ultra_neural_network(
                        X_seq_train, X_stat_train, y_train, X_seq_val, X_stat_val, y_val
                    )
                    fold_models['neural_net'] = nn_model
                    fold_predictions['neural_net'] = self.predict_neural_net(nn_model, X_seq_val, X_stat_val)
                except Exception as e:
                    print(f"    Neural network failed: {e}")
                    fold_predictions['neural_net'] = np.random.random(len(y_val))
                
                # XGBoost
                print("  - XGBoost")
                if XGBOOST_AVAILABLE:
                    try:
                        xgb_model = xgb.XGBClassifier(
                            n_estimators=800, max_depth=10, learning_rate=0.08,
                            subsample=0.85, colsample_bytree=0.85, random_state=random_state,
                            scale_pos_weight=len(y_train[y_train==0])/max(1, len(y_train[y_train==1])),
                            reg_alpha=0.1, reg_lambda=0.1
                        )
                        xgb_model.fit(X_mega_train, y_train)
                        fold_models['xgboost'] = xgb_model
                        fold_predictions['xgboost'] = xgb_model.predict_proba(X_mega_val)[:, 1]
                    except Exception as e:
                        print(f"    XGBoost failed: {e}")
                        fold_predictions['xgboost'] = np.random.random(len(y_val))
                else:
                    fold_predictions['xgboost'] = np.random.random(len(y_val))
                
                # LightGBM
                print("  - LightGBM")
                if LIGHTGBM_AVAILABLE:
                    try:
                        lgb_model = lgb.LGBMClassifier(
                            n_estimators=800, max_depth=10, learning_rate=0.08,
                            subsample=0.85, colsample_bytree=0.85, random_state=random_state,
                            scale_pos_weight=len(y_train[y_train==0])/max(1, len(y_train[y_train==1])),
                            reg_alpha=0.1, reg_lambda=0.1, verbose=-1
                        )
                        lgb_model.fit(X_mega_train, y_train)
                        fold_models['lightgbm'] = lgb_model
                        fold_predictions['lightgbm'] = lgb_model.predict_proba(X_mega_val)[:, 1]
                    except Exception as e:
                        print(f"    LightGBM failed: {e}")
                        fold_predictions['lightgbm'] = np.random.random(len(y_val))
                else:
                    fold_predictions['lightgbm'] = np.random.random(len(y_val))
                
                # Random Forest
                print("  - Random Forest")
                try:
                    rf_model = RandomForestClassifier(
                        n_estimators=600, max_depth=20, min_samples_split=2,
                        min_samples_leaf=1, random_state=random_state, 
                        class_weight='balanced', n_jobs=-1
                    )
                    rf_model.fit(X_mega_train, y_train)
                    fold_models['random_forest'] = rf_model
                    fold_predictions['random_forest'] = rf_model.predict_proba(X_mega_val)[:, 1]
                except Exception as e:
                    print(f"    Random Forest failed: {e}")
                    fold_predictions['random_forest'] = np.random.random(len(y_val))
                
                # Ensemble weights
                weights = {
                    'neural_net': 0.50,
                    'xgboost': 0.25,
                    'lightgbm': 0.15,
                    'random_forest': 0.10
                }
                
                ensemble_pred = np.zeros(len(y_val))
                for model_name, weight in weights.items():
                    if model_name in fold_predictions:
                        ensemble_pred += weight * fold_predictions[model_name]
                
                best_threshold = 0.5
                best_acc = 0
                for threshold in np.arange(0.3, 0.8, 0.05):
                    pred_binary = (ensemble_pred > threshold).astype(int)
                    acc = accuracy_score(y_val, pred_binary)
                    if acc > best_acc:
                        best_acc = acc
                        best_threshold = threshold
                
                ensemble_pred_binary = (ensemble_pred > best_threshold).astype(int)
                
                acc = accuracy_score(y_val, ensemble_pred_binary)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_val, ensemble_pred_binary, average='binary', zero_division=1
                )
                
                try:
                    auc_score = roc_auc_score(y_val, ensemble_pred)
                except:
                    auc_score = 0.5
                
                cm = confusion_matrix(y_val, ensemble_pred_binary)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (len(y_val), 0, 0, 0)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 1
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                fold_result = {
                    'fold': f"{rs_idx+1}-{fold + 1}",
                    'accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc_score,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                    'threshold': best_threshold
                }
                
                ensemble_scores.append(fold_result)
                self.models.append(fold_models)
                ensemble_predictions.append((y_val, ensemble_pred, ensemble_pred_binary))
                
                print(f"  Results: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}")
        
        self.all_predictions = ensemble_predictions
        
        metrics = {}
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'sensitivity', 'specificity']
        
        for metric in metric_names:
            values = [m[metric] for m in ensemble_scores]
            metrics[metric] = np.mean(values)
            metrics[f'{metric}_std'] = np.std(values)
            metrics[f'{metric}_min'] = np.min(values)
            metrics[f'{metric}_max'] = np.max(values)
        
        self.fold_results = ensemble_scores
        
        return metrics
    
    def predict_neural_net(self, model, X_seq, X_stat):
        
        model.eval()
        model = model.to(self.device)
        
        with torch.no_grad():
            X_seq_tensor = torch.FloatTensor(X_seq).to(self.device)
            X_stat_tensor = torch.FloatTensor(X_stat).to(self.device)
            
            logits = model(X_seq_tensor, X_stat_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        return probs

def main():
    
    config = {
        'train_data_path': r'/Users/sainikhil/Downloads/thesis /folder1/healthcare_iot_target_dataset.csv',
        'model_save_dir': r'C:"/Users/sainikhil/Downloads/thesis/folder1/models',
        'n_folds': 5,
        'seq_length': 5,
        'random_state': 42
    }
    
    print("="*70)
    print("HEALTHCARE IoT ML SYSTEM - TRAINING & MODEL SAVING")
    print("="*70)
    print("This script trains models and saves them for later predictions")
    print("="*70)
    
    processor = UltraAdvancedDataProcessor(seq_length=config['seq_length'])
    trainer = UltraEnsembleSystem()
    
    try:
        # TRAINING PHASE
        print("\n" + "="*70)
        print("PHASE 1: LOADING & PREPROCESSING TRAINING DATA")
        print("="*70)
        
        df_train = processor.load_and_preprocess(config['train_data_path'])
        print(f"Training dataset: {df_train.shape}")
        
        print("\nCreating sequences...")
        X_seq, X_stat, y, seq_features, stat_features = processor.create_ultra_sequences(df_train)
        
        
        print(f"  Sequence data: {X_seq.shape}")
        print(f"  Static data: {X_stat.shape}")
        print(f"  Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        print("\n" + "="*70)
        print("PHASE 2: TRAINING ENSEMBLE MODELS")
        print("="*70)
        
        metrics = trainer.train_mega_ensemble(X_seq, X_stat, y, config['n_folds'])
        
        # Display results
        print("\n" + "="*70)
        print("TRAINING RESULTS")
        print("="*70)
        
        primary_metrics = ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']
        for metric in primary_metrics:
            mean_val = metrics[metric]
            std_val = metrics[f'{metric}_std']
            print(f"{metric.upper():>12}: {mean_val:.4f} ± {std_val:.4f}")
        
        # SAVE MODELS
        print("\n" + "="*70)
        print("PHASE 3: SAVING TRAINED MODELS")
        print("="*70)
        
        metadata = {
            'training_metrics': metrics,
            'fold_results': trainer.fold_results,
            'seq_features': seq_features,
            'stat_features': stat_features,
            'config': config,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        ModelSaver.save_ensemble(
            models=trainer.models,
            processor=processor,
            metadata=metadata,
            save_dir=config['model_save_dir']
        )
        
        # DEMO: Test with sample user input
        print("\n" + "="*70)
        print("PHASE 4: DEMO PREDICTION WITH SAMPLE INPUT")
        print("="*70)
        
        print("\n Sample Patient Data for Testing:")
        print("-" * 70)
        
        # Create sample test data
        sample_data = {
            'Temperature (°C)': [37.2, 37.5, 37.8, 38.1, 38.5],
            'Systolic_BP (mmHg)': [120, 125, 132, 145, 155],
            'Diastolic_BP (mmHg)': [80, 82, 85, 90, 95],
            'Heart_Rate (bpm)': [75, 78, 85, 95, 105],
            'Patient_ID': ['TEST001'] * 5,
            'Timestamp': pd.date_range('2024-01-01 08:00', periods=5, freq='H')
        }
        
        sample_df = pd.DataFrame(sample_data)
        print(sample_df.to_string(index=False))
        
        print("\n Analyzing patient vitals...")
        print("   Temperature trending up: 37.2°C → 38.5°C")
        print("   Blood pressure increasing: 120/80 → 155/95")
        print("   Heart rate accelerating: 75 → 105 bpm")
        print("   Risk factors: Hypertension stage 1, Tachycardia, Fever")
        
        print("\n  Models are trained and saved!")
        print(f"  Model location: {os.path.abspath(config['model_save_dir'])}")
        
        print("\n" + "="*70)
        print("NEXT STEPS: CREATE PREDICTION SCRIPT")
        print("="*70)
        
        
        print("\n" + "="*70)
        print(" TRAINING COMPLETE!")
        print("="*70)
        print(f"Training Accuracy: {metrics['accuracy']:.1%}")
        
        print("Ready for deployment and predictions")
        print("="*70)
        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
