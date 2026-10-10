

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')



class UltraPowerfulHealthNet(nn.Module):
    """Neural network architecture - must match training"""
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



class HealthcarePredictionSystem:
    
    
    def __init__(self, model_dir=None, device='cpu'):
              
        if model_dir is None:
            possible_dirs = [
                r'C:\Users\krake\Downloads\nikil sweden\trained_models\models',
                r'C:\Users\krake\Downloads\nikil sweden\models',
                './models',
                '../models',
                './trained_models/models'
            ]
            
            model_dir = None
            for path in possible_dirs:
                if os.path.exists(path):
                    model_dir = path
                    break
            
            if model_dir is None:
                raise FileNotFoundError(
                    "Could not find models directory. Tried:\n" + 
                    "\n".join(f"  - {p}" for p in possible_dirs) +
                    "\n\nPlease specify model_dir explicitly."
                )
        
        self.model_dir = model_dir
        self.device = device
        self.models = []
        self.processor = None
        self.metadata = None
        
        print("="*70)
        print("HEALTHCARE IoT PREDICTION SYSTEM v2.1")
        print("="*70)
        
        self.load_saved_models()
    
    def load_saved_models(self):
        """Load all saved models and processor"""
        print(f"\n Loading models from: {os.path.abspath(self.model_dir)}")
        
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                f"Please train models first by running: python v2-1.train.py"
            )
        
        # Load processor
        processor_path = os.path.join(self.model_dir, 'data_processor.pkl')
        if not os.path.exists(processor_path):
            raise FileNotFoundError(
                f"Processor file not found: {processor_path}\n"
                f"Please train models first."
            )
        
        with open(processor_path, 'rb') as f:
            processor_data = pickle.load(f)
        
        self.seq_scaler = processor_data['seq_scaler']
        self.stat_scaler = processor_data['stat_scaler']
        self.seq_length = processor_data['seq_length']
        print(f"   Data processor loaded (sequence length: {self.seq_length})")
        
        # Load metadata
        metadata_path = os.path.join(self.model_dir, 'metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"   Metadata loaded")
            print(f"  Training accuracy: {self.metadata['training_metrics']['accuracy']:.4f}")
            print(f"  Training F1-score: {self.metadata['training_metrics']['f1']:.4f}")
        
        # Load all fold models
        fold_idx = 0
        while True:
            fold_dir = os.path.join(self.model_dir, f'fold_{fold_idx}')
            if not os.path.exists(fold_dir):
                break
            
            fold_models = {}
            
            # Load XGBoost
            xgb_path = os.path.join(fold_dir, 'xgboost.pkl')
            if os.path.exists(xgb_path):
                fold_models['xgboost'] = joblib.load(xgb_path)
            
            # Load LightGBM
            lgb_path = os.path.join(fold_dir, 'lightgbm.pkl')
            if os.path.exists(lgb_path):
                fold_models['lightgbm'] = joblib.load(lgb_path)
            
            # Load Random Forest
            rf_path = os.path.join(fold_dir, 'random_forest.pkl')
            if os.path.exists(rf_path):
                fold_models['random_forest'] = joblib.load(rf_path)
            
            # Load Neural Network - PROPER FIX
            nn_path = os.path.join(fold_dir, 'neural_net.pth')
            if os.path.exists(nn_path):
                checkpoint = torch.load(nn_path, map_location=self.device)
                
                
                saved_config = checkpoint.get('model_config', {})
                
                
                full_config = {
                    'seq_in_dim': saved_config.get('seq_in_dim', 12),
                    'stat_in_dim': saved_config.get('stat_in_dim', 56),
                    'lstm_hidden': saved_config.get('lstm_hidden', 192),
                    'lstm_layers': saved_config.get('lstm_layers', 3),
                    'mlp_hidden': 384,   
                    'dropout': 0.15      
                }
                
                nn_model = UltraPowerfulHealthNet(**full_config)
                nn_model.load_state_dict(checkpoint['model_state_dict'])
                nn_model.to(self.device)
                nn_model.eval()
                fold_models['neural_net'] = nn_model
            
            self.models.append(fold_models)
            fold_idx += 1
        
        if len(self.models) == 0:
            raise ValueError("No models found! Please train models first.")
        
        print(f"   Loaded {len(self.models)} model folds successfully\n")
    
    def preprocess_patient_data(self, patient_data):
        """Preprocess patient vital signs data - matches training preprocessing"""
        df = pd.DataFrame(patient_data)
        
        
        required_cols = ['Temperature (°C)', 'Systolic_BP (mmHg)', 
                        'Diastolic_BP (mmHg)', 'Heart_Rate (bpm)']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}\nRequired: {required_cols}")
        
        # Create all derived features (same as training)
        df['Pulse_Pressure'] = df['Systolic_BP (mmHg)'] - df['Diastolic_BP (mmHg)']
        df['Mean_Arterial_Pressure'] = df['Diastolic_BP (mmHg)'] + (df['Pulse_Pressure'] / 3)
        df['BP_Ratio'] = df['Systolic_BP (mmHg)'] / (df['Diastolic_BP (mmHg)'] + 1e-6)
        df['BP_Product'] = df['Systolic_BP (mmHg)'] * df['Diastolic_BP (mmHg)']
        df['HR_Normalized'] = (df['Heart_Rate (bpm)'] - 70) / 30
        df['HR_BP_Ratio'] = df['Heart_Rate (bpm)'] / (df['Mean_Arterial_Pressure'] + 1e-6)
        df['Temp_Deviation'] = df['Temperature (°C)'] - 37.0
        df['Temp_Squared'] = df['Temperature (°C)'] ** 2
        
        
        df['Hypertension_Risk'] = ((df['Systolic_BP (mmHg)'] > 140) | 
                                   (df['Diastolic_BP (mmHg)'] > 90)).astype(int)
        df['Stage2_Hypertension'] = ((df['Systolic_BP (mmHg)'] > 160) | 
                                     (df['Diastolic_BP (mmHg)'] > 100)).astype(int)
        df['Tachycardia_Risk'] = (df['Heart_Rate (bpm)'] > 100).astype(int)
        df['Bradycardia_Risk'] = (df['Heart_Rate (bpm)'] < 60).astype(int)
        df['Extreme_Tachycardia'] = (df['Heart_Rate (bpm)'] > 120).astype(int)
        df['Extreme_Bradycardia'] = (df['Heart_Rate (bpm)'] < 50).astype(int)
        df['Fever_Risk'] = (df['Temperature (°C)'] > 37.5).astype(int)
        df['High_Fever_Risk'] = (df['Temperature (°C)'] > 38.5).astype(int)
        df['Hypotension_Risk'] = ((df['Systolic_BP (mmHg)'] < 90) | 
                                  (df['Diastolic_BP (mmHg)'] < 60)).astype(int)
        
        
        df['Composite_Risk_Score'] = (df['Hypertension_Risk'] * 2 + 
                                      df['Stage2_Hypertension'] * 3 +
                                      df['Tachycardia_Risk'] + 
                                      df['Bradycardia_Risk'] * 2 + 
                                      df['Fever_Risk'] +
                                      df['High_Fever_Risk'] * 2)
        
        
        df['BP_HR_Interaction'] = df['Systolic_BP (mmHg)'] * df['Heart_Rate (bpm)'] / 10000
        df['BP_HR_Sum'] = df['Systolic_BP (mmHg)'] + df['Heart_Rate (bpm)']
        df['Temp_HR_Interaction'] = df['Temperature (°C)'] * df['Heart_Rate (bpm)']
        
        
        df['Temperature (°C)_Log'] = np.log(df['Temperature (°C)'] + 1e-6)
        df['Temperature (°C)_Sqrt'] = np.sqrt(np.abs(df['Temperature (°C)']))
        df['Systolic_BP (mmHg)_Log'] = np.log(df['Systolic_BP (mmHg)'] + 1e-6)
        df['Systolic_BP (mmHg)_Sqrt'] = np.sqrt(np.abs(df['Systolic_BP (mmHg)']))
        df['Diastolic_BP (mmHg)_Log'] = np.log(df['Diastolic_BP (mmHg)'] + 1e-6)
        df['Diastolic_BP (mmHg)_Sqrt'] = np.sqrt(np.abs(df['Diastolic_BP (mmHg)']))
        df['Heart_Rate (bpm)_Log'] = np.log(df['Heart_Rate (bpm)'] + 1e-6)
        df['Heart_Rate (bpm)_Sqrt'] = np.sqrt(np.abs(df['Heart_Rate (bpm)']))
        df['Temp_Cubed'] = df['Temperature (°C)'] ** 3
        
        
        vital_cols = ['Temperature (°C)', 'Systolic_BP (mmHg)', 'Diastolic_BP (mmHg)', 'Heart_Rate (bpm)']
        for col in vital_cols:
            if col in df.columns:
                df[f'{col}_Patient_Mean'] = df[col].mean()
                df[f'{col}_Patient_Std'] = df[col].std() if len(df) > 1 else 0.0
                df[f'{col}_Patient_Min'] = df[col].min()
                df[f'{col}_Patient_Max'] = df[col].max()
                df[f'{col}_Patient_Range'] = df[f'{col}_Patient_Max'] - df[f'{col}_Patient_Min']
                df[f'{col}_Deviation'] = df[col] - df[f'{col}_Patient_Mean']
                df[f'{col}_Zscore'] = df[f'{col}_Deviation'] / (df[f'{col}_Patient_Std'] + 1e-6)
        
        
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['Hour'] = df['Timestamp'].dt.hour
            df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
            df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
            df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
            df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            df['DOW_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
            df['DOW_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        else:
           
            df['Hour'] = 12
            df['Day_of_Week'] = 2
            df['Is_Weekend'] = 0
            df['Is_Night'] = 0
            df['Hour_Sin'] = 0
            df['Hour_Cos'] = 1
            df['DOW_Sin'] = 0
            df['DOW_Cos'] = 1
        
        return df
    
    def create_sequences(self, df):
        """Create sequences from patient data - matches training"""
        
        seq_features = [
            'Temperature (°C)', 'Systolic_BP (mmHg)', 'Diastolic_BP (mmHg)', 
            'Heart_Rate (bpm)', 'Pulse_Pressure', 'Mean_Arterial_Pressure', 
            'BP_Ratio', 'HR_Normalized', 'Temp_Deviation', 'HR_BP_Ratio', 
            'BP_Product', 'Temp_Squared'
        ]
        
      
        stat_features = [
            'Hour', 'Day_of_Week', 'Is_Weekend', 'Is_Night', 'Hour_Sin', 'Hour_Cos', 
            'DOW_Sin', 'DOW_Cos', 'Hypertension_Risk', 'Stage2_Hypertension', 
            'Tachycardia_Risk', 'Bradycardia_Risk', 'Fever_Risk', 'High_Fever_Risk', 
            'BP_HR_Interaction', 'Temp_HR_Interaction', 'Composite_Risk_Score', 
            'Extreme_Tachycardia', 'Extreme_Bradycardia', 'Hypotension_Risk',
            'Temperature (°C)_Log', 'Temperature (°C)_Sqrt', 
            'Systolic_BP (mmHg)_Log', 'Systolic_BP (mmHg)_Sqrt',
            
            'Heart_Rate (bpm)_Log', 'Heart_Rate (bpm)_Sqrt', 'Temp_Cubed',
            'BP_HR_Sum'
        ]
        
       
        patient_stat_features = []
        for col in ['Temperature (°C)', 'Systolic_BP (mmHg)', 'Diastolic_BP (mmHg)', 'Heart_Rate (bpm)']:
            patient_stat_features.extend([
                f'{col}_Patient_Mean',
                f'{col}_Patient_Std',
                f'{col}_Patient_Min',
                f'{col}_Patient_Max',
                f'{col}_Patient_Range',
                f'{col}_Deviation',
                f'{col}_Zscore'
            ])
        
        stat_features.extend(patient_stat_features)
        
      
        seq_features = [f for f in seq_features if f in df.columns]
        stat_features = [f for f in stat_features if f in df.columns]
        
        
        print(f"   Created sequences: seq={len(seq_features)} features, static={len(stat_features)} features")
        
        
        seq_data = df[seq_features].values
        
        
        stat_data = df[stat_features].iloc[-1].values
        
        
        if len(seq_data) < self.seq_length:
            
            padding = np.repeat(seq_data[0:1], self.seq_length - len(seq_data), axis=0)
            seq_data = np.vstack([padding, seq_data])
        elif len(seq_data) > self.seq_length:
            
            seq_data = seq_data[-self.seq_length:]
        
        
        seq_data_scaled = self.seq_scaler.transform(seq_data)
        stat_data_scaled = self.stat_scaler.transform(stat_data.reshape(1, -1))[0]
        
        return seq_data_scaled, stat_data_scaled
    
    def create_mega_features(self, X_seq, X_stat):
        
        features = []
        
        
        features.append(np.mean(X_seq, axis=0))
        features.append(np.std(X_seq, axis=0))
        features.append(np.max(X_seq, axis=0))
        features.append(np.min(X_seq, axis=0))
        features.append(np.median(X_seq, axis=0))
        features.append(np.percentile(X_seq, 25, axis=0))
        features.append(np.percentile(X_seq, 75, axis=0))
        
        
        if X_seq.shape[0] > 1:
            features.append(X_seq[-1] - X_seq[0])
            features.append(np.mean(np.diff(X_seq, axis=0), axis=0))
            features.append(np.std(np.diff(X_seq, axis=0), axis=0))
        
        
        if X_seq.shape[0] > 2:
            slopes = []
            for j in range(X_seq.shape[1]):
                x = np.arange(X_seq.shape[0])
                y = X_seq[:, j]
                if len(np.unique(y)) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                else:
                    slope = 0
                slopes.append(slope)
            features.append(np.array(slopes))
        
        combined_features = np.concatenate(features)
        
        
        if len(X_stat) > 0:
            combined_features = np.concatenate([combined_features, X_stat])
        
        return combined_features.reshape(1, -1)
    
    def predict(self, patient_data, verbose=True):
        """Make prediction on patient data"""
        if verbose:
            print("\n" + "="*70)
            print("MAKING PREDICTION")
            print("="*70)
        
        
        df = self.preprocess_patient_data(patient_data)
        if verbose:
            print(f"   Preprocessed {len(df)} time points")
        
        
        X_seq, X_stat = self.create_sequences(df)
        if verbose:
            print(f"   Created sequences: seq={X_seq.shape}, static={X_stat.shape}")
        
        
        X_mega = self.create_mega_features(X_seq, X_stat)
        if verbose:
            print(f"   Created mega features: {X_mega.shape}")
        
        
        all_predictions = []
        
        for fold_idx, fold_models in enumerate(self.models):
            fold_preds = {}
            
            # XGBoost
            if 'xgboost' in fold_models:
                fold_preds['xgboost'] = fold_models['xgboost'].predict_proba(X_mega)[0, 1]
            
            # LightGBM
            if 'lightgbm' in fold_models:
                fold_preds['lightgbm'] = fold_models['lightgbm'].predict_proba(X_mega)[0, 1]
            
            # Random Forest
            if 'random_forest' in fold_models:
                fold_preds['random_forest'] = fold_models['random_forest'].predict_proba(X_mega)[0, 1]
            
            # Neural Network
            if 'neural_net' in fold_models:
                model = fold_models['neural_net']
                model.eval()
                
                with torch.no_grad():
                    X_seq_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
                    X_stat_tensor = torch.FloatTensor(X_stat).unsqueeze(0).to(self.device)
                    
                    logit = model(X_seq_tensor, X_stat_tensor)
                    prob = torch.sigmoid(logit).cpu().numpy()[0]
                    fold_preds['neural_net'] = prob
            
            # Ensemble with weights (same as training)
            weights = {
                'neural_net': 0.50,
                'xgboost': 0.25,
                'lightgbm': 0.15,
                'random_forest': 0.10
            }
            
            fold_ensemble = sum(fold_preds.get(k, 0) * v for k, v in weights.items())
            all_predictions.append(fold_ensemble)
        
        # Average across folds
        final_prob = np.mean(all_predictions)
        final_pred = 1 if final_prob > 0.5 else 0
        confidence = max(final_prob, 1 - final_prob)
        
        return {
            'risk_probability': final_prob,
            'risk_class': final_pred,
            'risk_level': 'HIGH RISK' if final_pred == 1 else 'LOW RISK',
            'confidence': confidence
        }
    
    def analyze_vitals(self, patient_data):
        """Analyze vital signs and provide insights"""
        df = pd.DataFrame(patient_data)
        
        insights = []
        
        # Temperature analysis
        temp = df['Temperature (°C)'].iloc[-1]
        if temp > 38.5:
            insights.append(f"  High fever detected ({temp:.1f}°C)")
        elif temp > 37.5:
            insights.append(f"  Mild fever ({temp:.1f}°C)")
        elif temp < 36.0:
            insights.append(f"  Low temperature ({temp:.1f}°C)")
        
        # Blood pressure analysis
        sbp = df['Systolic_BP (mmHg)'].iloc[-1]
        dbp = df['Diastolic_BP (mmHg)'].iloc[-1]
        
        if sbp > 160 or dbp > 100:
            insights.append(f"  Hypertension Stage 2 ({sbp:.0f}/{dbp:.0f} mmHg)")
        elif sbp > 140 or dbp > 90:
            insights.append(f"  Hypertension Stage 1 ({sbp:.0f}/{dbp:.0f} mmHg)")
        elif sbp < 90 or dbp < 60:
            insights.append(f"  Hypotension ({sbp:.0f}/{dbp:.0f} mmHg)")
        
        # Heart rate analysis
        hr = df['Heart_Rate (bpm)'].iloc[-1]
        if hr > 120:
            insights.append(f"  Extreme Tachycardia ({hr:.0f} bpm)")
        elif hr > 100:
            insights.append(f"  Tachycardia ({hr:.0f} bpm)")
        elif hr < 50:
            insights.append(f"  Extreme Bradycardia ({hr:.0f} bpm)")
        elif hr < 60:
            insights.append(f"  Bradycardia ({hr:.0f} bpm)")
        
        # Trend analysis
        if len(df) >= 3:
            temp_trend = df['Temperature (°C)'].iloc[-1] - df['Temperature (°C)'].iloc[0]
            bp_trend = df['Systolic_BP (mmHg)'].iloc[-1] - df['Systolic_BP (mmHg)'].iloc[0]
            hr_trend = df['Heart_Rate (bpm)'].iloc[-1] - df['Heart_Rate (bpm)'].iloc[0]
            
            if abs(temp_trend) > 1:
                direction = "increasing" if temp_trend > 0 else "decreasing"
                insights.append(f" Temperature {direction} ({temp_trend:+.1f}°C)")
            
            if abs(bp_trend) > 15:
                direction = "increasing" if bp_trend > 0 else "decreasing"
                insights.append(f" Blood pressure {direction} ({bp_trend:+.0f} mmHg)")
            
            if abs(hr_trend) > 10:
                direction = "increasing" if hr_trend > 0 else "decreasing"
                insights.append(f" Heart rate {direction} ({hr_trend:+.0f} bpm)")
        
        return insights



def main():
    """Example usage of the prediction system"""
    
    try:
        
        predictor = HealthcarePredictionSystem(
            model_dir=r'C:"/Users/sainikhil/Downloads/thesis/folder1/models'
        )
        
        
        print("\n" + "="*70)
        print("EXAMPLE 1: HIGH RISK PATIENT")
        print("="*70)
        
        high_risk_patient = {
            'Temperature (°C)': [37.2, 37.8, 38.3, 38.8, 39.1],
            'Systolic_BP (mmHg)': [125, 135, 145, 158, 168],
            'Diastolic_BP (mmHg)': [82, 88, 92, 98, 102],
            'Heart_Rate (bpm)': [78, 85, 95, 108, 118],
        }
        
        print("\n Patient Vital Signs:")
        df_display = pd.DataFrame(high_risk_patient)
        df_display.index = ['Reading ' + str(i+1) for i in range(len(df_display))]
        print(df_display.to_string())
        
       
        insights = predictor.analyze_vitals(high_risk_patient)
        print("\n Clinical Insights:")
        for insight in insights:
            print(f"  {insight}")
        
        
        result = predictor.predict(high_risk_patient)
        
        print("\n" + "="*70)
        print("PREDICTION RESULT")
        print("="*70)
        print(f"Risk Level:       {result['risk_level']}")
        print(f"Risk Probability: {result['risk_probability']:.1%}")
        print(f"Confidence:       {result['confidence']:.1%}")
        print(f"Classification:   {result['risk_class']} (0=Low Risk, 1=High Risk)")
        
        # Example 2: Low Risk Patient
        print("\n\n" + "="*70)
        print("EXAMPLE 2: LOW RISK PATIENT")
        print("="*70)
        
        low_risk_patient = {
            'Temperature (°C)': [36.8, 36.9, 36.8, 36.9, 37.0],
            'Systolic_BP (mmHg)': [118, 120, 119, 121, 120],
            'Diastolic_BP (mmHg)': [78, 79, 78, 80, 79],
            'Heart_Rate (bpm)': [72, 74, 73, 75, 74],
        }
        
        print("\n Patient Vital Signs:")
        df_display = pd.DataFrame(low_risk_patient)
        df_display.index = ['Reading ' + str(i+1) for i in range(len(df_display))]
        print(df_display.to_string())
        
        insights = predictor.analyze_vitals(low_risk_patient)
        print("\n Clinical Insights:")
        if insights:
            for insight in insights:
                print(f"  {insight}")
        else:
            print("     All vitals within normal range")
        
        result = predictor.predict(low_risk_patient)
        
        print("\n" + "="*70)
        print("PREDICTION RESULT")
        print("="*70)
        print(f"Risk Level:       {result['risk_level']}")
        print(f"Risk Probability: {result['risk_probability']:.1%}")
        print(f"Confidence:       {result['confidence']:.1%}")
        print(f"Classification:   {result['risk_class']} (0=Low Risk, 1=High Risk)")
        
        # Example 3: Custom patient
        print("\n\n" + "="*70)
        print("EXAMPLE 3: CUSTOM PATIENT DATA")
        print("="*70)
        print("\ntesting with own data:")
        print("""
my_patient = {
    'Temperature (°C)': [37.0, 37.2, 37.5, 37.8, 38.0],
    'Systolic_BP (mmHg)': [130, 135, 140, 145, 150],
    'Diastolic_BP (mmHg)': [85, 88, 90, 92, 95],
    'Heart_Rate (bpm)': [80, 85, 90, 95, 100],
}

result = predictor.predict(my_patient, verbose=False)
print(f"Risk: {result['risk_level']}, Probability: {result['risk_probability']:.1%}")
        """)
               
    except FileNotFoundError as e:
        print("\n" + "="*70)
        print(" ERROR: MODELS NOT FOUND")
        print("="*70)
        print(str(e))
        print("\n Please run the training script first!")
        print("   Run: python v2-1.train.py")
        print("\nMake sure the training script completes successfully and")
        print("creates the './models' directory with all model files.")
        print("\n Expected model location:")
        print("   C:\\Users\\krake\\Downloads\\nikil sweden\\trained_models\\models")
        print("="*70)
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()