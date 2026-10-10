

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
# REMOVE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data_path = r'/Users/sainikhil/Downloads/thesis /folder1/healthcare_iot_target_dataset.csv'

print("Loading dataset...")
print(f"Reading from: {data_path}")
df = pd.read_csv(data_path)

print("\nDataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Variable Distribution:")
print(df['Target_Health_Status'].value_counts())

# EDA - Exploratory Data Analysis
print("\n" + " "*50)
print("Starting EDA - Creating Plots")
print("="*50)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Plot 1: Target Distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

df['Target_Health_Status'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
axes[0, 0].set_title('Target Health Status Distribution')
axes[0, 0].set_xlabel('Health Status')
axes[0, 0].set_ylabel('Count')

# Plot 2: Temperature Distribution
axes[0, 1].hist(df['Temperature (°C)'].dropna(), bins=30, color='orange', edgecolor='black')
axes[0, 1].set_title('Temperature Distribution')
axes[0, 1].set_xlabel('Temperature (°C)')
axes[0, 1].set_ylabel('Frequency')

# Plot 3: Heart Rate Distribution
axes[1, 0].hist(df['Heart_Rate (bpm)'].dropna(), bins=30, color='red', edgecolor='black')
axes[1, 0].set_title('Heart Rate Distribution')
axes[1, 0].set_xlabel('Heart Rate (bpm)')
axes[1, 0].set_ylabel('Frequency')

# Plot 4: Blood Pressure Scatter
axes[1, 1].scatter(df['Systolic_BP (mmHg)'].dropna(), df['Diastolic_BP (mmHg)'].dropna(), alpha=0.5, color='blue')
axes[1, 1].set_title('Blood Pressure Scatter Plot')
axes[1, 1].set_xlabel('Systolic BP (mmHg)')
axes[1, 1].set_ylabel('Diastolic BP (mmHg)')

plt.tight_layout()
plt.savefig('eda_plots_1.png', dpi=300, bbox_inches='tight')
print("\nSaved: eda_plots_1.png")
plt.show()




# Plot 5: Correlation Heatmap
plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: correlation_heatmap.png")
plt.show()




# Plot 6: Sensor Type Distribution
plt.figure(figsize=(10, 6))
df['Sensor_Type'].value_counts().plot(kind='bar', color='purple')
plt.title('Sensor Type Distribution')
plt.xlabel('Sensor Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sensor_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: sensor_distribution.png")
plt.show()



# Data Preprocessing
print("\n" + " "*50)
print("Data Preprocessing")
print("="*50)

# Handle missing values
df_clean = df.copy()
df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

# Encode categorical variables
le_sensor = LabelEncoder()
le_target = LabelEncoder()

if 'Sensor_Type' in df_clean.columns:
    df_clean['Sensor_Type_Encoded'] = le_sensor.fit_transform(df_clean['Sensor_Type'])

# Encode target separately
y_encoded = le_target.fit_transform(df_clean['Target_Health_Status'])

# Use all sensor measurements including target thresholds
features_to_use = [
    'Temperature (°C)',
    'Systolic_BP (mmHg)',
    'Diastolic_BP (mmHg)',
    'Heart_Rate (bpm)',
    'Device_Battery_Level (%)',
    'Target_Blood_Pressure',
    'Target_Heart_Rate',
    'Battery_Level (%)',
    'Sensor_Type_Encoded'
]

X = df_clean[features_to_use].copy()
y = y_encoded

print("\nFeatures selected:", X.columns.tolist())
print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# Add significant noise to reduce overfitting
np.random.seed(42)
noise_levels = {
    'Temperature (°C)': 0.4,
    'Systolic_BP (mmHg)': 10.0,
    'Diastolic_BP (mmHg)': 7.0,
    'Heart_Rate (bpm)': 8.0,
    'Device_Battery_Level (%)': 12.0,
    'Target_Blood_Pressure': 5.0,
    'Target_Heart_Rate': 5.0,
    'Battery_Level (%)': 12.0
}

for col in noise_levels:
    if col in X.columns:
        noise = np.random.normal(0, noise_levels[col], size=len(X))
        X[col] = X[col] + noise

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("\nTrain set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("\nScaler saved as: scaler.pkl")

# Model Training with heavy regularization
print("\n" + " "*50)
print("Model Training and Evaluation")
print("="*50)

models = {
    'XGBoost': XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        max_depth=3,
        n_estimators=30,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.0
    ),
    'LightGBM': LGBMClassifier(
        random_state=42,
        verbose=-1,
        num_threads=1,          # ← THIS is what actually stops the hang
        deterministic=True,
        max_depth=3, 
        n_estimators=30,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=5,
        reg_alpha=0.5,
        reg_lambda=1.0
    ),
    'Random Forest': RandomForestClassifier(
        random_state=42,
        n_estimators=30,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        max_samples=0.7
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    print(f"{'='*50}")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, average='weighted')
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
    
    # Store results
    results[name] = {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'f1_score': test_f1,
        'recall': test_recall,
        'precision': test_precision
    }
    
    # Print results
    print(f"\n{name} Results:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"Precision: {test_precision:.4f}")
    
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred_test, target_names=le_target.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: confusion_matrix_{name.replace(' ', '_')}.png")
    plt.show()


    
    # Save the model
    model_filename = f'model_{name.replace(" ", "_")}.pkl'
    joblib.dump(model, model_filename)
    print(f"Model saved as: {model_filename}")

# Compare all models
print("\n" + " "*50)
print("Model Comparison Summary")
print("="*50)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Accuracy': [results[m]['train_accuracy'] for m in results],
    'Test Accuracy': [results[m]['test_accuracy'] for m in results],
    'F1 Score': [results[m]['f1_score'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'Precision': [results[m]['precision'] for m in results]
})

print("\n", comparison_df)

# Test on custom sample data
print("\n" + " "*50)
print("Testing on Custom Sample Data - Including Challenging Cases")
print("="*50)

# Create diverse test cases with some that should be misclassified
sample_data = pd.DataFrame({
    'Temperature (°C)': [37.2, 38.9, 36.1, 37.5, 39.2, 36.8, 38.0, 37.0, 37.8, 38.5],
    'Systolic_BP (mmHg)': [120, 145, 110, 135, 150, 115, 130, 125, 138, 142],
    'Diastolic_BP (mmHg)': [80, 95, 70, 85, 98, 75, 88, 82, 90, 93],
    'Heart_Rate (bpm)': [72, 98, 65, 88, 105, 68, 92, 75, 95, 100],
    'Device_Battery_Level (%)': [85, 45, 92, 60, 30, 88, 55, 78, 50, 40],
    'Target_Blood_Pressure': [120, 120, 120, 130, 130, 120, 130, 120, 130, 120],
    'Target_Heart_Rate': [75, 75, 75, 80, 80, 75, 80, 75, 80, 75],
    'Battery_Level (%)': [85, 45, 92, 60, 30, 88, 55, 78, 50, 40],
    'Sensor_Type_Encoded': [1, 2, 0, 1, 2, 0, 1, 2, 1, 2]
})

sample_data = sample_data[X.columns]

# Scale the sample data
sample_data_scaled = scaler.transform(sample_data.values)


print("\nSample Test Data:")
print(sample_data)

# Test with all models
print("\n" + " "*50)
print("Predictions on Sample Data", flush=True)
print("="*50)

for name, result in results.items():
    model = result['model']
    predictions = model.predict(sample_data_scaled)
    predictions_label = le_target.inverse_transform(predictions)
    
    print(f"\n{name} Predictions:", flush=True)
    for i, pred in enumerate(predictions_label):
        print(f"Sample {i+1}: {pred}")

# Additional edge cases
print("\n" + " "*50)
print("Testing on Edge Cases")
print("="*50)

edge_data = pd.DataFrame({
    'Temperature (°C)': [36.0, 39.5, 37.3],
    'Systolic_BP (mmHg)': [100, 160, 125],
    'Diastolic_BP (mmHg)': [65, 100, 83],
    'Heart_Rate (bpm)': [55, 110, 78],
    'Device_Battery_Level (%)': [95, 25, 70],
    'Target_Blood_Pressure': [120, 130, 120],
    'Target_Heart_Rate': [75, 80, 75],
    'Battery_Level (%)': [95, 25, 70],
    'Sensor_Type_Encoded': [0, 2, 1]
})

edge_data = edge_data[X.columns]
edge_data_scaled = scaler.transform(edge_data.values)


print("\nEdge Test Cases:")
print(edge_data)

for name, result in results.items():
    model = result['model']
    predictions = model.predict(edge_data_scaled)
    predictions_label = le_target.inverse_transform(predictions)
    
    print(f"\n{name} Predictions on Edge Cases:")
    for i, pred in enumerate(predictions_label):
        print(f"Case {i+1}: {pred}")

print("\n" + " "*50)




