import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fastapi import FastAPI
import uvicorn

# Step 1: Data Collection (Synthetic merchant data)
data = pd.DataFrame({
    'merchant_id': range(1000),
    'transaction_volume': np.random.normal(10000, 2000, 1000),
    'avg_transaction_size': np.random.normal(50, 10, 1000),
    'fraud_rate': np.random.uniform(0, 0.05, 1000),
    'merchant_type': np.random.choice(['retail', 'ecommerce', 'hospitality'], 1000),
    'credit_score': np.random.randint(300, 850, 1000),
    'processing_fee': np.random.uniform(0.01, 0.03, 1000)  # Target
})

# Step 2: Data Preprocessing
def preprocess_data(df):
    # Encode categorical variables
    df['merchant_type'] = df['merchant_type'].astype('category').cat.codes
    
    # Scale numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop(columns=['processing_fee']))
    df_scaled = pd.DataFrame(scaled_features, columns=df.drop(columns=['processing_fee']).columns)
    df_scaled['processing_fee'] = df['processing_fee']
    
    return df_scaled, scaler

# Step 3: Feature Engineering
def engineer_features(df):
    df['risk_score'] = df['fraud_rate'] * (850 - df['credit_score']) / 850
    df['volume_per_transaction'] = df['transaction_volume'] / df['avg_transaction_size']
    return df

# Step 4: Model Development
def train_model(X_train, y_train):
    model = CatBoostRegressor(random_state=42, verbose=0)
    model.fit(X_train, y_train)
    return model

# Step 5: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    return rmse, mae

# Step 6: FastAPI Deployment
app = FastAPI()

@app.post("/predict_pricing")
async def predict_pricing(merchant: dict):
    merchant_df = pd.DataFrame([merchant])
    merchant_scaled = scaler.transform(merchant_df)
    fee = model.predict(merchant_scaled)[0]
    return {"predicted_fee": float(fee)}

# Main Pipeline
if __name__ == "__main__":
    # Preprocess and engineer features
    data_scaled, scaler = preprocess_data(data)
    data_engineered = engineer_features(data_scaled)
    
    # Split data
    X = data_engineered.drop(columns=['processing_fee'])
    y = data_engineered['processing_fee']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    rmse, mae = evaluate_model(model, X_test, y_test)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
