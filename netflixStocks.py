from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

app = Flask(__name__)

@app.route('/predict_stock_price', methods=['GET'])
def predict_stock_price():
    data_path = r"APIS\NFLX.csv"
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    X = df[['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Volume']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    future_date = pd.DataFrame({'Year': [2024], 'Month': [4], 'Day': [24], 'Open': [df['Open'].iloc[-1]], 'High': [df['High'].iloc[-1]], 'Low': [df['Low'].iloc[-1]], 'Volume': [df['Volume'].iloc[-1]]})
    future_price = model.predict(future_date)
    return jsonify({'rmse': rmse, 'predicted_price': future_price[0]}), 200

if __name__ == '__main__':
    app.run(debug=True)
