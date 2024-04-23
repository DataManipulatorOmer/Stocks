# Stock Price Prediction API

## Introduction

This API predicts future stock prices using a linear regression model trained on historical stock data.

## Send a POST Request

Use a tool like Postman to send a POST request to the `/predict_stock_price` endpoint with the following JSON data:

```json
{
    "file_path": "path/to/your/csv/file.csv",
    "year": 2024,
    "month": 4,
    "day": 24
}
