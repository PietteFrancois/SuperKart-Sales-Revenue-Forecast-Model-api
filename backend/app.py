
#Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
SuperKart_Forecast_api = Flask("SuperKart Sales Revenue Forecast")

# Load the trained machine learning model
model = joblib.load("SuperKart_Sales_Forecast_model_v1_0.joblib")

# Define a route for the home page (GET request)
@SuperKart_Forecast_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Revenue Forecast API!"

# Define an endpoint for single sales revenue prediction (POST request)
@SuperKart_Forecast_api.post('/v1/predict')
def predict_sales_revenue():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing property details and returns
    the predicted sales revenue as a JSON response.
    """
    # Get the JSON data from the request body
    data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': data['Product_Weight'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_MRP': data['Product_MRP'],
        'Store_Age_Years': data['Store_Age_Years'],
        'Product_Id_char': data['Product_Id_char'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Type_Category': data['Product_Type_Category'],
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
                    
        
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction (get Sales Revenue)
    prediction = model.predict(input_data).tolist()[0]

    # Return the actual price
    return jsonify({'Predicted Sales Revenue (in dollars)': prediction})


# Define an endpoint for batch prediction (POST request)
@SuperKart_Forecast_api.post('/v1/predictbatch')
def predict_sales_revenue_batch():
    """
    This function handles POST requests to the '/v1/salesbatch' endpoint.
    It expects a CSV file containing property details for multiple properties
    and returns the predicted rental prices as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make Forecast for all line entries in the DataFrame
    predictions = model.predict(input_data).tolist()

    # Create a dictionary of forecasts with row index as ID
    output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    SuperKart_Forecast_api.run(debug=True)
