Stock Price Forecasting Using LSTM
Overview
This project focuses on developing a stock price forecasting system using Long Short-Term Memory (LSTM) neural networks. The system leverages historical stock price data and incorporates multivariate analysis to improve prediction accuracy, particularly during challenging market conditions such as pandemics or recessions. The project includes a web application built with Flask, providing a user-friendly interface for both clients and admins to interact with stock price predictions.
Features

LSTM-Based Prediction: Utilizes a 4-layer LSTM neural network to predict stock prices based on historical data.
Multivariate Analysis: Incorporates a "Pandemic" feature to enhance prediction accuracy during specific market conditions.
Web Application: Built with Flask, offering:
Client Area: Allows users to log in, select a company, and view predicted stock prices with interactive graphs.
Admin Area: Enables admins to generate prediction graphs for specific companies, with options to account for pandemic or non-pandemic scenarios.


Data Visualization: Uses Plotly and Matplotlib to display historical and predicted stock prices.
Dataset: Historical stock data sourced from Yahoo Finance using the yfinance library.

Installation
Prerequisites

Hardware:
Intel Core i5 processor
8 GB RAM
512 GB SSD


Software:
Python 3.10.11
Jupyter Notebook
Required Python libraries (listed below)



Libraries and Versions



Library
Version



Python
3.10.11


Pandas
1.4.4


scikit-learn
1.1.2


Matplotlib
3.5.1


TensorFlow
2.11.0


yfinance
0.2.14


Setup Instructions

Clone the Repository:
git clone https://github.com/Ghulamhai/StockMarketForcasting.git
cd stock-price-forecasting


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download Dataset:

The project uses stock data from Yahoo Finance. The dataset is downloaded automatically using the yfinance library during script execution.


Run the Application:
python app.py


Access the web application at http://localhost:5000.



Project Structure

app.py: Main Flask application file for the web interface.
Reliance.csv: Sample dataset for Reliance stock (downloaded via yfinance).
Reliance_Mul_1_LSTM.h5: Trained LSTM model file.
Reliance_Graph_pan_0.html: Generated HTML file for visualization of predictions.
templates/: Directory containing HTML templates for the web interface.
static/: Directory for static files (e.g., CSS, JavaScript).
notebooks/: Jupyter notebooks for data preprocessing, model training, and evaluation.

Usage

Client Usage:

Register or log in to the client area.
Select a company to view its stock price predictions.
Visualize historical and predicted stock prices through interactive graphs.


Admin Usage:

Log in to the admin area.
Choose a company and specify whether to account for pandemic conditions.
Generate and view prediction graphs.


Running the LSTM Model:

Use the provided Jupyter notebook (notebooks/train_test_LSTM.ipynb) to train the LSTM model and evaluate its performance.
The notebook includes steps for data preprocessing, model training, and prediction generation.



Methodology

Data Collection:

Historical stock data is sourced from Yahoo Finance using the yfinance library.
A "Pandemic" column is added to indicate pandemic (1) or non-pandemic (0) periods.


Feature Engineering:

Selected features: Close price and Pandemic indicator.
Data is normalized using MinMaxScaler for model training.


Model Architecture:

A 4-layer LSTM model with 50 neurons in the first three layers (ReLU activation) and 1 neuron in the output layer.
Optimizer: Adam
Loss function: Mean Squared Error (MSE)


Train-Test Split:

65% training data, 35% testing data.


Evaluation Metrics:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)


Prediction:

Uses the last 100 days of data to predict the next 30 days.
Predictions are visualized using Plotly graphs.



Results

Univariate LSTM (Reliance Dataset):
Train: MAE: 0.00159, MSE: 8.64e-06, RMSE: 0.00294
Test: MAE: 0.01658, MSE: 0.00070, RMSE: 0.02651


Multivariate LSTM (Reliance Dataset):
Train: MAE: 0.00175, MSE: 7.40e-06, RMSE: 0.00272
Test: MAE: 0.01791, MSE: 0.00099, RMSE: 0.03147


The multivariate LSTM model maintains similar accuracy to the univariate model while adding the capability to handle pandemic scenarios.

Future Work

Incorporate additional data sources (e.g., news sentiment, economic indicators).
Explore ensemble methods combining LSTM with ARIMA or random forests.
Enhance model interpretability and long-term forecasting accuracy.

Limitations

Limited historical context and sensitivity to input features.
Challenges with non-stationarity of stock prices and long-term predictions.
Dependence on data quality and availability.

Conclusion
This project demonstrates the effectiveness of LSTM models in stock price forecasting, with the multivariate approach offering improved handling of specific market conditions like pandemics. The web application provides an accessible interface for users to interact with predictions, making it a valuable tool for investors and traders.
License
This project is licensed under the MIT License.
