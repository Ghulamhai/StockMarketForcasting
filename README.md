# Stock Price Prediction using Multivariate LSTM 

This project presents a robust system for stock price forecasting using a **Long Short-Term Memory (LSTM)** neural network. The model employs a **multivariate analysis** approach by incorporating an external factor—the presence of a pandemic—to improve prediction accuracy, especially during volatile market conditions. The system is deployed as a web application with distinct interfaces for users and administrators.

---

## Key Features

* **Accurate Forecasting:** Utilizes a 4-layer LSTM neural network to capture long-term dependencies in time-series data.
* **Multivariate Analysis:** Enhances prediction by factoring in market conditions like a pandemic, making the model more resilient to economic shocks.
* **Web Application:** A user-friendly interface built with Flask, allowing users to get real-time stock predictions.
* **User & Admin Roles:**
    * **Client:** Can register/login and view 30-day price forecasts for desired stocks.
    * **Admin:** Can generate and analyze prediction graphs for different scenarios (pandemic vs. non-pandemic).
* **Interactive Visualizations:** Historical data and predicted prices are displayed using interactive Plotly graphs for clear visual comparison.
* **Comprehensive Evaluation:** Model performance is rigorously tested using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

---

##  Technology Stack

* **Backend:** Python, Flask
* **Machine Learning:** TensorFlow, Keras, Scikit-learn
* **Data Handling & Analysis:** Pandas, NumPy
* **Data Retrieval:** `yfinance`
* **Data Visualization:** Plotly, Matplotlib
* **Development Environment:** Jupyter Notebook

---

## System Architecture

The project follows a structured workflow from data collection to deployment:

1.  **Data Collection:** Historical stock data (Open, High, Low, Close, Volume) is fetched using the `yfinance` library.
2.  **Data Preprocessing:**
    * **Feature Engineering:** A `pandemic` column is added to the dataset (1 for pandemic period, 0 otherwise) to enable multivariate analysis.
    * **Normalization:** Data is scaled to a `[-1, 1]` range using `MinMaxScaler` to optimize model training.
3.  **Train-Test Split:** The dataset is split into 65% for training and 35% for testing.
4.  **Model Training:**
    * A 4-layer stacked LSTM model is constructed with 50 neurons in the first three layers (`relu` activation) and 1 neuron in the output layer.
    * The model is compiled using the **Adam optimizer** and **Mean Squared Error** as the loss function.
5.  **Prediction & Forecasting:** The trained model predicts the next day's stock price. This process is repeated iteratively to forecast prices for the next 30 days.
6.  **Web Application:** The trained model is integrated into a Flask web application that serves predictions to users.




## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    A `requirements.txt` file should contain the following:
    ```
    pandas
    numpy
    yfinance
    tensorflow
    scikit-learn
    plotly
    flask
    ```
    Install them using pip:
    ```sh
    pip install -r requirements.txt
    ```

4.  **Train the Model:**
    Run the training script to fetch data and train the LSTM model. The script will save the trained model as an `.h5` file.
    ```python
    # (From the provided source code in 4.4.1)
    import yfinance as yf
    # ... rest of the training code ...
    model.save('Reliance_Mul_1_LSTM.h5')
    ```

5.  **Run the Flask Application:**
    Start the web server to launch the application.
    ```sh
    flask run
    ```
    Navigate to `http://127.0.0.1:5000` in your web browser.

---

## Results and Analysis

The project successfully demonstrates that a multivariate LSTM model can effectively forecast stock prices.

* **Performance:** The model achieved low MAE, MSE, and RMSE values on the test dataset, indicating a high degree of prediction accuracy.
* **Univariate vs. Multivariate:** While the univariate model performed well, the multivariate model offers the critical advantage of adapting to major economic events (like a pandemic), making its predictions more robust and context-aware.
* **Visual Validation:** The prediction graphs show that the model's forecasts closely follow the actual trends of the stock price, confirming its ability to learn underlying patterns.

| Evaluation Metrics (Test Data) | Multivariate LSTM Result |
| :----------------------------- | :----------------------- |
| **MAE** | 0.0179                   |
| **MSE** | 0.00099                  |
| **RMSE** | 0.0314                   |

---

## Future Work

To further enhance this project, the following directions can be explored:

* **Incorporate More Features:** Add other data sources like news sentiment, social media trends, and macroeconomic indicators.
* **Explore Ensemble Methods:** Combine the LSTM model with other algorithms (e.g., ARIMA, Random Forest) to improve prediction stability and accuracy.
* **Hyperparameter Tuning:** Systematically optimize model parameters (e.g., number of layers, neurons, learning rate) for even better performance.
* **Enhance Interpretability:** Use techniques like SHAP (SHapley Additive exPlanations) to better understand the model's decision-making process.
