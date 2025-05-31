# UAE Rent Prediction Dashboard

## 🚀 Project Overview

This project provides an interactive web dashboard for predicting annual rent prices in the UAE, powered by a machine learning model. It leverages Flask for the backend API and Dash with Plotly for the interactive user interface, enhanced with `dash-bootstrap-components` for a professional look.

The dashboard allows users to input various property features (like number of beds, baths, area, property type, furnishing, and city) and receive an instant estimated annual rent. Additionally, it offers data insights through visualizations of property distributions.

## ✨ Features

* **Interactive Rent Prediction:** Get estimated annual rent based on user-defined property attributes.
* **Machine Learning Backend:** Utilizes a pre-trained XGBoost model for predictions.
* **Dynamic Data Visualizations:** Explore distributions of key features like property type, city, and area.
* **Professional UI:** Built with `dash-bootstrap-components` for a clean, responsive, and modern design.
* **API Endpoint:** A dedicated `/predict` API endpoint for programmatic access to the prediction model.
* **Collapsible Navigation:** Features an `Offcanvas` sidebar for clean navigation on all screen sizes.
* **Input Icons:** Intuitive icons for input fields to enhance user experience.
* **Loading Indicators:** Provides visual feedback during prediction processing.

## 🛠️ Technologies Used

* **Python 3.x**
* **Flask:** Web framework for the backend API.
* **Dash:** Python framework for building analytical web applications.
* **Dash Bootstrap Components (dbc):** For beautiful and responsive layouts.
* **Plotly:** For creating interactive charts and visualizations.
* **Pandas:** Data manipulation and analysis.
* **NumPy:** Numerical operations.
* **Scikit-learn:** For data preprocessing (StandardScaler, OneHotEncoder).
* **Joblib:** For saving and loading machine learning models and preprocessors.
* **XGBoost:** The machine learning algorithm used for rent prediction.

## 📦 Installation & Setup

Follow these steps to get the project up and running on your local machine.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/RRGrania/UAE-Rent-Prediction-Dashboard.git](https://github.com/RRGrania/UAE-Rent-Prediction-Dashboard.git)
    cd UAE-Rent-Prediction-Dashboard
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install project dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (If `requirements.txt` doesn't exist, create it by running `pip freeze > requirements.txt` after installing all packages, or manually list them: `dash dash-bootstrap-components plotly pandas scikit-learn joblib Flask numpy`).

5.  **Obtain Machine Learning Models and Preprocessors:**
    The application relies on three `.pkl` files for prediction:
    * `xgb_rent_model_optimized.pkl` (the trained XGBoost model)
    * `scaler.pkl` (the StandardScaler object)
    * `encoder.pkl` (the OneHotEncoder object)

    **You need to place these three files inside a directory named `src` in your project root.**
    Create the directory if it doesn't exist:
    ```bash
    mkdir src
    ```
    Then, move or copy your `.pkl` files into the `src` directory.
    *(If you don't have these files, you'll need to train your model and save these objects first, or obtain them from the project maintainer.)*

## 🚀 Usage

### Running the Dashboard

Once all dependencies are installed and model files are in the `src` directory:

1.  **Start the Flask/Dash application:**
    ```bash
    python app.py
    ```
2.  **Access the Dashboard:**
    Open your web browser and navigate to:
    ```
    [http://127.0.0.1:8000/dash/](http://127.0.0.1:8000/dash/)
    ```
    (Or `http://localhost:8000/dash/`)

### Using the Prediction API

You can also send POST requests to the `/predict` endpoint to get predictions programmatically.

**Endpoint:** `http://127.0.0.1:8000/predict`
**Method:** `POST`
**Content-Type:** `application/json`

**Example Request Body:**

```json
{
    "Beds": 2,
    "Baths": 2,
    "Area in square meters": 100,
    "Type": 0,       
    "Furnishing": 1, 
    "City": 3      
}