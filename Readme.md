# UAE Rent Prediction Dashboard

![UAE Rent Prediction Banner](https://www.sealra.com/media/images/banner.jpg)

## Project Overview
This project is a web-based interactive dashboard for predicting annual rental prices of properties in the UAE using machine learning. It provides data insights, exploratory data analysis (EDA), and an intuitive user interface for users to input property features and get estimated rent values.

The application combines *Flask* and *Dash* frameworks with an XGBoost regression model trained on real rental data across major UAE cities.

---

## Features

- *Rent Prediction*: Input property features such as number of bedrooms, bathrooms, area, property type, furnishing status, and city to get a predicted annual rent.
- *Data Insights*: Interactive plots showing distributions and relationships within the rental dataset.
- *Exploratory Data Analysis (EDA)*: Detailed steps, findings, and dataset overview.
- *REST API*: Endpoint for prediction to integrate with other systems or services.

---

## Dataset

- The data contains rental listings from major UAE cities including Abu Dhabi, Dubai, Sharjah, Ajman, Ras Al Khaimah, Umm Al Quwain, and Al Ain.
- Features include property type, rent, area size, furnishing status, city, and more.
- Original dataset source: [Kaggle - Real Estate Goldmine Dubai UAE Rental Market](https://www.kaggle.com/datasets/azharsaleem/real-estate-goldmine-dubai-uae-rental-market)

---

## Technologies & Libraries

- Python 3.x
- Flask (Web server)
- Dash & Dash Bootstrap Components (Dashboard UI)
- XGBoost (Machine Learning model)
- Scikit-learn (Preprocessing & model pipeline)
- Pandas & NumPy (Data handling)
- Plotly (Interactive visualizations)

---
## Installation & Setup

### Create and activate a virtual environment (optional but recommended):

bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate


## Install dependencies

bash
pip install -r requirements.txt


## Ensure model and preprocessing files are in the src folder:

- xgb_rent_model_optimized.pkl
- scaler.pkl
- encoder.pkl

## Run the app

bash
python app.py


Open your browser at: http://localhost:8000/dash/

---

## Application Structure

- *Home*: Welcome page with navigation cards.
- *Predict*: Form inputs for rent prediction with live results.
- *Data Insights*: Interactive visualizations of dataset distributions.
- *EDA*: Detailed exploratory data analysis with dataset description, EDA steps, key findings, and encoding details.

---

## API Endpoint

POST /predict

### Request JSON example:

json
{
  "Beds": 2,
  "Baths": 2,
  "Area in square meters": 75,
  "Type": 0,
  "Furnishing": 1,
  "City": 3
}

### Response example:

json
{
  "predicted_rent": 123456.78
}

## Additional Notes

- Predictions are estimates and may vary based on the model and input data.
- The project includes sample data visualizations generated using synthetic data reflecting UAE market trends.
- The dashboard uses Bootstrap theming for a responsive and modern UI.

---
## Contact & Contributions

Feel free to open issues or submit pull requests to improve the dashboard.

---

### Developed by Team ITI

![Team ITI Logo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQI6KFPp2QZ-rAUkI30ruT8CqgNR-wPHV9EqA&s)

*Team Members:*


- [*Rania*](https://github.com/RRGrania)
- [*Rowaina*](https://github.com/Raoina) 
- [*Mohy*](https://github.com/iDourgham)
- [*Seif*](https://github.com/OPCoderman)
