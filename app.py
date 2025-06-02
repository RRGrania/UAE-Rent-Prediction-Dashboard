import os
import joblib
import numpy as np
import logging
from flask import Flask, request, jsonify, redirect
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
df = pd.read_csv("cleaned_dubai_rental_data.csv")
# -----------------------
# Load model, scaler, and encoder
# -----------------------
def load_model_preprocessors():
    try:
        model = joblib.load("src/xgb_rent_model_optimized.pkl")
        scaler = joblib.load("src/scaler.pkl")
        ohe = joblib.load("src/encoder.pkl")
        logger.info("Loaded model, scaler, and encoder successfully.")
        return model, scaler, ohe
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure 'xgb_rent_model_optimized.pkl', 'scaler.pkl', and 'encoder.pkl' are in the 'src' directory.")
    except Exception as e:
        logger.error(f"Error loading model or preprocessors: {e}")
    return None, None, None

model, scaler, ohe = load_model_preprocessors()

# -----------------------
# Flask & Dash Initialization
# -----------------------
server = Flask(__name__)
dash_app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname="/dash/",
    external_stylesheets=[dbc.themes.CERULEAN, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}]
)

# -----------------------
# Feature config & mappings
# -----------------------
feature_config = {
    "Beds": {"min": 0, "max": 10, "default": 2, "icon": "fa-solid fa-bed"},
    "Baths": {"min": 0, "max": 10, "default": 2, "icon": "fa-solid fa-bath"},
    "Area in square meters": {"min": 10, "max": 1000, "default": 75, "icon": "fa-solid fa-ruler-combined"},
    "Type": {"min": 0, "max": 8, "default": 0, "icon": "fa-solid fa-building"},
    "Furnishing": {"min": 0, "max": 1, "default": 0, "icon": "fa-solid fa-couch"},
    "City": {"min": 0, "max": 7, "default": 0, "icon": "fa-solid fa-city"}
}

categorical_features = ["Type", "Furnishing", "City"]
category_mappings = {
    "Type": {
        0: "Apartment", 1: "Penthouse", 2: "Villa", 3: "Townhouse", 4: "Villa Compound",
        5: "Residential Building", 6: "Residential Floor", 7: "Hotel Apartment", 8: "Residential Plot"
    },
    "Furnishing": {0: "Unfurnished", 1: "Furnished"},
    "City": {
        0: "Abu Dhabi", 1: "Ajman", 2: "Al Ain", 3: "Dubai",
        4: "Fujairah", 5: "Ras Al Khaimah", 6: "Sharjah", 7: "Umm Al Quwain"
    },
}

city_coordinates = {
    "Abu Dhabi": {"lat": 24.4667, "lon": 54.3667, "predicted_rent": 85000},
    "Dubai": {"lat": 25.2048, "lon": 55.2708, "predicted_rent": 120000},
    "Sharjah": {"lat": 25.3575, "lon": 55.3908, "predicted_rent": 50000},
    "Ajman": {"lat": 25.4136, "lon": 55.4456, "predicted_rent": 40000},
    "Al Ain": {"lat": 24.2075, "lon": 55.7447, "predicted_rent": 70000},
    "Ras Al Khaimah": {"lat": 25.7667, "lon": 55.9500, "predicted_rent": 60000},
    "Fujairah": {"lat": 25.1222, "lon": 56.3344, "predicted_rent": 45000},
    "Umm Al Quwain": {"lat": 25.5533, "lon": 55.5475, "predicted_rent": 35000},
}

# -----------------------
# Bootstrap colors for plots
# -----------------------
BOOTSTRAP_COLORS = {
    'primary': '#007bff',
    'secondary': '#6c757d',
    'info': '#17a2b8',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'light': '#f8f9fa',
    'dark': '#343a40'
}
# -----------------------
# Navbar
# -----------------------
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/dash/", active="exact")),
        dbc.NavItem(dbc.NavLink("Predict", href="/dash/predict", active="exact")),
        dbc.NavItem(dbc.NavLink("Data Insights", href="/dash/insights", active="exact")),
        dbc.NavItem(dbc.NavLink("EDA", href="/dash/eda", active="exact")),
    ],
    brand="UAE Rent Guide",
    brand_href="/dash/",
    color="primary",
    dark=True,
    className="mb-4 shadow-sm rounded",  
    fluid=True,
    sticky="top",  
    style={"fontWeight": "600", "letterSpacing": "1px"}  
)

# -----------------------
# Helper functions to create input cards
# -----------------------
def create_input_card(name, config):
    return dbc.Card(
        dbc.CardBody([
            html.Label(name, className="form-label mb-2 fw-bold"),
            dbc.InputGroup([
                dbc.InputGroupText(html.I(className=config["icon"])),
                dbc.Input(
                    id=name,
                    type="number",
                    min=config["min"],
                    max=config["max"],
                    step=config.get("step", 1),
                    value=config["default"],
                    placeholder=f"Enter {name}",
                    style={"width": "100%"}  
                )
            ], className="shadow-sm rounded-lg")
        ]),
        className="mb-3 border-0"
    )

def create_dropdown_card(name, options_dict, default_idx, icon):
    options = [{"label": label, "value": idx} for idx, label in options_dict.items()]
    return dbc.Card(
        dbc.CardBody([
            html.Label(name, className="form-label mb-2 fw-bold"),
            dbc.InputGroup([
                dbc.InputGroupText(html.I(className=icon)),
                dcc.Dropdown(
                    id=name,
                    options=options,
                    value=default_idx,
                    clearable=False,
                    searchable=True,
                    className="form-select border-0",
                    style={"width": "100%"}  
                )
            ], className="shadow-sm rounded-lg")
        ]),
        className="mb-3 border-0"
    )

# -----------------------
# Build Pages
# -----------------------

def build_home_page():
    return dbc.Container([
        html.Div(
            style={"position": "relative", "marginBottom": "30px", "borderRadius": "8px", "overflow": "hidden"},
            children=[
                html.Img(
                    src="https://www.sealra.com/media/images/banner.jpg",
                    className="banner-image",
                    style={
                        "width": "100%",
                        "height": "250px",
                        "objectFit": "cover",
                        "filter": "brightness(0.85)"
                    }
                ),
                # Optional overlay text (commented out)
                # html.Div(
                #     "UAE Rent Predictor",
                #     style={
                #         "position": "absolute",
                #         "top": "50%",
                #         "left": "50%",
                #         "transform": "translate(-50%, -50%)",
                #         "color": "white",
                #         "fontSize": "2.5rem",
                #         "fontWeight": "bold",
                #         "textShadow": "2px 2px 8px rgba(0,0,0,0.7)"
                #     }
                # )
            ]
        ),
        html.H1("Rent Analytics Hub", className="text-center text-primary fw-bold mb-3"),
        html.P("Choose an option below to get started.", className="text-center text-muted mb-5 fs-5"),
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H4("🔍 Predict Rent", className="card-title text-primary fw-bold"),
                        html.P("Input property features and get an estimated annual rent.", className="mb-3"),
                        dbc.Button("Go to Prediction", href="/dash/predict", color="primary", className="mt-auto")
                    ]),
                    className="shadow-sm rounded h-100",
                    style={"transition": "transform 0.3s", "cursor": "pointer"},
                    id="card-predict"
                ),
                md=4
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H4("📊 Data Insights", className="card-title text-info fw-bold"),
                        html.P("View distributions and relationships in the dataset.", className="mb-3"),
                        dbc.Button("Explore Insights", href="/dash/insights", color="info", className="mt-auto")
                    ]),
                    className="shadow-sm rounded h-100",
                    style={"transition": "transform 0.3s", "cursor": "pointer"},
                    id="card-insights"
                ),
                md=4
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H4("📈 Exploratory Data Analysis", className="card-title text-success fw-bold"),
                        html.P("Review the data exploration steps and findings.", className="mb-3"),
                        dbc.Button("View EDA", href="/dash/eda", color="success", className="mt-auto")
                    ]),
                    className="shadow-sm rounded h-100",
                    style={"transition": "transform 0.3s", "cursor": "pointer"},
                    id="card-eda"
                ),
                md=4
            )
        ], className="g-4 justify-content-center")
    ], className="pt-5 px-3")

# Optional: Add Dash callbacks or CSS to add hover scale effect for cards:
# e.g. in assets/style.css:
# .card:hover {
#     transform: scale(1.05);
#     box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
# }
def build_prediction_page():
    prediction_inputs = []
    for name, cfg in feature_config.items():
        if name in categorical_features:
            prediction_inputs.append(create_dropdown_card(name, category_mappings[name], cfg["default"], cfg["icon"]))
        else:
            prediction_inputs.append(create_input_card(name, cfg))
    col1_inputs = prediction_inputs[0::2]
    col2_inputs = prediction_inputs[1::2]

    return dbc.Container(
        className="py-5",
        children=[
            # Banner Image with animation
            html.Div(
                html.Img(
                    src="https://strapiallsopp.s3.eu-west-1.amazonaws.com/banner_1_114f7cb85f.jpg",
                    className="banner-image",
                    style={
                        "width": "100%",
                        "height": "280px",
                        "objectFit": "cover",
                        "borderRadius": "8px",
                        "marginBottom": "30px",
                        "filter": "brightness(0.85)"
                    }
                ),
               className="fade-in"
            ),

            dbc.Card([
                dbc.CardHeader(
                    html.H2("Predict Annual Rent", className="text-center text-primary mb-0 fw-bold")
                ),
                dbc.CardBody([
                    html.P(
                        "Enter the property details to get an estimated annual rent in AED.",
                        className="text-center text-muted mb-4 fs-5"
                    ),
                    dbc.Row(className="g-4", children=[
                        dbc.Col(col1_inputs, md=6),
                        dbc.Col(col2_inputs, md=6)
                    ]),
                    dbc.Row(className="mt-4", children=[
                        dbc.Col([
                            dbc.Button(
                                [html.I(className="fa-solid fa-magnifying-glass me-2"), "Get Prediction"],
                                id="predict-btn",
                                color="primary",
                                size="lg",
                                className="d-block mx-auto mb-3 w-75 shadow"
                            ),
                            dcc.Loading(
                                id="loading-output",
                                type="circle",
                                children=html.Div(
                                    id="prediction-output",
                                    className="text-center fs-4 fw-bold text-dark mt-3"
                                )
                            )
                        ], width=12, className="d-flex flex-column align-items-center")
                    ])
                ]),
                dbc.CardFooter(
                    html.Small("Prediction results are estimates and may vary.", className="text-muted fst-italic")
                )
            ], className="shadow-lg rounded-lg border-0")
        ]
    )


def build_insights_page():
    return dbc.Container([
        html.Div(
            html.Img(
                src="https://s3-eu-west-1.amazonaws.com/ebu-itweb-eurovision-prod/media_public/insights/0001/05/thumb_4185_insights_banner.jpeg",
                className="banner-image",
                style={
                    "width": "100%",
                    "height": "250px",
                    "objectFit": "cover",
                    "borderRadius": "12px",
                    "marginBottom": "40px",
                    "filter": "brightness(0.85)",
                    "boxShadow": "0 4px 15px rgba(0,0,0,0.3)"
                }
            ),
            className="fade-in"
        ),
        html.H2(
            "Data Distribution Insights",
            className="text-center",
            style={
                "color": "#007bff",
                "fontWeight": "700",
                "marginBottom": "40px",
                "fontSize": "2.5rem",
            }
        ),

        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                  
                        html.H5(
                            "Area Distribution (sqm)",
                            style={
                                "color": "#004085",
                                "fontWeight": "700",
                                "marginBottom": "12px",
                                "fontSize": "1.25rem"
                            }
                        ),
                        dcc.Graph(id='area-distribution-plot')
                    ])
                ], className="shadow-sm rounded-lg border-0 mb-4"),
                md=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5(
                            "Property Type Distribution",
                            style={
                                "color": "#004085",
                                "fontWeight": "700",
                                "marginBottom": "12px",
                                "fontSize": "1.25rem"
                            }
                        ),
                        dcc.Graph(id='type-distribution-plot')
                    ])
                ], className="shadow-sm rounded-lg border-0 mb-4"),
                md=6
            )
        ]),

        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5(
                            "City Distribution",
                            style={
                                "color": "#004085",
                                "fontWeight": "700",
                                "marginBottom": "12px",
                                "fontSize": "1.25rem"
                            }
                        ),
                        dcc.Graph(id='city-distribution-plot')
                    ])
                ], className="shadow-sm rounded-lg border-0 mb-4"),
                md=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5(
                            "Rent Distribution by Property Type",
                            style={
                                "color": "#004085",
                                "fontWeight": "700",
                                "marginBottom": "12px",
                                "fontSize": "1.25rem"
                            }
                        ),
                        dcc.Graph(id='rent-by-type-violin-plot')
                    ])
                ], className="shadow-sm rounded-lg border-0 mb-4"),
                md=6
            )
        ]),

        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5(
                            "Beds vs. Rent (Sample Data)",
                            style={
                                "color": "#004085",
                                "fontWeight": "700",
                                "marginBottom": "12px",
                                "fontSize": "1.25rem"
                            }
                        ),
                        dcc.Graph(id='beds-rent-scatter-plot')
                    ])
                ], className="shadow-sm rounded-lg border-0 mb-4"),
                md=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5(
                            "Baths vs. Rent (Sample Data)",
                            style={
                                "color": "#004085",
                                "fontWeight": "700",
                                "marginBottom": "12px",
                                "fontSize": "1.25rem"
                            }
                        ),
                        dcc.Graph(id='baths-rent-scatter-plot')
                    ])
                ], className="shadow-sm rounded-lg border-0 mb-4"),
                md=6
            )
        ]),

        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5(
                            "Annual Rent Across UAE Cities (Sample Data)",
                            style={
                                "color": "#004085",
                                "fontWeight": "700",
                                "marginBottom": "12px",
                                "fontSize": "1.25rem"
                            }
                        ),
                        dcc.Graph(id="rent-map-plot", style={"height": "500px"})
                    ])
                ], className="shadow-sm rounded-lg border-0 mb-4"),
                width=12
            )
        ])
    ], className="pt-5 px-3")


def build_eda_page():
    sample_data = df.head(10)  

    return dbc.Container([
        html.Div(
            html.Img(
                src="https://www.fragomen.com/a/web/cQzR1LVbyYDVwAGqYp7TGi/8tDMMS/best-practices-for-maintaining-entities-in-the-uae-main-header.jpg",
                className="banner-image",
                style={
                    "width": "100%",
                    "height": "250px",
                    "objectFit": "cover",
                    "borderRadius": "8px",
                    "marginBottom": "30px",
                    "filter": "brightness(0.85)"
                }
            ),
            className="fade-in"
        ),
        html.H2("🔎 Exploratory Data Analysis (EDA)", className="text-center my-4"),
        dbc.Tabs([
            dbc.Tab(label="About Dataset", tab_id="tab-about", children=[
                dbc.Card([
                    dbc.CardBody([
                        html.P(
                            "The dataset contains rental property listings from major UAE cities including Abu Dhabi, Dubai, Sharjah, Ajman, Ras Al Khaimah, Umm Al Quwain, and Al Ain. "
                            "It includes 17 columns detailing features such as property type, rent amount, area size, furnishing status, and city location."
                        ),
                        html.P([
                            "Source: ",
                            html.A(
                                "Kaggle Dataset - Real Estate Goldmine Dubai UAE Rental Market",
                                href="https://www.kaggle.com/datasets/azharsaleem/real-estate-goldmine-dubai-uae-rental-market",
                                target="_blank",
                                style={"fontWeight": "bold"}
                            )
                        ]),
                        html.H5("Sample Data Preview", className="mt-4"),
                        dbc.Table.from_dataframe(sample_data, striped=True, bordered=True, hover=True, responsive=True),
                    ])
                ], className="mt-3")
            ]),
            dbc.Tab(label="EDA Steps", tab_id="tab-steps", children=[
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Steps Undertaken During EDA", className="mb-3"),
                        html.Ul([
                            html.Li("Data Cleaning: Handling missing values, removing duplicates, filtering outliers."),
                            html.Li("Feature Engineering: Created new features such as rent per square foot."),
                            html.Li("Encoding: Label encoding applied to categorical variables."),
                            html.Li("Distribution Analysis: Examined key features like rent, area, property types."),
                            html.Li("Correlation Analysis: Investigated relationships e.g. bedrooms vs rent."),
                            html.Li("Geographical Insights: Compared rental prices across different UAE cities."),
                            html.Li("Summarized findings to support rent prediction and market understanding."),
                        ])
                    ])
                ], className="mt-3")
            ]),
            dbc.Tab(label="Key Findings", tab_id="tab-findings", children=[
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Key Insights", className="mb-3"),
                        html.Ul([
                            html.Li("Apartments and villas are the majority of listings."),
                            html.Li("Dubai and Abu Dhabi lead the rental market in listings and prices."),
                            html.Li("More bedrooms/bathrooms generally means higher rent."),
                            html.Li("Rent per square foot varies significantly by property type and location."),
                            html.Li("Insights help improve market understanding and rent predictions."),
                        ])
                    ])
                ], className="mt-3")
            ]),
            dbc.Tab(label="Encoding Details", tab_id="tab-encoding", children=[
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Categorical Encoding Explanation", className="mb-3"),
                        html.P(
                            "Categorical columns were converted into numeric values using label encoding to prepare for modeling."
                        ),
                        html.P("Example mappings:"),
                        html.Ul([
                            html.Li("Type: Apartment=0, Penthouse=1, Villa=2, etc."),
                            html.Li("Furnishing: Unfurnished=0, Furnished=1"),
                            html.Li("City: Abu Dhabi=0, Ajman=1, Dubai=3, etc."),
                        ]),
                        html.P("This allows machine learning algorithms to interpret categorical data numerically.")
                    ])
                ], className="mt-3")
            ]),
            dbc.Tab(label="Resources", tab_id="tab-resources", children=[
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Additional Resources", className="mb-3"),
                        html.Ul([
                            html.Li(html.A("Kaggle Dataset", href="https://www.kaggle.com/datasets/azharsaleem/real-estate-goldmine-dubai-uae-rental-market", target="_blank")),
                            html.Li(html.A("Scikit-Learn LabelEncoder", href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html", target="_blank")),
                            html.Li(html.A("Scikit-Learn OneHotEncoder", href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html", target="_blank")),
                        ])
                    ])
                ], className="mt-3")
            ]),
        ], id="tabs", active_tab="tab-about"),
    ], className="pt-4 px-3")

# -----------------------
# Layout and Routing
# -----------------------
dash_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@dash_app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == "/dash/":
        page_content = build_home_page()
    elif pathname == "/dash/predict":
        page_content = build_prediction_page()
    elif pathname == "/dash/insights":
        page_content = build_insights_page()
    elif pathname == "/dash/eda":                
        page_content = build_eda_page()
    else:
        page_content = html.Div("404 - Page not found", className="text-danger p-5 fs-3")
    
    return html.Div([navbar, page_content])

# -----------------------
# Prediction callback
# -----------------------
@dash_app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State(name, "value") for name in feature_config.keys()]
)
def make_prediction(n_clicks, *values):
    if not n_clicks:
        return "Click 'Get Prediction' to get an estimate."
    if any(v is None for v in values):
        return dbc.Alert("Please fill in all fields to get a prediction.", color="warning", className="mt-3")
    if model is None or scaler is None or ohe is None:
        return dbc.Alert("Prediction model is not loaded. Please check server logs and ensure model files are in 'src' directory.", color="danger", className="mt-3")
    try:
        X = preprocess_inputs(values)
        pred_log = model.predict(X)[0]
        pred = np.expm1(pred_log)
        return html.Div([
            html.H4(f"Predicted Annual Rent: AED {pred:,.2f}", className="text-success mt-3"),
            html.Small("This is an estimated annual rent based on the provided features.", className="text-muted")
        ])
    except Exception as e:
        logger.error(f"Prediction error in Dash callback: {e}")
        return dbc.Alert(f"An error occurred during prediction: {str(e)}. Please check your inputs.", color="danger", className="mt-3")

# -----------------------
# Preprocess input before prediction
# -----------------------
def preprocess_inputs(values):
    area_sqm = values[2]
    area_sqft = area_sqm * 10.7639
    numeric_vals = np.array([values[0], values[1], area_sqft]).reshape(1, -1)
    cat_vals = np.array(values[3:]).reshape(1, -1)
    if scaler is None or ohe is None:
        raise ValueError("Scaler or OneHotEncoder not loaded for preprocessing.")
    scaled_num = scaler.transform(numeric_vals)
    encoded_cat = ohe.transform(cat_vals)
    X = np.hstack([scaled_num, encoded_cat])
    return X

# -----------------------
# Callbacks for Data Visualization Plots
# -----------------------
@dash_app.callback(
    Output('area-distribution-plot', 'figure'),
    Input('url', 'pathname')
)
def update_area_distribution_plot(pathname):
    np.random.seed(42)
    areas = np.random.normal(loc=150, scale=50, size=500)
    areas = areas[areas > 10].round(0)

    fig = go.Figure(data=[go.Histogram(x=areas, nbinsx=30, marker_color=BOOTSTRAP_COLORS['primary'], opacity=0.8)])
    fig.update_layout(
        title_text='Distribution of Area in Square Meters',
        xaxis_title_text='Area (sqm)',
        yaxis_title_text='Count',
        bargap=0.05,
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=20),
        transition=dict(duration=500, easing="cubic-in-out"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=BOOTSTRAP_COLORS['dark'])
    )
    return fig

@dash_app.callback(
    Output('type-distribution-plot', 'figure'),
    Input('url', 'pathname')
)
def update_type_distribution_plot(pathname):
    np.random.seed(42)
    types_numeric = np.random.choice(list(category_mappings["Type"].keys()), size=300,
                                     p=[0.4, 0.1, 0.2, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025])
    type_labels = [category_mappings["Type"][t] for t in types_numeric]
    type_df = pd.DataFrame({'Type': type_labels})
    type_counts = type_df['Type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']

    fig = px.bar(type_counts, x='Type', y='Count',
                 color='Type',
                 color_discrete_sequence=px.colors.qualitative.Plotly
                )
    fig.update_layout(
        title_text='Distribution of Property Types',
        xaxis_title_text='Property Type',
        yaxis_title_text='Count',
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=20),
        xaxis={'categoryorder':'total descending'},
        transition=dict(duration=500, easing="cubic-in-out"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=BOOTSTRAP_COLORS['dark'])
    )
    return fig

@dash_app.callback(
    Output('city-distribution-plot', 'figure'),
    Input('url', 'pathname')
)
def update_city_distribution_plot(pathname):
    np.random.seed(42)
    cities_numeric = np.random.choice(list(category_mappings["City"].keys()), size=400,
                                     p=[0.3, 0.1, 0.05, 0.35, 0.05, 0.05, 0.05, 0.05])
    city_labels = [category_mappings["City"][c] for c in cities_numeric]
    city_df = pd.DataFrame({'City': city_labels})
    city_counts = city_df['City'].value_counts().reset_index()
    city_counts.columns = ['City', 'Count']

    fig = px.bar(city_counts, x='City', y='Count',
                 color='City',
                 color_discrete_sequence=px.colors.qualitative.Pastel
                )
    fig.update_layout(
        title_text='Distribution of Cities',
        xaxis_title_text='City',
        yaxis_title_text='Count',
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=20),
        xaxis={'categoryorder':'total descending'},
        transition=dict(duration=500, easing="cubic-in-out"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=BOOTSTRAP_COLORS['dark'])
    )
    return fig

@dash_app.callback(
    Output('rent-by-type-violin-plot', 'figure'),
    Input('url', 'pathname')
)
def update_rent_by_type_violin_plot(pathname):
    np.random.seed(42)
    data_for_violin = []
    violin_colors = px.colors.qualitative.D3

    for i, (type_idx, type_label) in enumerate(category_mappings["Type"].items()):
        if type_label == "Apartment":
            rent = np.random.normal(loc=100000, scale=30000, size=50)
        elif type_label == "Villa":
            rent = np.random.normal(loc=250000, scale=70000, size=20)
        elif type_label == "Penthouse":
            rent = np.random.normal(loc=350000, scale=80000, size=15)
        else:
            rent = np.random.normal(loc=70000, scale=20000, size=10)

        rent[rent < 10000] = 10000

        data_for_violin.append(
            go.Violin(
                y=rent,
                name=type_label,
                box_visible=True,
                meanline_visible=True,
                jitter=0.05,
                scalemode='count',
                line_color=violin_colors[i % len(violin_colors)],
                fillcolor=violin_colors[i % len(violin_colors)],
                opacity=0.6
            )
        )

    fig = go.Figure(data=data_for_violin)
    fig.update_layout(
        title_text='Annual Rent Distribution by Property Type',
        xaxis_title_text='Property Type',
        yaxis_title_text='Annual Rent (AED)',
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=20),
        showlegend=False,
        transition=dict(duration=500, easing="cubic-in-out"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=BOOTSTRAP_COLORS['dark'])
    )
    return fig

@dash_app.callback(
    Output('beds-rent-scatter-plot', 'figure'),
    Input('url', 'pathname')
)
def update_beds_rent_scatter_plot(pathname):
    np.random.seed(42)
    beds = np.random.randint(1, 6, size=200)
    rent = 50000 + beds * 20000 + np.random.normal(0, 15000, size=200)
    rent[rent < 10000] = 10000

    df_scatter = pd.DataFrame({'Beds': beds, 'Rent': rent})

    fig = go.Figure(data=[go.Scatter(
        x=df_scatter['Beds'],
        y=df_scatter['Rent'],
        mode='markers',
        marker=dict(
            color=BOOTSTRAP_COLORS['info'],
            size=10,
            opacity=0.7,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        hoverinfo='text',
        hovertext=[f'Beds: {b}<br>Rent: AED {r:,.2f}' for b, r in zip(df_scatter['Beds'], df_scatter['Rent'])]
    )])
    fig.update_layout(
        title_text='Beds vs. Annual Rent (Sample Data)',
        xaxis_title_text='Number of Beds',
        yaxis_title_text='Annual Rent (AED)',
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=20),
        transition=dict(duration=500, easing="cubic-in-out"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=BOOTSTRAP_COLORS['dark'])
    )
    return fig

@dash_app.callback(
    Output('baths-rent-scatter-plot', 'figure'),
    Input('url', 'pathname')
)
def update_baths_rent_scatter_plot(pathname):
    np.random.seed(42)
    baths = np.random.randint(1, 5, size=180)
    rent = 60000 + baths * 15000 + np.random.normal(0, 10000, size=180)
    rent[rent < 10000] = 10000

    df_scatter = pd.DataFrame({'Baths': baths, 'Rent': rent})

    fig = go.Figure(data=[go.Scatter(
        x=df_scatter['Baths'],
        y=df_scatter['Rent'],
        mode='markers',
        marker=dict(
            color=BOOTSTRAP_COLORS['success'],
            size=10,
            opacity=0.7,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        hoverinfo='text',
        hovertext=[f'Baths: {b}<br>Rent: AED {r:,.2f}' for b, r in zip(df_scatter['Baths'], df_scatter['Rent'])]
    )])
    fig.update_layout(
        title_text='Baths vs. Annual Rent (Sample Data)',
        xaxis_title_text='Number of Baths',
        yaxis_title_text='Annual Rent (AED)',
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=20),
        transition=dict(duration=500, easing="cubic-in-out"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=BOOTSTRAP_COLORS['dark'])
    )
    return fig

@dash_app.callback(
    Output('rent-map-plot', 'figure'),
    Input('url', 'pathname')
)
def update_rent_map_plot(pathname):
    cities = list(city_coordinates.keys())
    lats = [city_coordinates[city]["lat"] for city in cities]
    lons = [city_coordinates[city]["lon"] for city in cities]
    rents = [city_coordinates[city]["predicted_rent"] for city in cities]

    fig = go.Figure(
        go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=[r / 3000 for r in rents],
                color=rents,
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Annual Rent (AED)", thickness=15, title_font_color=BOOTSTRAP_COLORS['dark']),
                opacity=0.9
            ),
            text=[f"City: {city}<br>Estimated Rent: AED {rent:,.2f}" for city, rent in zip(cities, rents)],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=6,
        mapbox_center={"lat": 24.5, "lon": 55.0},
        margin={"r":0,"t":40,"l":0,"b":0},
        title_text="Annual Rent Distribution Across UAE Cities (Sample Data)",
        transition=dict(duration=500, easing="cubic-in-out"),
        font=dict(color=BOOTSTRAP_COLORS['dark'])
    )
    return fig

# -----------------------
# Flask API endpoint for prediction
# -----------------------
@server.route("/predict", methods=["POST"])
def predict_api():
    if model is None or scaler is None or ohe is None:
        return jsonify({"error": "Model or preprocessors not loaded"}), 503
    try:
        data = request.json
        feature_values = [
            data.get("Beds"),
            data.get("Baths"),
            data.get("Area in square meters"),
            data.get("Type"),
            data.get("Furnishing"),
            data.get("City")
        ]
        if any(v is None for v in feature_values):
            return jsonify({"error": "Missing one or more required features"}), 400
        X = preprocess_inputs(feature_values)
        pred_log = model.predict(X)[0]
        pred = np.expm1(pred_log)
        return jsonify({"predicted_rent": float(pred)})
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------
# Other Flask routes
# -----------------------
@server.route("/")
def index():
    return redirect("/dash/")

@server.route("/health")
def health():
    return jsonify({
        "status": "ok" if model is not None and scaler is not None and ohe is not None else "error",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoder_loaded": ohe is not None
    })

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    if not os.path.exists("src/xgb_rent_model_optimized.pkl"):
        logger.warning("Model file 'src/xgb_rent_model_optimized.pkl' not found.")
    if not os.path.exists("src/scaler.pkl"):
        logger.warning("Scaler file 'src/scaler.pkl' not found.")
    if not os.path.exists("src/encoder.pkl"):
        logger.warning("Encoder file 'src/encoder.pkl' not found.")

    logger.info("Starting Flask + Dash Rent Prediction App...")
    server.run(host="0.0.0.0", port=8000, debug=True)