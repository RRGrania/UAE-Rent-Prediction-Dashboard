import os
import joblib
import numpy as np
import logging
from flask import Flask, request, jsonify, send_from_directory, redirect
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd # Added for map plot data handling

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# Flask app initialization
# -----------------------
server = Flask(__name__)

# -----------------------
# Dash app initialization
# Using a different theme for a more professional look
# -----------------------
dash_app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname="/dash/",
    external_stylesheets=[dbc.themes.CERULEAN, dbc.icons.FONT_AWESOME], # Changed theme to CERULEAN
    assets_folder="assets",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}] # Responsive meta tag
)

# -----------------------
# Feature configuration for the input fields
# -----------------------
feature_config = {
    "Beds": {"min": 0, "max": 10, "default": 2, "icon": "fa-solid fa-bed"},
    "Baths": {"min": 0, "max": 10, "default": 2, "icon": "fa-solid fa-bath"},
    "Area in square meters": {"min": 10, "max": 1000, "default": 75, "icon": "fa-solid fa-ruler-combined"},
    "Type": {"min": 0, "max": 8, "default": 0, "icon": "fa-solid fa-building"},
    "Furnishing": {"min": 0, "max": 1, "default": 0, "icon": "fa-solid fa-couch"},
    "City": {"min": 0, "max": 7, "default": 0, "icon": "fa-solid fa-city"}
}

# List of categorical features that will use dropdowns
categorical_features = ["Type", "Furnishing", "City"]

# -----------------------
# Categorical mappings for dropdowns
# -----------------------
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

# -----------------------
# Geographical data for UAE cities (for map plot)
# Note: These are approximate coordinates. Replace with your specific data if needed.
# The 'predicted_rent' values are synthetic for demonstration purposes only.
# -----------------------
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
# Helper functions to create input cards
# -----------------------
def create_input_card(name, config):
    """
    Creates a numeric input card with an icon.
    """
    return dbc.Card(
        dbc.CardBody([
            html.Label(name, className="form-label mb-2 fw-bold"),
            dbc.InputGroup(
                [
                    dbc.InputGroupText(html.I(className=config["icon"])),
                    dbc.Input(
                        id=name,
                        type="number",
                        min=config["min"],
                        max=config["max"],
                        step=config.get("step", 1),
                        value=config["default"],
                        placeholder=f"Enter {name}",
                    )
                ],
                className="shadow-sm rounded-lg"
            )
        ]),
        className="mb-3 border-0" # No border, use shadow from InputGroup
    )

def create_dropdown_card(name, options_dict, default_idx, icon):
    """
    Creates a dropdown card with an icon.
    """
    options = [{"label": label, "value": idx} for idx, label in options_dict.items()]
    return dbc.Card(
        dbc.CardBody([
            html.Label(name, className="form-label mb-2 fw-bold"),
            dbc.InputGroup(
                [
                    dbc.InputGroupText(html.I(className=icon)),
                    dcc.Dropdown(
                        id=name,
                        options=options,
                        value=default_idx,
                        clearable=False,
                        searchable=True,
                        className="form-select border-0" # Remove dropdown border, use input group border
                    )
                ],
                className="shadow-sm rounded-lg"
            )
        ]),
        className="mb-3 border-0"
    )

# -----------------------
# Build Dash layout with Navbar, Sidebar, Inputs, and Plots
# -----------------------
def build_layout():
    """
    Constructs the main layout of the Dash application with a Navbar,
    Offcanvas sidebar, prediction inputs, and data visualization plots.
    """
    prediction_inputs = []
    for name, cfg in feature_config.items():
        if name in categorical_features:
            prediction_inputs.append(create_dropdown_card(name, category_mappings[name], cfg["default"], cfg["icon"]))
        else:
            prediction_inputs.append(create_input_card(name, cfg))
    
    col1_inputs = prediction_inputs[0::2]
    col2_inputs = prediction_inputs[1::2]

    # Navbar component
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/dash/")),
            dbc.NavItem(dbc.NavLink("Predict", href="#prediction-section")),
            dbc.NavItem(dbc.NavLink("Data Insights", href="#data-insights-section")),
            dbc.Button(
                html.I(className="fa-solid fa-bars"), # Hamburger icon for offcanvas toggle
                id="open-offcanvas",
                n_clicks=0,
                className="ms-3 d-lg-none", # Show only on small screens
                color="light"
            ),
        ],
        brand="UAE Rent Predictor",
        brand_href="/dash/",
        color="primary", # Primary color from theme
        dark=True,
        className="shadow-sm fixed-top", # Fixed at top, with shadow
        fluid=True # Full width
    )

    # Offcanvas (sidebar) component
    offcanvas = dbc.Offcanvas(
        children=[
            html.P("Quick Navigation", className="lead text-muted"),
            dbc.Nav(
                [
                    dbc.NavLink("Home", href="/dash/", active="exact"),
                    dbc.NavLink("Prediction", href="#prediction-section", active="exact"),
                    dbc.NavLink("Data Insights", href="#data-insights-section", active="exact"),
                ],
                vertical=True,
                pills=True, # Highlight active link
            ),
            html.Hr(className="my-3"),
            html.Div([
                html.P("About", className="mb-1"),
                html.Small("Powered by Dash & Plotly", className="text-muted d-block"),
                html.Small("Model Version 1.0", className="text-muted")
            ], className="px-3")
        ],
        id="offcanvas",
        title="Menu",
        is_open=False,
        placement="start", # From left
        scrollable=True,
        backdrop=True
    )

    return html.Div(
        className="bg-light min-vh-100", # Light background, full height
        children=[
            dcc.Location(id="url", refresh=False),
            navbar, # Add the navbar
            offcanvas, # Add the offcanvas sidebar
            
            # Main content area - using a container with top margin for navbar
            dbc.Container(
                className="py-5 mt-5", # Margin-top to clear the fixed navbar
                children=[
                    # Prediction Section
                    html.Div(id="prediction-section", className="mb-5 pt-4"), # Anchor for prediction section
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H2("Predict Annual Rent", className="text-center text-primary mb-0")),
                            dbc.CardBody([
                                html.P("Enter the property details to get an estimated annual rent in AED.", 
                                       className="text-center text-muted mb-4"),
                                dbc.Row(
                                    className="g-4",
                                    children=[
                                        dbc.Col(col1_inputs, md=6),
                                        dbc.Col(col2_inputs, md=6)
                                    ]
                                ),
                                dbc.Row(
                                    className="mt-4",
                                    children=[
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    [html.I(className="fa-solid fa-magnifying-glass me-2"), "Get Prediction"],
                                                    id="predict-btn",
                                                    color="primary",
                                                    size="lg",
                                                    className="d-block mx-auto mb-3 w-75"
                                                ),
                                                dcc.Loading( # Add loading component
                                                    id="loading-output",
                                                    type="circle",
                                                    children=html.Div(
                                                        id="prediction-output",
                                                        className="text-center fs-4 fw-bold text-dark mt-3"
                                                    )
                                                )
                                            ],
                                            width=12,
                                            className="d-flex flex-column align-items-center"
                                        )
                                    ]
                                )
                            ]),
                            dbc.CardFooter(
                                html.Small("Prediction results are estimates and may vary.", className="text-muted")
                            )
                        ],
                        className="shadow-lg rounded-lg border-0" # More prominent card
                    ),

                    # Data Visualization Section
                    html.Div(id="data-insights-section", className="mt-5 pt-4"), # Anchor for data insights
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H2("Data Distribution Insights", className="text-center text-secondary mb-0")),
                            dbc.CardBody([
                                html.P("Explore the distributions and relationships within the dataset.",
                                       className="text-center text-muted mb-4"),
                                dbc.Row(
                                    className="g-4",
                                    children=[
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("Area Distribution (sqm)", className="card-title"),
                                                    dcc.Graph(id='area-distribution-plot')
                                                ]),
                                                className="shadow-sm rounded-lg border-0"
                                            ),
                                            md=6
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("Property Type Distribution", className="card-title"),
                                                    dcc.Graph(id='type-distribution-plot')
                                                ]),
                                                className="shadow-sm rounded-lg border-0"
                                            ),
                                            md=6
                                        )
                                    ]
                                ),
                                dbc.Row(
                                    className="g-4 mt-4", # Margin top for second row of plots
                                    children=[
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("City Distribution", className="card-title"),
                                                    dcc.Graph(id='city-distribution-plot')
                                                ]),
                                                className="shadow-sm rounded-lg border-0"
                                            ),
                                            md=6
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("Rent Distribution by Property Type", className="card-title"),
                                                    dcc.Graph(id='rent-by-type-violin-plot') # New plot
                                                ]),
                                                className="shadow-sm rounded-lg border-0"
                                            ),
                                            md=6
                                        )
                                    ]
                                ),
                                # New Row for additional plots
                                dbc.Row(
                                    className="g-4 mt-4",
                                    children=[
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("Beds vs. Rent (Sample Data)", className="card-title"),
                                                    dcc.Graph(id='beds-rent-scatter-plot')
                                                ]),
                                                className="shadow-sm rounded-lg border-0"
                                            ),
                                            md=6
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("Baths vs. Rent (Sample Data)", className="card-title"),
                                                    dcc.Graph(id='baths-rent-scatter-plot') # New plot
                                                ]),
                                                className="shadow-sm rounded-lg border-0"
                                            ),
                                            md=6
                                        )
                                    ]
                                ),
                                # New Row for Map Plot
                                dbc.Row(
                                    className="g-4 mt-4",
                                    children=[
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H5("Rent Distribution Map (Sample Data)", className="card-title"),
                                                    dcc.Graph(id="rent-map-plot", style={"height": "500px"}) # Added map plot
                                                ]),
                                                className="shadow-sm rounded-lg border-0"
                                            ),
                                            width=12 # Map plot takes full width
                                        )
                                    ]
                                )
                            ]),
                            dbc.CardFooter(
                                html.Small("Plots are generated from sample data for demonstration.", className="text-muted")
                            )
                        ],
                        className="shadow-lg rounded-lg border-0" # More prominent card
                    )
                ]
            )
        ]
    )

# Assign the built layout to the Dash app
dash_app.layout = build_layout()

# -----------------------
# Callback to toggle offcanvas (sidebar)
# -----------------------
@dash_app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    State("offcanvas", "is_open"),
)
def toggle_offcanvas(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# -----------------------
# Preprocess input before prediction
# -----------------------
def preprocess_inputs(values):
    """
    Preprocesses the input values from the Dash form for model prediction.
    Assumes values order: Beds, Baths, Area_in_sqm, Type, Furnishing, City.
    Converts Area from square meters to square feet.
    Applies scaling to numeric features and one-hot encoding to categorical features.
    """
    area_sqm = values[2]
    area_sqft = area_sqm * 10.7639
    
    numeric_vals = np.array([values[0], values[1], area_sqft]).reshape(1, -1)
    cat_vals = np.array(values[3:]).reshape(1, -1)

    # Check if scaler and ohe are loaded
    if scaler is None or ohe is None:
        raise ValueError("Scaler or OneHotEncoder not loaded for preprocessing.")

    scaled_num = scaler.transform(numeric_vals)
    encoded_cat = ohe.transform(cat_vals)

    X = np.hstack([scaled_num, encoded_cat])
    return X


# -----------------------
# Dash callback for prediction
# -----------------------
@dash_app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State(name, "value") for name in feature_config.keys()]
)
def make_prediction(n_clicks, *values):
    """
    Callback function to handle the prediction when the button is clicked.
    """
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
# Callbacks for Data Visualization Plots
# -----------------------

# Callback for Area Distribution Plot
@dash_app.callback(
    Output('area-distribution-plot', 'figure'),
    Input('url', 'pathname') # Trigger on page load
)
def update_area_distribution_plot(pathname):
    np.random.seed(42)
    areas = np.random.normal(loc=150, scale=50, size=500) # Slightly larger range for areas
    areas = areas[areas > 10].round(0)

    fig = go.Figure(data=[go.Histogram(x=areas, nbinsx=30, marker_color='#636EFA', opacity=0.8)])
    fig.update_layout(
        title_text='Distribution of Area in Square Meters',
        xaxis_title_text='Area (sqm)',
        yaxis_title_text='Count',
        bargap=0.05,
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=20)
    )
    return fig

# Callback for Property Type Distribution Plot
@dash_app.callback(
    Output('type-distribution-plot', 'figure'),
    Input('url', 'pathname')
)
def update_type_distribution_plot(pathname):
    np.random.seed(42)
    types_numeric = np.random.choice(list(category_mappings["Type"].keys()), size=300, 
                                     p=[0.4, 0.1, 0.2, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025])
    type_labels = [category_mappings["Type"][t] for t in types_numeric]
    type_df = pd.DataFrame(type_labels, columns=['Type'])
    type_counts = type_df['Type'].value_counts().sort_values(ascending=False)

    fig = go.Figure(data=[go.Bar(x=type_counts.index, y=type_counts.values, marker_color='#EF553B')])
    fig.update_layout(
        title_text='Distribution of Property Types',
        xaxis_title_text='Property Type',
        yaxis_title_text='Count',
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=20),
        xaxis={'categoryorder':'total descending'} # Order bars by count
    )
    return fig

# Callback for City Distribution Plot
@dash_app.callback(
    Output('city-distribution-plot', 'figure'),
    Input('url', 'pathname')
)
def update_city_distribution_plot(pathname):
    np.random.seed(42)
    cities_numeric = np.random.choice(list(category_mappings["City"].keys()), size=400, 
                                      p=[0.3, 0.1, 0.05, 0.35, 0.05, 0.05, 0.05, 0.05])
    city_labels = [category_mappings["City"][c] for c in cities_numeric]
    city_df = pd.DataFrame(city_labels, columns=['City'])
    city_counts = city_df['City'].value_counts().sort_values(ascending=False)

    fig = go.Figure(data=[go.Bar(x=city_counts.index, y=city_counts.values, marker_color='#00CC96')])
    fig.update_layout(
        title_text='Distribution of Cities',
        xaxis_title_text='City',
        yaxis_title_text='Count',
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=20),
        xaxis={'categoryorder':'total descending'}
    )
    return fig

# New Plot: Rent Distribution by Property Type (Violin Plot)
@dash_app.callback(
    Output('rent-by-type-violin-plot', 'figure'),
    Input('url', 'pathname')
)
def update_rent_by_type_violin_plot(pathname):
    np.random.seed(42)
    # Generate synthetic data mimicking different rent distributions for property types
    data_for_violin = []
    for type_idx, type_label in category_mappings["Type"].items():
        # Simulate different rent ranges for different types
        if type_label == "Apartment":
            rent = np.random.normal(loc=100000, scale=30000, size=50)
        elif type_label == "Villa":
            rent = np.random.normal(loc=250000, scale=70000, size=20)
        elif type_label == "Penthouse":
            rent = np.random.normal(loc=350000, scale=80000, size=15)
        else: # Other types
            rent = np.random.normal(loc=70000, scale=20000, size=10)
        
        rent[rent < 10000] = 10000 # Ensure rent is positive

        data_for_violin.append(
            go.Violin(
                y=rent,
                name=type_label,
                box_visible=True,
                meanline_visible=True,
                jitter=0.05,
                scalemode='count' # violin width proportional to the number of points in that violin
            )
        )

    fig = go.Figure(data=data_for_violin)
    fig.update_layout(
        title_text='Annual Rent Distribution by Property Type',
        xaxis_title_text='Property Type',
        yaxis_title_text='Annual Rent (AED)',
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=20),
        showlegend=False
    )
    return fig


# Callback for Beds vs. Rent Scatter Plot
@dash_app.callback(
    Output('beds-rent-scatter-plot', 'figure'),
    Input('url', 'pathname')
)
def update_beds_rent_scatter_plot(pathname):
    np.random.seed(42)
    beds = np.random.randint(1, 6, size=200) # 1 to 5 beds
    rent = 50000 + beds * 20000 + np.random.normal(0, 15000, size=200)
    rent[rent < 10000] = 10000
    
    df_scatter = pd.DataFrame({'Beds': beds, 'Rent': rent})

    fig = go.Figure(data=[go.Scatter(
        x=df_scatter['Beds'], 
        y=df_scatter['Rent'], 
        mode='markers', 
        marker=dict(
            color='#FF6692', 
            size=8,
            opacity=0.7,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        hoverinfo='text', # Show custom text on hover
        hovertext=[f'Beds: {b}<br>Rent: AED {r:,.2f}' for b, r in zip(df_scatter['Beds'], df_scatter['Rent'])]
    )])
    fig.update_layout(
        title_text='Beds vs. Annual Rent (Sample Data)',
        xaxis_title_text='Number of Beds',
        yaxis_title_text='Annual Rent (AED)',
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=20)
    )
    return fig

# New Plot: Baths vs. Rent Scatter Plot
@dash_app.callback(
    Output('baths-rent-scatter-plot', 'figure'),
    Input('url', 'pathname')
)
def update_baths_rent_scatter_plot(pathname):
    np.random.seed(42)
    baths = np.random.randint(1, 5, size=180) # 1 to 4 baths
    rent = 60000 + baths * 15000 + np.random.normal(0, 10000, size=180)
    rent[rent < 10000] = 10000
    
    df_scatter = pd.DataFrame({'Baths': baths, 'Rent': rent})

    fig = go.Figure(data=[go.Scatter(
        x=df_scatter['Baths'], 
        y=df_scatter['Rent'], 
        mode='markers', 
        marker=dict(
            color='#7FFFD4', # Aquamarine color
            size=8,
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
        margin=dict(l=40, r=20, t=40, b=20)
    )
    return fig

# New Callback for Rent Map Plot
@dash_app.callback(
    Output('rent-map-plot', 'figure'),
    Input('url', 'pathname') # Trigger on page load
)
def update_rent_map_plot(pathname):
    # Prepare data for the map plot
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
                size=[r / 5000 for r in rents], # Scale marker size by rent for visibility
                color=rents,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Annual Rent (AED)"),
                opacity=0.8
            ),
            text=[f"City: {city}<br>Estimated Rent: AED {rent:,.2f}" for city, rent in zip(cities, rents)],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        mapbox_style="open-street-map", # You can try other styles like "carto-positron", "stamen-terrain"
        mapbox_zoom=6,
        mapbox_center={"lat": 24.5, "lon": 55.0}, # Centered around UAE
        margin={"r":0,"t":40,"l":0,"b":0},
        title_text="Annual Rent Distribution Across UAE Cities (Sample Data)",
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
        # The API expects raw feature values, then preprocess_inputs handles them
        # Convert map values back to original feature names for preprocessing
        feature_values = [
            data.get("Beds"),
            data.get("Baths"),
            data.get("Area in square meters"),
            data.get("Type"), # This will be the numeric index
            data.get("Furnishing"), # This will be the numeric index
            data.get("City") # This will be the numeric index
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

@server.route("/assets/<path:path>")
def static_files(path):
    return send_from_directory("assets", path)

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    if not os.path.exists("src/xgb_rent_model_optimized.pkl"):
        logger.warning("Model file 'src/xgb_rent_model_optimized.pkl' not found.")
    if not os.path.exists("src/scaler.pkl"):
        logger.warning("Scaler file 'src/scaler.pkl' not found.")
    if not os.path.exists("src/encoder.pkl"):
        logger.warning("Encoder file 'src/encoder_new.pkl' not found.") # Corrected filename
    
    logger.info("Starting Flask + Dash Rent Prediction App...")
    server.run(host="0.0.0.0", port=8000, debug=True)