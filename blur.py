import streamlit as st
import pickle
import numpy as np
import base64

# Set up the sidebar and user inputs
st.sidebar.title("Crop Yield Prediction")

# Average Rain Fall
Average_Rain_Fall_mm_per_year = st.sidebar.number_input("Average Rain Fall (mm per year)", min_value=0.0, max_value=3000.0, value=1000.0)

# Pesticides Tonnes
Pesticides_Tonnes = st.sidebar.number_input("Pesticides (Tonnes)", min_value=0.0, max_value=100000.0, value=500.0)

# Avg Temp
Avg_Temp = st.sidebar.number_input("Average Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)

# Item
Item = st.sidebar.selectbox("Crop Type", 
                            ["Cassava", "Maize", "Potatoes", "Rice, paddy", "Sorghum", 
                             "Soybeans", "Sweet potatoes", "Wheat"])

# One hot encoding for Item
Item_dict = {'Cassava': 0, 'Maize': 1, 'Potatoes': 2, 'Rice, paddy': 3, 'Sorghum': 4,
             'Soybeans': 5, 'Sweet potatoes': 6, 'Wheat': 7}

Item_encoded = [0] * 8
Item_encoded[Item_dict[Item]] = 1

# Load the saved model
predict_button = st.sidebar.button("Predict Crop Yield")
back_button = st.sidebar.button("Make Another Prediction")

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

if predict_button:
    # Create the feature vector
    Features = [Average_Rain_Fall_mm_per_year, Pesticides_Tonnes, Avg_Temp] + Item_encoded
    
    # Load model
    Model = pickle.load(open('ind_yield.pkl', 'rb'))
    
    # Make prediction
    prediction = Model.predict([Features])
    prediction_value = float(prediction[0])  # Ensure the prediction is a float
    
    gif = get_img_as_base64("crop.gif")
    page_bg_gif = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/gif;base64,{gif}");
    background-size: cover;
    background-position: center;
    margin-top: -100px;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    .animated-text {{
        font-size: 3rem;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        animation: colorchange 2s infinite;
        text-align: center;
    }}
    @keyframes colorchange {{
        0% {{ color: #FF5722; }}
        25% {{ color: #4CAF50; }}
        50% {{ color: #FFC107; }}
        75% {{ color: #00BCD4; }}
        100% {{ color: #FF5722; }}
    }}
    .blur-box {{
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 20px;
        margin-top: 50px;
        text-align: center;
    }}
    </style>
    """
    st.markdown(page_bg_gif, unsafe_allow_html=True)
    st.markdown(f"<div class='blur-box'><h2 class='animated-text'>Predicted Crop Yield: {prediction_value:.2f} kg</h2></div>", unsafe_allow_html=True)
    
    if back_button:
        st.experimental_rerun()
else:
    st.markdown(f"<h2 class='animated-text'>Crop Yield Prediction</h2>", unsafe_allow_html=True)
    img = get_img_as_base64("crop.jpg")
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/jpeg;base64,{img}");
    background-size: cover;
    background-position: center;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    .animated-text {{
        font-size: 3rem;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        animation: colorchange 2s infinite;
        text-align: center;
    }}
    .blur-box {{
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 20px;
        margin-top: 50px;
        text-align: center;
    }}
    @keyframes colorchange {{
        0% {{ color: #FF5722; }}
        25% {{ color: #4CAF50; }}
        50% {{ color: #FFC107; }}
        75% {{ color: #00BCD4; }}
        100% {{ color: #FF5722; }}
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
