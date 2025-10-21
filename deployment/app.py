import streamlit as st
import requests

st.title("Bank Marketing Prediction")


# Feature configuration: type and choices
feature_config = {
    "age": {"type": "int", "min": 18, "max": 100},
    "job": {"type": "cat", "choices": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"]},
    "marital": {"type": "cat", "choices": ["married", "single", "divorced"]},
    "education": {"type": "cat", "choices": ["primary", "secondary", "tertiary", "unknown"]},
    "default": {"type": "cat", "choices": ["yes", "no"]},
    "balance": {"type": "int", "min": -10000, "max": 100000},
    "housing": {"type": "cat", "choices": ["yes", "no"]},
    "loan": {"type": "cat", "choices": ["yes", "no"]},
    "contact": {"type": "cat", "choices": ["cellular", "telephone", "unknown"]},
    "day": {"type": "int", "min": 1, "max": 31},
    "month": {"type": "cat", "choices": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]},
    "duration": {"type": "int", "min": 0, "max": 5000},
    "campaign": {"type": "int", "min": 1, "max": 50},
    "pdays": {"type": "int", "min": -1, "max": 999},
    "previous": {"type": "int", "min": 0, "max": 100},
    "poutcome": {"type": "cat", "choices": ["success", "failure", "other", "unknown"]}
}

st.markdown("""
### Enter Customer Information
""")

user_input = {}
for feature, config in feature_config.items():
    if config["type"] == "cat":
        value = st.selectbox(f"{feature.capitalize()}", config["choices"])
    elif config["type"] == "int":
        value = st.number_input(f"{feature.capitalize()}", min_value=config["min"], max_value=config["max"], value=config["min"])
    else:
        value = st.text_input(f"Enter {feature}")
    user_input[feature] = value

if st.button("Predict"):
    # Convert input types
    input_data = {}
    try:
        for feature, config in feature_config.items():
            val = user_input[feature]
            if config["type"] == "int":
                input_data[feature] = int(val)
            else:
                input_data[feature] = val
        response = requests.post("http://localhost:8000/predict", json={"data": input_data})

        if response.status_code == 200:
            
            st.success(f"Prediction: {'Subscribe' if response.json()['prediction'] == 1 else 'No Subscribe'}")
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Input error: {e}")
