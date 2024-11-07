import tkinter as tk
from tkinter import ttk
import time
import serial
import re
import pandas as pd
import numpy as np
import joblib
import threading
from functools import partial

# Global variable to store the output
output_result = ""
serial_data = []
selected_language = "English"

translations = {
    "English": {
        "language_select": "Select Language",
        "initialize": "SAGE Initializing",
        "read_carefully": "Please read the following carefully\nbefore pressing the OK button.",
        "instructions": """    • Fill the container up to half inch from the opening\n       and pour 10 drops (10mL) of distilled water
    • Insert the probe until almost all the metallic part\n       is submerged in soil
    • Make sure that the probe is steady while collecting\n       the data""",
        "insert_probe": "Please insert the probe into the soil.\nThen click the NEXT button\nafter inserting the probe.",
        "press_button": "Press the button to initialize recommendation.",
        "select_crop": "Please\nselect\na crop.",
        "scroll_here": "Scroll here",
        "no_selection": "Please select a crop before confirming.",
        "collecting_data": "Collecting data for {}",
        "insufficient_data": "Insufficient/Incorrect data.\nTry again.",
        "recommended_fertilizer": "Recommended fertilizer for {}:",
        "test_again": "Do you want to test again?",
        "shutdown_confirm": "Are you sure you want to shut down the device?",
        "shutting_down": "SHUTTING DOWN",
        "next": "NEXT",
        "start": "Start",
        "confirm": "CONFIRM",
        "crops": "Crops",
        "retry": "RETRY",
        "yes": "YES",
        "no": "NO",
        "warning_text": "Warning",
        "instructions_text": "Instructions",
    },
    "Filipino": {
        "language_select": "Pumili ng Wika",
        "initialize": "Nagsisimula ang SAGE",
        "read_carefully": "Mangyaring basahing mabuti ang\nsumusunod bago pindutin\nang OK.",
        "instructions": """    • Punuin and lalagayan hanggang bago mag-kalahating pulgada\n       galing sa ibabaw at patakan ng 10 mL ng distilled water
    • Ipasok ang probe hanggang halos walang nakikitang metallic\n       na bahagi
    • Siguraduhing hindi gumagalaw ang probe habang kinukuha\n       ang data""",
        "insert_probe": "Mangyaring ipasok ang probe sa lupa.\nPagkatapos, i-click ang NEXT button",
        "press_button": "Pindutin ang button para simulan ang rekomendasyon.",
        "select_crop": "Mangyaring\npumili ng\npananim.",
        "scroll_here": "I-iscroll ito",
        "no_selection": "Mangyaring pumili ng pananim bago kumpirmahin.",
        "collecting_data": "Kumukuha ng data para sa {}",
        "insufficient_data": "Hindi sapat/Mali ang data.\nSubukang muli.",
        "recommended_fertilizer": "Inirerekomendang pataba para sa {}:",
        "test_again": "Gusto mo bang subukan muli?",
        "shutdown_confirm": "Sigurado ka bang gusto mong patayin ang device?",
        "shutting_down": "NAGSASARA NA",
        "next": "SUSUNOD",
        "start": "Simulan",
        "confirm": "KUMPIRMAHIN",
        "crops": "Mga Pananim",
        "retry": "SUBUKAN MULI",
        "yes": "OO",
        "no": "HINDI",
        "warning_text": "Babala",
        "instructions_text": "Mga Tagubilin",
    }
}

def create_language_selection():
    lang_frame = ttk.Frame(root)
    lang_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=1, relheight=1)

    label = tk.Label(lang_frame, text="Select Language / Pumili ng Wika", font=("Tahoma", 40, "bold"))
    label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    english_button = tk.Button(lang_frame, text="English", bg='#738678', font=("Copperplate Gothic Bold", 30), 
                               command=partial(set_language, "English"))
    english_button.place(relx=0.4, rely=0.5, width=250, height=100, anchor=tk.CENTER)

    filipino_button = tk.Button(lang_frame, text="Filipino", bg='#738678', font=("Copperplate Gothic Bold", 30), 
                                command=partial(set_language, "Filipino"))
    filipino_button.place(relx=0.6, rely=0.5, width=250, height=100, anchor=tk.CENTER)

def set_language(lang):
    global selected_language
    selected_language = lang
    initialize_window()


# First Application Classification model loading
classification_model = joblib.load('Source_Classification.joblib')
ct_classification = joblib.load('ct_classification.joblib')
fs_classification = joblib.load('fs_classification.joblib')

# Second Application Classification model loading
classification_model_2nd = joblib.load('Source_Classification2nd.joblib')
ct_classification_2nd = joblib.load('ct_classification2nd.joblib')
fs_classification_2nd = joblib.load('fs_classification2nd.joblib')

# Function to preprocess input data for classification (first application)
def preprocess_classification_input(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    encoded_input = ct_classification.transform(input_array)
    encoded_input = np.delete(encoded_input, 0, axis=1)
    scaled_input = fs_classification.transform(encoded_input)
    return scaled_input

# Function to preprocess input data for classification (second application)
def preprocess_classification_input_2nd(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    encoded_input = ct_classification_2nd.transform(input_array)
    encoded_input = np.delete(encoded_input, 0, axis=1)
    scaled_input = fs_classification_2nd.transform(encoded_input)
    return scaled_input

# Function to predict fertilizer combination (first application)
def predict_fertilizer_combination(input_data):
    processed_input = preprocess_classification_input(input_data)
    prediction = classification_model.predict(processed_input)
    return prediction[0]

# Function to predict fertilizer combination (second application)
def predict_fertilizer_combination_2nd(input_data):
    processed_input = preprocess_classification_input_2nd(input_data)
    prediction = classification_model_2nd.predict(processed_input)
    return prediction[0]

# Regression input processing
def process_regression_input(input_data):
    input_df = pd.DataFrame([input_data], columns=['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC'])
    
    input_df['N/P'] = (input_df['N (ppm)'] / input_df['P (ppm)']) + 1e-6
    input_df['P/K'] = (input_df['P (ppm)'] / input_df['K(ppm)']) + 1e-6
    input_df['N/K'] = (input_df['N (ppm)'] / input_df['K(ppm)']) + 1e-6
    input_df['EC/N'] = (input_df['Soil EC'] / input_df['N (ppm)']) + 1e-6
    input_df['EC/P'] = (input_df['Soil EC'] / input_df['P (ppm)']) + 1e-6
    input_df['EC/K'] = (input_df['Soil EC'] / input_df['K(ppm)']) + 1e-6
    input_df['Soil EC'] = input_df['Soil EC'] * 1000
    return input_df

# First Application Regression functions
def predict_complete_fertilizer(input_df):
    # Load the trained model
    model_complete = joblib.load('model_completev2.joblib')
    
    # Select the relevant columns
    X_complete = input_df[['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC', 'N/P', 'P/K', 'N/K', 'EC/N']].values

    # Apply one-hot encoding
    ct_complete = joblib.load('ct_complete.joblib')
    X_encoded_complete = ct_complete.transform(X_complete)

    # Apply dummy trap (remove first column)
    X_dt_complete = np.delete(X_encoded_complete, 0, axis=1)

    # Apply feature scaling
    fs_complete = joblib.load('fs_complete.joblib')
    X_scaled_complete = fs_complete.transform(X_dt_complete)

    # Make prediction
    prediction_complete = model_complete.predict(X_scaled_complete)

    return prediction_complete[0]

def predict_urea_fertilizer(input_df):
    # Load the trained model
    model_urea = joblib.load('model_ureav1.joblib')
    
    # Select the relevant columns
    X_urea = input_df[['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC', 'N/P', 'P/K']].values

    # Apply one-hot encoding
    ct_urea = joblib.load('ct_urea.joblib')
    X_encoded_urea = ct_urea.transform(X_urea)

    # Apply dummy trap (remove first column)
    X_dt_urea = np.delete(X_encoded_urea, 0, axis=1)

    # Apply feature scaling
    fs_urea = joblib.load('fs_urea.joblib')
    X_scaled_urea = fs_urea.transform(X_dt_urea)

    # Make prediction
    prediction_urea = model_urea.predict(X_scaled_urea)

    return prediction_urea[0]

def predict_mop_fertilizer(input_df):
    # Load the trained model
    model_mop = joblib.load('model_mopv1.joblib')
    
    # Select the relevant columns
    X_mop = input_df[['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC', 'N/P']].values

    # Apply one-hot encoding
    ct_mop = joblib.load('ct_mop.joblib')
    X_encoded_mop = ct_mop.transform(X_mop)

    # Apply dummy trap (remove first column)
    X_dt_mop = np.delete(X_encoded_mop, 0, axis=1)

    # Apply feature scaling
    fs_mop = joblib.load('fs_mop.joblib')
    X_scaled_mop = fs_mop.transform(X_dt_mop)

    # Make prediction
    prediction_mop = model_mop.predict(X_scaled_mop)

    return prediction_mop[0]

def predict_ammosul_fertilizer(input_df):
    # Load the trained model
    model_ammosul = joblib.load('model_ammosulv1.joblib')
    
    # Select the relevant columns
    X_ammosul = input_df[['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC', 'N/P', 'P/K']].values

    # Apply one-hot encoding
    ct_ammosul = joblib.load('ct_ammosul.joblib')
    X_encoded_ammosul = ct_ammosul.transform(X_ammosul)

    # Apply dummy trap (remove first column)
    X_dt_ammosul = np.delete(X_encoded_ammosul, 0, axis=1)

    # Apply feature scaling
    fs_ammosul = joblib.load('fs_ammosul.joblib')
    X_scaled_ammosul = fs_ammosul.transform(X_dt_ammosul)

    # Make prediction
    prediction_ammosul = model_ammosul.predict(X_scaled_ammosul)

    return prediction_ammosul[0]

def predict_ammophos_fertilizer(input_df):
    # Load the trained model
    model_ammophos = joblib.load('model_ammophosv1.joblib')
    
    # Select the relevant columns
    X_ammophos = input_df[['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC', 'N/P']].values

    # Apply one-hot encoding
    ct_ammophos = joblib.load('ct_ammophos.joblib')
    X_encoded_ammophos = ct_ammophos.transform(X_ammophos)

    # Apply dummy trap (remove first column)
    X_dt_ammophos = np.delete(X_encoded_ammophos, 0, axis=1)

    # Apply feature scaling
    fs_ammophos = joblib.load('fs_ammophos.joblib')
    X_scaled_ammophos = fs_ammophos.transform(X_dt_ammophos)

    # Make prediction
    prediction_ammophos = model_ammophos.predict(X_scaled_ammophos)

    return prediction_ammophos[0]

def predict_solophos_fertilizer(input_df):
    # Load the trained model
    model_solophos = joblib.load('model_solophosv2.joblib')
    
    # Select the relevant columns
    X_solophos = input_df[['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC', 'N/P', 'P/K', 'N/K', 'EC/N', 'EC/P']].values

    # Apply one-hot encoding
    ct_solophos = joblib.load('ct_solophos.joblib')
    X_encoded_solophos = ct_solophos.transform(X_solophos)

    # Apply dummy trap (remove first column)
    X_dt_solophos = np.delete(X_encoded_solophos, 0, axis=1)

    # Apply feature scaling
    fs_solophos = joblib.load('fs_solophos.joblib')
    X_scaled_solophos = fs_solophos.transform(X_dt_solophos)

    # Make prediction
    prediction_solophos = model_solophos.predict(X_scaled_solophos)

    return prediction_solophos[0]

def predict_duophos_fertilizer(input_df):
    # Load the trained model
    model_duophos = joblib.load('model_duophos.joblib')
    
    # Select the relevant columns
    X_duophos = input_df[['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC', 'N/P', 'P/K', 'N/K', 'EC/N']].values

    # Apply one-hot encoding
    ct_duophos = joblib.load('ct_duophos.joblib')
    X_encoded_duophos = ct_duophos.transform(X_duophos)

    # Apply dummy trap (remove first column)
    X_dt_duophos = np.delete(X_encoded_duophos, 0, axis=1)

    # Apply feature scaling
    fs_duophos = joblib.load('fs_duophos.joblib')
    X_scaled_duophos = fs_duophos.transform(X_dt_duophos)

    # Make prediction
    prediction_duophos = model_duophos.predict(X_scaled_duophos)

    return prediction_duophos[0]

# Second Application Regression functions (as provided in your new code)
def predict_complete2nd_fertilizer(input_df):
    # Load the trained model
    model_complete2nd = joblib.load('model_complete_2nd_v1.joblib')
    
    # Select the relevant columns
    X_complete2nd = input_df[['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC', 'N/P', 'P/K', 'N/K', 'EC/N', 'EC/P', 'EC/K']].values

    # Apply one-hot encoding
    ct_complete2nd = joblib.load('ct_complete_2nd.joblib')
    X_encoded_complete2nd = ct_complete2nd.transform(X_complete2nd)

    # Apply dummy trap (remove first column)
    X_dt_complete2nd = np.delete(X_encoded_complete2nd, 0, axis=1)

    # Apply feature scaling
    fs_complete2nd = joblib.load('fs_complete2nd.joblib')
    X_scaled_complete2nd = fs_complete2nd.transform(X_dt_complete2nd)

    # Make prediction
    prediction_complete2nd = model_complete2nd.predict(X_scaled_complete2nd)

    return prediction_complete2nd[0]

def predict_urea2nd_fertilizer(input_df):
    # Load the trained model
    model_urea2nd = joblib.load('model_urea_2nd_v1.joblib')
    
    # Select the relevant columns
    X_urea2nd = input_df[['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC', 'N/P', 'P/K']].values

    # Apply one-hot encoding
    ct_urea2nd = joblib.load('ct_urea2nd.joblib')
    X_encoded_urea2nd = ct_urea2nd.transform(X_urea2nd)

    # Apply dummy trap (remove first column)
    X_dt_urea2nd = np.delete(X_encoded_urea2nd, 0, axis=1)

    # Apply feature scaling
    fs_urea2nd = joblib.load('fs_urea2nd.joblib')
    X_scaled_urea2nd = fs_urea2nd.transform(X_dt_urea2nd)

    # Make prediction
    prediction_urea2nd = model_urea2nd.predict(X_scaled_urea2nd)

    return prediction_urea2nd[0]


def predict_mop2nd_fertilizer(input_df):
    # Load the trained model
    model_mop2nd = joblib.load('model_mop_2nd_v1.joblib')
    
    # Select the relevant columns
    X_mop2nd = input_df[['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC']].values

    # Apply one-hot encoding
    ct_mop2nd = joblib.load('ct_mop2nd.joblib')
    X_encoded_mop2nd = ct_mop2nd.transform(X_mop2nd)

    # Apply dummy trap (remove first column)
    X_dt_mop2nd = np.delete(X_encoded_mop2nd, 0, axis=1)

    # Apply feature scaling
    fs_mop2nd = joblib.load('fs_mop2nd.joblib')
    X_scaled_mop2nd = fs_mop2nd.transform(X_dt_mop2nd)

    # Make prediction
    prediction_mop2nd = model_mop2nd.predict(X_scaled_mop2nd)

    return prediction_mop2nd[0]

def predict_ammosul2nd_fertilizer(input_df):
    # Load the trained model
    model_ammosul2nd = joblib.load('model_ammosul_2nd_v1.joblib')
    
    # Select the relevant columns
    X_ammosul2nd = input_df[['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC', 'N/P', 'P/K', 'N/K', 'EC/N', 'EC/P', 'EC/K']].values

    # Apply one-hot encoding
    ct_ammosul2nd = joblib.load('ct_ammosul2nd.joblib')
    X_encoded_ammosul2nd = ct_ammosul2nd.transform(X_ammosul2nd)

    # Apply dummy trap (remove first column)
    X_dt_ammosul2nd = np.delete(X_encoded_ammosul2nd, 0, axis=1)

    # Apply feature scaling
    fs_ammosul2nd = joblib.load('fs_ammosul2nd.joblib')
    X_scaled_ammosul2nd = fs_ammosul2nd.transform(X_dt_ammosul2nd)

    # Make prediction
    prediction_ammosul2nd = model_ammosul2nd.predict(X_scaled_ammosul2nd)

    return prediction_ammosul2nd[0]

def predict_ammophos2nd_fertilizer(input_df):
    # Load the trained model
    model_ammophos2nd = joblib.load('model_ammophos_2nd_v1.joblib')
    
    # Select the relevant columns
    X_ammophos2nd = input_df[['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC', 'N/P', 'P/K', 'N/K', 'EC/N', 'EC/P', 'EC/K']].values

    # Apply one-hot encoding
    ct_ammophos2nd = joblib.load('ct_ammophos2nd.joblib')
    X_encoded_ammophos2nd = ct_ammophos2nd.transform(X_ammophos2nd)

    # Apply dummy trap (remove first column)
    X_dt_ammophos2nd = np.delete(X_encoded_ammophos2nd, 0, axis=1)

    # Apply feature scaling
    fs_ammophos2nd = joblib.load('fs_ammophos2nd.joblib')
    X_scaled_ammophos2nd = fs_ammophos2nd.transform(X_dt_ammophos2nd)

    # Make prediction
    prediction_ammophos2nd = model_ammophos2nd.predict(X_scaled_ammophos2nd)

    return prediction_ammophos2nd[0]

def predict_solophos2nd_fertilizer(input_df):
    # Load the trained model
    model_solophos2nd = joblib.load('model_solophos_2nd_v1.joblib')
    
    # Select the relevant columns
    X_solophos2nd = input_df[['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC', 'N/P', 'P/K', 'N/K', 'EC/N', 'EC/P', 'EC/K']].values

    # Apply one-hot encoding
    ct_solophos2nd = joblib.load('ct_solophos2nd.joblib')
    X_encoded_solophos2nd = ct_solophos2nd.transform(X_solophos2nd)

    # Apply dummy trap (remove first column)
    X_dt_solophos2nd = np.delete(X_encoded_solophos2nd, 0, axis=1)

    # Apply feature scaling
    fs_solophos2nd = joblib.load('fs_solophos2nd.joblib')
    X_scaled_solophos2nd = fs_solophos2nd.transform(X_dt_solophos2nd)

    # Make prediction
    prediction_solophos2nd = model_solophos2nd.predict(X_scaled_solophos2nd)

    return prediction_solophos2nd[0]

def predict_duophos2nd_fertilizer(input_df):
    # Load the trained model
    model_duophos2nd = joblib.load('model_duophos_2nd.joblib')
    
    # Select the relevant columns
    X_duophos2nd = input_df[['Soil pH', 'N (ppm)', 'P (ppm)', 'K(ppm)', 'Crop', 'Soil EC', 'N/P', 'P/K', 'N/K', 'EC/N', 'EC/P']].values

    # Apply one-hot encoding
    ct_duophos2nd = joblib.load('ct_duophos2nd.joblib')
    X_encoded_duophos2nd = ct_duophos2nd.transform(X_duophos2nd)

    # Apply dummy trap (remove first column)
    X_dt_duophos2nd = np.delete(X_encoded_duophos2nd, 0, axis=1)

    # Apply feature scaling
    fs_duophos2nd = joblib.load('fs_duophos2nd.joblib')
    X_scaled_duophos2nd = fs_duophos2nd.transform(X_dt_duophos2nd)

    # Make prediction
    prediction_duophos2nd = model_duophos2nd.predict(X_scaled_duophos2nd)

    return prediction_duophos2nd[0]

def predict_fertilizer(input_data):
    combination_1st = predict_fertilizer_combination(input_data)
    combination_2nd = predict_fertilizer_combination_2nd(input_data)
    processed_input = process_regression_input(input_data)
    
    results_1st = {}
    results_2nd = {}
    
    fertilizers = [
        ("Complete Fertilizer (14-14-14)", predict_complete_fertilizer, predict_complete2nd_fertilizer),
        ("Urea (46-0-0)", predict_urea_fertilizer, predict_urea2nd_fertilizer),
        ("Muriate of Potash (0-0-60)", predict_mop_fertilizer, predict_mop2nd_fertilizer),
        ("Ammonium Sulfate/Ammosul (21-0-0)", predict_ammosul_fertilizer, predict_ammosul2nd_fertilizer),
        ("Ammonium Phosphate/Ammophos (16-20-0)", predict_ammophos_fertilizer, predict_ammophos2nd_fertilizer),
        ("Solophos (0-20-0)", predict_solophos_fertilizer, predict_solophos2nd_fertilizer),
        ("Dicalcium Phosphate/Duophos (0-22-0)", predict_duophos_fertilizer, predict_duophos2nd_fertilizer)
    ]
    
    for i, (fertilizer_name, predict_func_1st, predict_func_2nd) in enumerate(fertilizers):
        if combination_1st[i] == 1:
            amount = predict_func_1st(processed_input)
            if amount > 0:  # Only add if positive
                results_1st[fertilizer_name] = amount
        if combination_2nd[i] == 1:
            amount = predict_func_2nd(processed_input)
            if amount > 0:  # Only add if positive
                results_2nd[fertilizer_name] = amount

    # Add Organic Fertilizer
    crop = input_data[4]  # Assuming crop is at index 4 in input_data
    if crop in ['Jackfruit', 'Banana', 'Calamansi', 'Papaya', 'Coconut', 'Avocado', 'Mango', 'Coffee']:
        organic_fertilizer = 1
    elif crop in ['Rice Dry Season', 'Rice Hybrid Dry Season', 'Rice Inbred Dry Season', 
                  'Rice Inbred Irrigated Dry Season', 'Rice Inbred Irrigated Wet Season', 
                  'Rice Inbred Rainfed Dry Season', 'Rice Inbred Rainfed Wet', 'Rice Inbred Wet Season']:
        organic_fertilizer = 0
    else:
        organic_fertilizer = 1000

    results_1st["Organic Fertilizer"] = organic_fertilizer
    results_2nd["Organic Fertilizer"] = organic_fertilizer

    return results_1st, results_2nd

def read_sensor_data(ser):
    global serial_data  # Declare as global to modify it
    data = []
    start_time = time.time()
    while time.time() - start_time < 10:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            print(f"Received: {line}")
            if "Soil Sensor Readings" in line:
                for _ in range(5):
                    line = ser.readline().decode('utf-8').strip()
                    print(f"Data line: {line}")
                    if ":" in line:
                        _, value = line.split(":", 1)
                        value = re.findall(r"[-+]?\d*\.\d+|\d+", value)[0]
                        if _ == 0:  # pH
                            data.append(float(value))
                        elif _ == 1:  # EC
                            data.append(float(value)/1000)
                        elif _ == 2:  # N
                            data.append(max(float(value) / 7.7, 0.000001))
                        elif _ == 3:  # P
                            data.append(max(float(value) / 12.5, 0.000001))
                        else:  # K (add 1e-6 if 0)
                            data.append(max(float(value), 0.000001))
                if len(data) == 5:
                    # Check if N, P, K are all 0 (or very close to 0)
                    if all(data[i] <= 1e-6 for i in range(2, 5)):
                        print("Please insert the sensor properly. N, P, K values are all 0 or very close to 0.")
                        return None
                    
                    # Rearrange data: [pH, N, P, K, Crop, EC] (with Crop to be filled later and EC swapped)
                    rearranged_data = [data[1], data[2], data[3], data[4], None, data[0]]
                    serial_data = rearranged_data
                    return rearranged_data

    print("Timeout: No valid sensor data received")
    return None

prediction_complete = threading.Event()

def start_progress(progress_bar, duration, next_function):
    progress_bar['value'] = 0
    max_value = 100
    step = max_value / (duration / 100)  # Update every 100ms

    def update_progress():
        if progress_bar['value'] < max_value:
            progress_bar['value'] += step
            root.after(100, update_progress)
        else:
            next_function()

    update_progress()

def sensor_data_thread(selected_crop):
    global output_result, serial_data
    ser = serial.Serial('/dev/ttyACM0', 4800, timeout=1)
    time.sleep(15)

    input_data = read_sensor_data(ser)

    if input_data:
        input_data[4] = crop_map.get(selected_crop, selected_crop).lower()  # Get mapped value, or use original if not found
        results_1st, results_2nd = predict_fertilizer(input_data)

        output_result = "First Application Fertilizer Recommendations:\n"
        for fertilizer, amount in results_1st.items():
            output_result += f"{fertilizer}: {amount:.2f} kg\n"

        output_result += "\nSecond Application Fertilizer Recommendations:\n"
        for fertilizer, amount in results_2nd.items():
            output_result += f"{fertilizer}: {amount:.2f} kg\n"

        prediction_complete.set()  # Signal that prediction is complete
    else:
        print("No valid sensor data available for prediction. Please check the sensor.")
        root.after(0, create_window_error)
    
    ser.close()

def update_gui_with_results():
    global progress_bar
    progress_bar.stop()
    create_window_output()

# Declare progress_bar, init_label, and clicked as global variables
progress_bar = None
init_label = None
clicked = None


def close_popup(popup):
    popup.destroy()


def initialize_window():
    global progress_bar, init_label

    init_label.config(text=translations[selected_language]["initialize"], font=("Copperplate Gothic Bold", 60, "bold"))
    init_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    progress_bar = ttk.Progressbar(root, orient="horizontal", length=600, mode="determinate")
    progress_bar.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    start_progress(progress_bar, 3000, lambda: [close_popup(init_label), close_popup(progress_bar), show_main_window1()])

def start_loading_animation():
    progress_bar.start(20)


def stop_loading_animation():
    progress_bar.stop(50)


def show_main_window1():
    main_frame = ttk.Frame(root)
    main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=1, relheight=1)

    # Logo frame and image
    logo_frame = tk.Frame(main_frame)
    logo_frame.place(relx=0.5, rely=0, anchor=tk.N)
    logo_image = tk.PhotoImage(file="dahon (1).png")
    logo_label = tk.Label(logo_frame, image=logo_image)
    logo_label.image = logo_image
    logo_label.pack(pady=10)

    # Warning sign image
    warning_image = tk.PhotoImage(file="warning-sign.png").subsample(4)
    warning_label = tk.Label(main_frame, image=warning_image)
    warning_label.image = warning_image
    warning_label.place(relx=0.15, rely=0.25)

    # Main label and instructions
    label = tk.Label(main_frame, text=translations[selected_language]["read_carefully"],
                     font=("Tahoma", 40, "bold"))
    label.place(relx=0.55, rely=0.32, anchor=tk.CENTER)

    label_text = translations[selected_language]["instructions"]
    label1 = tk.Label(main_frame, text=label_text, font=("Tahoma", 37), justify=tk.LEFT, anchor='w', bg=None)
    label1.place(relx=0.5, rely=0.58, anchor=tk.CENTER)

    ok_button = tk.Button(main_frame, text="OK", bg='#738678', font=("Copperplate Gothic Bold", 50), command=create_window2)
    ok_button.place(relx=0.88, rely=0.85, width=250, height=100, anchor=tk.CENTER)

def create_window2():
    window2_frame = ttk.Frame(root)
    window2_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=1, relheight=1)

    # Logo frame and image (unchanged)
    logo_frame = tk.Frame(window2_frame)
    logo_frame.place(relx=0.5, rely=0, anchor=tk.N)
    logo_image = tk.PhotoImage(file="dahon (1).png")
    logo_label = tk.Label(logo_frame, image=logo_image)
    logo_label.image = logo_image
    logo_label.pack(pady=10)

    # Main label
    label = tk.Label(window2_frame,
                     text=translations[selected_language]["insert_probe"],
                     font=("Tahoma", 40, "bold"), bg=None)
    label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    next_button = tk.Button(window2_frame, text=translations[selected_language]["next"], bg='#738678', font=("Copperplate Gothic Bold", 30), command=create_window3)
    next_button.place(relx=0.92, rely=0.85, anchor=tk.CENTER)


def create_window3():
    window3_frame = ttk.Frame(root)
    window3_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=1, relheight=1)

    # Logo image (unchanged)
    logo_image = tk.PhotoImage(file="dahon (1).png")
    logo_label = tk.Label(window3_frame, image=logo_image)
    logo_label.image = logo_image
    logo_label.pack(pady=10)

    label = tk.Label(window3_frame, text=translations[selected_language]["press_button"], font=("Tahoma", 40, "bold"))
    label.place(relx=0.5, rely=0.35, anchor=tk.CENTER)

    button = tk.Button(window3_frame, text="Start", font=("Copperplate Gothic Bold", 40), command=create_window4,
                       anchor="center", bg="#738678", width=8, height=2)
    button.place(relx=0.5, rely=0.60, anchor=tk.CENTER)


def create_window4():
    global selected_crop
    
    window4_frame = ttk.Frame(root)
    window4_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=1, relheight=1)

    # Logo image (unchanged)
    logo_image = tk.PhotoImage(file="dahon (1).png")
    logo_label = tk.Label(window4_frame, image=logo_image)
    logo_label.image = logo_image
    logo_label.pack(pady=10)

    label = tk.Label(window4_frame, text=translations[selected_language]["select_crop"], font=("Tahoma", 45, "bold"))
    label.place(relx=0.13, rely=0.45, anchor=tk.CENTER)

    label = tk.Label(window4_frame, text=translations[selected_language]["scroll_here"], font=("Tahoma", 40, "bold"))
    label.place(relx=0.87, rely=0.28, anchor=tk.CENTER)

    # Create Treeview
    crops_column = translations[selected_language]["crops"]
    tree = ttk.Treeview(window4_frame, columns=(crops_column,), show="headings", height=10)
    tree.heading(crops_column, text=crops_column)
    tree.column(crops_column, width=730, anchor="center")

    # Add crops to the Treeview (unchanged)
    crops = ['Avocado', 'Baguio Beans', 'Banana', 'Basil', 'Beans', 'Bellpepper', 'Bittergourd', 'Broccoli', 'Cabbage', 'Cacao', 'Calamansi','Carrot', 'Cassava',
               'Cauliflower', 'Chili', 'Chinese Cabbage', 'Coconut', 'Coffee', 'Corn Hybrid Wet Season','Corn OPV Wet Season', 'Cucumber', 'Dragonfruit', 'Eggplant',
               'Garlic', 'Ginger', 'Green Chili', 'Green Corn Hybrid Dry Season', 'Green Corn Hybrid Wet Season', 'Green Corn OPV Dry Season', 'Green Corn OPV Wet Season',
               'Green Corn Wet Season', 'Hot Chili', 'Hot Pepper', 'Jackfruit', 'Kangkong', 'Lanzones', 'Lettuce', 'Mango', 'Mungbean', 'Mustard', 'Okra', 'Onion', 'Papaya',
               'Patola', 'Peanut', 'Pineapple', 'Potato', 'Pumpkin', 'Radish', 'Rice Dry Season', 'Rice Hybrid Dry Season', 'Rice Inbred Dry Season',
               'Rice Inbred Irrigated Dry Season', 'Rice Inbred Irrigated Wet Season', 'Rice Inbred Rainfed Dry Season', 'Rice Inbred Rainfed Wet', 'Rice Inbred Wet Season',
               'Rosemary', 'Sittao','Sorghum', 'Soybean', 'Spinach', 'Squash', 'Sweet Potato','Thyme', 'Tomato', 'Turmeric', 'Ube', 'Upo', 'Watermelon', 'Winged Beans', 
               'Yellow Corn Dry Season', 'Yellow Corn Wet Season']
    for crop in crops:
        tree.insert("", "end", values=(crop,))

    tree.place(relx=0.50, rely=0.57, anchor=tk.CENTER)

    # Rest of the function remains the same...

    # Style configuration (unchanged)
    style = ttk.Style()
    style.configure("Treeview", font=("Tahoma", 35), rowheight=58)
    style.configure("Treeview.Heading", font=("Tahoma", 30, "bold"))
    style.configure("Custom.Vertical.TScrollbar", background="grey", troughcolor="grey", width=50, arrowsize=100)

    # Scrollbar for Treeview (unchanged)
    scrollbar = ttk.Scrollbar(window4_frame, orient="vertical", command=tree.yview, style="Custom.Vertical.TScrollbar")
    scrollbar.place(relx=0.86, rely=0.55, anchor=tk.W, height=370, width=25)
    tree.configure(yscrollcommand=scrollbar.set)
    
    def on_select(event):
        global selected_crop
        selected_items = tree.selection()
        if selected_items:
            item = selected_items[0]
            selected_crop = tree.item(item, "values")[0]
            confirm_button.config(state=tk.NORMAL)
        else:
            selected_crop = None
            confirm_button.config(state=tk.DISABLED)

    tree.bind("<<TreeviewSelect>>", on_select)

    def confirm_selection():
        if selected_crop:
            create_window5()
        else:
            messagebox.showwarning("No Selection", translations[selected_language]["no_selection"])

    confirm_button = tk.Button(window4_frame, text="CONFIRM", bg='#738678', font=("Copperplate Gothic Bold", 40), command=confirm_selection, state=tk.DISABLED)
    confirm_button.config(width=8, height=1)
    confirm_button.place(relx=0.870, rely=0.85, anchor=tk.CENTER)

crop_map = {
    'Avocado': 'avocado',
    'Baguio Beans': 'baguio beans',
    'Banana': 'banana',
    'Basil': 'basil',
    'Beans': 'beans',
    'Bellpepper': 'bellpepper',
    'Bittergourd': 'bittergourd',
    'Broccoli': 'broccoli',
    'Cabbage': 'cabbage',
    'Cacao': 'cacao',
    'Calamansi': 'calamansi',
    'Carrot': 'carrot', 
    'Cassava': 'cassava',
    'Cauliflower': 'cauliflower', 
    'Chili': 'chili', 
    'Chinese Cabbage': 'chinese cabbage', 
    'Coconut': 'coconut',
    'Coffee': 'coffee', 
    'Corn Hybrid Wet Season': 'corn hybrid wet',
    'Corn OPV Wet Season': 'corn OPV wet', 
    'Cucumber': 'cucumber', 
    'Dragonfruit': 'dragonfruit', 
    'Eggplant': 'eggplant',
    'Garlic': 'garlic', 
    'Ginger': 'ginger', 
    'Green Chili': 'green chili', 
    'Green Corn Hybrid Dry Season': 'green corn hybrid dry', 
    'Green Corn Hybrid Wet Season': 'green corn hybrid wet', 
    'Green Corn OPV Dry Season': 'green corn OPV dry', 
    'Green Corn OPV Wet Season': 'green corn OPV wet',
    'Green Corn Wet Season': 'green corn wet', 
    'Hot Chili': 'hot chili', 
    'Hot Pepper': 'hot pepper', 
    'Jackfruit': 'jackfruit', 
    'Kangkong': 'kangkong', 
    'Lanzones': 'lanzones', 
    'Lettuce': 'lettuce', 
    'Mango': 'mango', 
    'Mungbean': 'mungbean', 
    'Mustard': 'mustard', 
    'Okra': 'okra', 
    'Onion': 'onion', 
    'Papaya': 'papaya',
    'Patola': 'patola', 
    'Peanut': 'peanut', 
    'Pineapple': 'pineapple', 
    'Potato': 'potato', 
    'Pumpkin': 'pumpkin', 
    'Radish': 'radish', 
    'Rice Dry Season': 'rice dry', 
    'Rice Hybrid Dry Season': 'rice hybrid dry', 
    'Rice Inbred Dry Season': 'rice inbred dry',
    'Rice Inbred Irrigated Dry Season': 'rice inbred irrigated dry', 
    'Rice Inbred Irrigated Wet Season': 'rice inbred irrigated wet', 
    'Rice Inbred Rainfed Dry Season': 'rice inbred rainfed dry', 
    'Rice Inbred Rainfed Wet': 'rice inbred rainfed wet', 
    'Rice Inbred Wet Season': 'rice inbred wet',
    'Rosemary': 'rosemary', 
    'Sittao': 'sittao',
    'Sorghum': 'sorghum', 
    'Soybean': 'soybean', 
    'Spinach': 'spinach', 
    'Squash': 'squash', 
    'Sweet Potato': 'sweet potato',
    'Thyme': 'thyme', 
    'Tomato': 'tomato', 
    'Turmeric': 'turmeric', 
    'Ube': 'ube', 
    'Upo': 'upo', 
    'Watermelon': 'watermelon', 
    'Winged Beans': 'winged beans', 
    'Yellow Corn Dry Season': 'corn dry', 
    'Yellow Corn Wet Season': 'corn wet'
}

def create_window5():
    global progress_bar, collecting_data_label

    window5_frame = ttk.Frame(root)
    window5_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=1, relheight=1)

    # Logo image (unchanged)
    logo_image = tk.PhotoImage(file="dahon (1).png")
    logo_label = tk.Label(window5_frame, image=logo_image)
    logo_label.image = logo_image
    logo_label.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    # Display the chosen crop
    chosen_crop = selected_crop
    collecting_data_label = tk.Label(window5_frame, text=translations[selected_language]["collecting_data"].format(chosen_crop), font=("Tahoma", 40, "bold"))
    collecting_data_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    # Progress bar (unchanged)
    progress_bar = ttk.Progressbar(window5_frame, orient="horizontal", length=200, mode="determinate")
    progress_bar.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    # Start the sensor data thread (unchanged)
    prediction_thread = threading.Thread(target=sensor_data_thread, args=(selected_crop,))
    prediction_thread.start()

    # Start progress bar (unchanged)
    start_progress(progress_bar, 18000, check_prediction_complete)

def create_window_error():
    window_error_frame = ttk.Frame(root)
    window_error_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=1, relheight=1)

    # Logo frame and image (unchanged)
    logo_frame = tk.Frame(window_error_frame)
    logo_frame.place(relx=0.5, rely=0, anchor=tk.N)
    logo_image = tk.PhotoImage(file="dahon (1).png")
    logo_label = tk.Label(logo_frame, image=logo_image)
    logo_label.image = logo_image
    logo_label.pack(pady=10)

    # Main label
    label = tk.Label(window_error_frame,
                     text=translations[selected_language]["insufficient_data"],
                     font=("Tahoma", 40, "bold"), bg=None)
    label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    next_button = tk.Button(window_error_frame, text=translations[selected_language]["retry"], bg='#738678', font=("Copperplate Gothic Bold", 40), command=show_main_window1)
    next_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)


def check_prediction_complete():
    if prediction_complete.is_set():
        stop_progress_and_continue()
    else:
        # Check again after 100ms if the progress bar hasn't reached 100%
        if progress_bar['value'] < 100:
            root.after(100, check_prediction_complete)
        else:
            stop_progress_and_continue()

def stop_progress_and_continue():
    global progress_bar, collecting_data_label
    if 'progress_bar' in globals() and progress_bar.winfo_exists():
        progress_bar.stop()
        progress_bar.destroy()
    if 'collecting_data_label' in globals() and collecting_data_label.winfo_exists():
        collecting_data_label.destroy()
    create_window_output() 

def create_window_error():
    window_error_frame = ttk.Frame(root)
    window_error_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=1, relheight=1)

    # Logo frame
    logo_frame = tk.Frame(window_error_frame)
    logo_frame.place(relx=0.5, rely=0, anchor=tk.N)

    # Logo image
    logo_image = tk.PhotoImage(file="dahon (1).png")
    logo_label = tk.Label(logo_frame, image=logo_image)
    logo_label.image = logo_image  # Keeping a reference to the image object
    logo_label.pack(pady=10)

    # Main label
    label = tk.Label(window_error_frame,
                     text="Insufficient/Incorrect data. \n Try again.",
                     font=("Tahoma", 40, "bold"), bg=None)
    label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    next_button = tk.Button(window_error_frame, text="RETRY", bg='#738678', font=("Copperplate Gothic Bold", 40), command=show_main_window1)
    next_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)


def create_window_restart():
    window_restart_frame = ttk.Frame(root)
    window_restart_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=1, relheight=1)

    # Logo frame
    logo_frame = tk.Frame(window_restart_frame)
    logo_frame.place(relx=0.5, rely=0, anchor=tk.N)

    # Logo image
    logo_image = tk.PhotoImage(file="dahon (1).png")
    logo_label = tk.Label(logo_frame, image=logo_image)
    logo_label.image = logo_image
    logo_label.pack(pady=10)

    # Main label
    label = tk.Label(window_restart_frame,
                     text=translations[selected_language]["test_again"],
                     font=("Tahoma", 32, "bold"), bg=None)
    label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    yes_button = tk.Button(window_restart_frame, text=translations[selected_language]["yes"], bg='#738678', font=("Copperplate Gothic Bold", 40),
                            command=show_main_window1)
    yes_button.place(relx=0.4, rely=0.6, anchor=tk.CENTER)

    no_button = tk.Button(window_restart_frame, text=translations[selected_language]["no"], bg='#738678', font=("Copperplate Gothic Bold", 40),
                           command=create_window_confirmation)
    no_button.place(relx=0.6, rely=0.6, anchor=tk.CENTER)

def create_window_confirmation():
    window_confirmation_frame = ttk.Frame(root)
    window_confirmation_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=1, relheight=1)

    # Logo frame
    logo_frame = tk.Frame(window_confirmation_frame)
    logo_frame.place(relx=0.5, rely=0, anchor=tk.N)

    # Logo image
    logo_image = tk.PhotoImage(file="dahon (1).png")
    logo_label = tk.Label(logo_frame, image=logo_image)
    logo_label.image = logo_image
    logo_label.pack(pady=10)

    # Main label
    label = tk.Label(window_confirmation_frame,
                     text=translations[selected_language]["shutdown_confirm"],
                     font=("Tahoma", 40, "bold"), bg=None)
    label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    yes_button = tk.Button(window_confirmation_frame, text=translations[selected_language]["yes"], bg='#738678', font=("Copperplate Gothic Bold", 40),
                           command=create_window_shutdown)
    yes_button.place(relx=0.4, rely=0.6, anchor=tk.CENTER)

    no_button = tk.Button(window_confirmation_frame, text=translations[selected_language]["no"], bg='#738678', font=("Copperplate Gothic Bold", 40),
                          command=create_window_restart)
    no_button.place(relx=0.6, rely=0.6, anchor=tk.CENTER)

def create_window_output():
    window_output_frame = ttk.Frame(root)
    window_output_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=1, relheight=1)

    # Logo frame and image (unchanged)
    logo_frame = tk.Frame(window_output_frame)
    logo_frame.place(relx=0.5, rely=0.05, anchor=tk.N)
    logo_image = tk.PhotoImage(file="dahon (1).png")
    logo_label = tk.Label(logo_frame, image=logo_image)
    logo_label.image = logo_image
    logo_label.pack(pady=5)

    # Main label
    chosen_crop = selected_crop
    collecting_data_label = tk.Label(window_output_frame, 
                                     text=translations[selected_language]["recommended_fertilizer"].format(chosen_crop),
                                     font=("Tahoma", 36, "bold"))
    collecting_data_label.place(relx=0.5, rely=0.25, anchor=tk.CENTER)

    # Create a frame for the output text (unchanged)
    output_frame = tk.Frame(window_output_frame, bg='white', bd=2, relief=tk.RAISED)
    output_frame.place(relx=0.5, rely=0.62, anchor=tk.CENTER, width=1000, height=600)

    # Display the output_result with highlighting (unchanged)
    output_text = tk.Text(output_frame, font=("Tahoma", 24), wrap=tk.WORD, padx=20, pady=20)
    output_text.pack(expand=True, fill=tk.BOTH)

    # Add a scrollbar (unchanged)
    scrollbar = tk.Scrollbar(output_frame, command=output_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    output_text.config(yscrollcommand=scrollbar.set)

    # Configure tags for highlighting (unchanged)
    output_text.tag_configure("bold", font=("Tahoma", 25, "bold"))
    output_text.tag_configure("black",  font=("Tahoma", 23, "bold"))
    output_text.tag_configure("black",  font=("Tahoma", 26))
    output_text.tag_configure("black",  font=("Tahoma", 23, "bold"))

    # Split the output_result into lines (unchanged)
    lines = output_result.split('\n')

    is_first_application = True
    for line in lines:
        if "Application" in line:
            output_text.insert(tk.END, line + "\n\n", "bold")
            is_first_application = "First" in line
        elif ":" in line:
            fertilizer, amount = line.split(":")
            amount = amount.strip()
            if float(amount.split()[0]) > 0:  # Check if the amount is greater than 0
                if "Organic Fertilizer" in fertilizer:
                    if is_first_application:
                        output_text.insert(tk.END, fertilizer + ":", "red")
                        output_text.insert(tk.END, amount + "\n\n", "red")
                else:
                    output_text.insert(tk.END, fertilizer + ":", "blue")
                    output_text.insert(tk.END, amount + "\n", "green")
        else:
            output_text.insert(tk.END, line + "\n")

    output_text.config(state=tk.DISABLED)  # Make the text widget read-only

    next_button = tk.Button(window_output_frame, text="OK", bg='#738678', font=("Copperplate Gothic Bold", 40), command=create_window_restart)
    next_button.place(relx=0.92, rely=0.9, anchor=tk.CENTER)

def create_window_shutdown():
    global progress_bar

    window_shutdown_frame = ttk.Frame(root)
    window_shutdown_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=1, relheight=1)

    # Logo frame
    logo_frame = tk.Frame(window_shutdown_frame)
    logo_frame.place(relx=0.5, rely=0, anchor=tk.N)

    # Logo image
    logo_image = tk.PhotoImage(file="dahon (1).png")
    logo_label = tk.Label(logo_frame, image=logo_image)
    logo_label.image = logo_image
    logo_label.pack(pady=10)

    # Main label
    label = tk.Label(window_shutdown_frame,
                     text=translations[selected_language]["shutting_down"],
                     font=("Copperplate Gothic Bold", 40, "bold"), bg=None)
    label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    progress_bar = ttk.Progressbar(window_shutdown_frame, orient="horizontal", length=200, mode="determinate")
    progress_bar.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    start_progress(progress_bar, 3000, root.quit)  # Quit the application after 3 seconds

root = tk.Tk()
root.title("Fertilizer Recommendation")
root.attributes("-fullscreen", True)

init_label = tk.Label(root, text="", font=("Copperplate Gothic Bold", 40))
progress_bar = ttk.Progressbar(root, orient="horizontal", mode="indeterminate", length=600)

create_language_selection()

root.mainloop()
