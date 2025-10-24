# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Task1: Perceptron & Adaline")

# -------------------------
# data loading 
# -------------------------
@st.cache_data
def load_penguins() -> pd.DataFrame:
    
    path = r"D:\semester_7\NN&DL\penguins.csv"
    df = pd.read_csv(path)
    # rename columns to match lab description if needed
    df = df.rename(columns={
        "species": "Species",
        "culmen_length_mm": "CulmenLength",
        "culmen_depth_mm": "CulmenDepth",
        "flipper_length_mm": "FlipperLength",
        "body_mass_g": "BodyMass",
        "island": "OriginLocation"
    })
    df = df[["Species", "CulmenLength", "CulmenDepth", "FlipperLength", "OriginLocation", "BodyMass"]]
    return df

# -------------------------
# preprocess
# -------------------------




# -------------------------
# Algorithms Section 
# -------------------------

# TODO: Implement weight initialization (small random values, optional bias)


def perceptron_train(X, y, lr=0.01, max_epochs=50, bias=True):
   
    samples, features = X.shape

    # Add bias input (x0 = 1)
    if bias:
        X = np.column_stack((np.ones(samples), X))

    # Random init for weights (small values)
    weights = np.random.randn(X.shape[1]) * 0.01
    epoch_errors = []

    for _ in range(max_epochs):
        mistakes = 0
        for xi, target in zip(X, y):
            activation = np.dot(weights, xi)
            prediction = 1 if activation >= 0 else -1

            if prediction != target:
                weights += lr * (target - prediction) * xi
                mistakes += 1

        epoch_errors.append(mistakes)

    return weights, epoch_errors



def train_adaline(X_train, y_train, eta=0.01, epochs=50, use_bias=True):
  
    n_samples, n_features = X_train.shape

    # Add bias as input 
    if use_bias:
        X_train = np.c_[np.ones((n_samples, 1)), X_train]

    # Initialize weights
    w = np.random.randn(X_train.shape[1]) * 0.01
    mse_history = []

    for _ in range(epochs):
        y_pred = np.dot(X_train, w)

        errors = y_train - y_pred
     
        w += eta * np.dot(X_train.T, errors)

        mse = np.mean(errors ** 2) / 2
        mse_history.append(mse)

    return w, mse_history




# -------------------------
# Helper Functions 
# -------------------------

# TODO: Split dataset into 30 training and 20 testing samples per class

# TODO: Standardize training and testing data using train mean and std

# TODO: Create manual confusion matrix (TP, TN, FP, FN)


# -------------------------
# Streamlit GUI
# -------------------------
st.title("Task 1: Perceptron & Adaline ")

df = load_penguins()
df_proc, info = preprocess(df)

with st.sidebar:
    st.header("Experiment settings")

    feat_options = ["CulmenLength", "CulmenDepth", "FlipperLength", "OriginLocation_enc", "BodyMass"]
    selected_features = st.multiselect(
        "Select exactly TWO features (for 2D decision boundary):",
        feat_options,
        default=[]
    )

    if len(selected_features) < 2:
        st.warning("You must select 2 features.")
    elif len(selected_features) > 2:
        st.warning("Please select ONLY 2 features.")
    
    st.divider()

    # Class selection
    classes_unique = sorted(df_proc["Species"].unique().tolist())
    class_pair = st.multiselect(
        "Select exactly TWO classes:",
        classes_unique,
        default=[]
    )

    if len(class_pair) < 2:
        st.warning("You must select 2 classes.")
    elif len(class_pair) > 2:
        st.warning("Please select ONLY 2 classes.")
    
    st.divider()

    # Algorithm settings
    algorithm = st.radio("Algorithm:", ["Perceptron", "Adaline"])
    eta = st.number_input("Learning rate (η):", value=0.01, format="%.5f", step=0.001)
    epochs = st.number_input("Number of epochs:", min_value=1, value=50, step=1)
    mse_threshold = None
    if algorithm == "Adaline":
        mse_threshold = st.number_input("MSE threshold (leave 0 for no early stop):", value=0.0, format="%.6f")
        if mse_threshold == 0.0:
            mse_threshold = None
    use_bias = st.checkbox("Include bias (offset)", value=True)
    seed = st.number_input("Random seed:", min_value=0, value=42)
    st.write("---")

    # Run button only enabled when valid selections made
    valid_selection = (len(selected_features) == 2) and (len(class_pair) == 2)
    run_button = st.button("Run Training", disabled=not valid_selection)

if not valid_selection:
    st.info("Please select exactly 2 features and exactly 2 classes to continue.")


# ===============================================================
# 🚧 MAIN IMPLEMENTATION SECTION — To be completed by the team 🚧
# ===============================================================
# Only run when button clicked and choices valid
if run_button:
    if len(selected_features) != 2 or len(class_pair) != 2:
        st.error("Please select exactly 2 features and exactly 2 classes in the sidebar.")
    else:
        # -----------------------------------------------------------
        # ✅ STEP 1 — Prepare train and test datasets
        # -----------------------------------------------------------
        # • Split the data into train and test sets for the two selected classes.
        # • Each class should have 30 training samples and 20 testing samples.
        # • Extract the two chosen features.
        # • Map labels: first selected class → -1, second selected class → +1.
        # • Standardize the features (fit on train, apply on test).
        # TODO: Implement Step 1
        pass

        # -----------------------------------------------------------
        # ✅ STEP 2 — Initialize weights
        # -----------------------------------------------------------
        # • Initialize random weights and bias (if enabled).
        # • Use a random seed for reproducibility.
        # TODO: Implement Step 2
        pass

        # -----------------------------------------------------------
        # ✅ STEP 3 — Train the model
        # -----------------------------------------------------------
        # • If the selected algorithm is “Perceptron”, train using the Perceptron rule.
        # • If the selected algorithm is “Adaline”, train using the Adaline rule.
        # • Record the training history (errors or MSE per epoch).
        # • Display the training progress as a line chart.
        # TODO: Implement Step 3
        pass

        # -----------------------------------------------------------
        # ✅ STEP 4 — Test and evaluate
        # -----------------------------------------------------------
        # • Compute the model’s predictions for the test set.
        # • Construct the confusion matrix manually (TP, TN, FP, FN).
        # • Calculate and display accuracy.
        # TODO: Implement Step 4
        pass

        # -----------------------------------------------------------
        # ✅ STEP 5 — Plot decision boundary
        # -----------------------------------------------------------
        # • Plot the 2D decision boundary using the two selected features.
        # • Show training and test samples with different markers.
        # • Add labels and legend for both classes.
        # • Display the figure in Streamlit.
        # TODO: Implement Step 5
        pass

        # -----------------------------------------------------------
        # ✅ STEP 6 — Display notes and dataset preview
        # -----------------------------------------------------------
        # • Add notes about label mapping and preprocessing.
        # • Optionally show sample rows from the training and testing datasets.
        # TODO: Implement Step 6
        pass
