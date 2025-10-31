# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from matplotlib.colors import ListedColormap

st.set_page_config(layout="wide", page_title="Task1: Perceptron & Adaline")

# -------------------------
# data loading 
# -------------------------
@st.cache_data
def load_penguins() -> pd.DataFrame:
    
    path = r"./penguins.csv"
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

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()

    # handling nulls
    df['CulmenLength'] = df.groupby('Species')['CulmenLength'].transform(lambda x: x.fillna(x.mean()))
    df['CulmenDepth'] = df.groupby('Species')['CulmenDepth'].transform(lambda x: x.fillna(x.mean()))
    df['FlipperLength'] = df.groupby('Species')['FlipperLength'].transform(lambda x: x.fillna(x.mean()))
    df['BodyMass'] = df.groupby('Species')['BodyMass'].transform(lambda x: x.fillna(x.mean()))
    
    if df["OriginLocation"].isnull().any():
        df["OriginLocation"] = df["OriginLocation"].fillna(df["OriginLocation"].mode()[0])
    # Encode OriginLocation as integers (not one-hot)
    orig_levels = sorted(df["OriginLocation"].unique().tolist())
    origin_map = {v: i for i, v in enumerate(orig_levels)}
    df["OriginLocation_enc"] = df["OriginLocation"].map(origin_map)

    info = {"origin_map": origin_map, "orig_levels": orig_levels}
    return df, info


# -------------------------
# Algorithms Section 
# -------------------------

# TODO: Implement weight initialization (small random values, optional bias)

def initialize_weights(n_features, use_bias=True, seed=42, scale=0.01):
    np.random.seed(seed)
    weights = np.random.normal(0.0, scale, n_features)
    bias = np.random.normal(0.0, scale) if use_bias else 0.0
    return weights, bias


# -------------------------
# Algorithms Section 
# -------------------------

# TODO: Implement weight initialization (small random values, optional bias)

def perceptron_train(X, y, weights, bias, eta=0.01, epochs=50, use_bias=True):
   
    n_samples = X.shape[0]
    errors_history = []
    
    current_weights = weights.copy()
    current_bias = bias
    
    for epoch in range(epochs):
        errors = 0
        for i in range(n_samples):
            # Calculate net input
            net_input = np.dot(X[i], current_weights) + (current_bias if use_bias else 0)
            
            # Apply activation function (step function)
            prediction = 1 if net_input >= 0 else -1
            
            # Update weights if misclassified
            if prediction != y[i]:
                errors += 1
                update = eta * y[i]
                current_weights += update * X[i]
                if use_bias:
                    current_bias += update
        
        errors_history.append(errors)
        
        # Early stopping if no errors
        if errors == 0:
            break
    
    return current_weights, current_bias, errors_history

def adaline_train(X, y, weights, bias, eta=0.01, epochs=50, use_bias=True, mse_threshold=None):
   
    n_samples = X.shape[0]
    mse_history = []
    
    current_weights = weights.copy()
    current_bias = bias
    
    for epoch in range(epochs):
        errors = []
        for i in range(n_samples):
            # Calculate net input (linear activation)
            net_input = np.dot(X[i], current_weights) + (current_bias if use_bias else 0)
            
            # Calculate error (difference between net input and target)
            error = y[i] - net_input
            errors.append(error)
            
            # Update weights using gradient descent
            update = eta * error
            current_weights += update * X[i]
            if use_bias:
                current_bias += update
        
        # Calculate MSE for this epoch
        mse = np.mean(np.array(errors) ** 2)
        mse_history.append(mse)
        
        # Early stopping if MSE threshold is met
        if mse_threshold is not None and mse <= mse_threshold:
            break
    
    return current_weights, current_bias, mse_history

# -------------------------
# Helper Functions 
# -------------------------

# TODO: Split dataset into 30 training and 20 testing samples per class
def split_by_class(data: pd.DataFrame, target_col: str, chosen_classes: Tuple[str, str], seed: int = 42):
    
    np.random.seed(seed)
    train_list, test_list = [], []

    for c in chosen_classes:
        class_rows = data[data[target_col] == c].copy()
        total = len(class_rows)

        if total < 50:
            st.warning(f"Class '{c}' has only {total} samples (expected ~50). Using all available data.")

        shuffled = np.random.permutation(class_rows.index)
        cutoff = 30 if total >= 50 else int(total * 0.6)

        train_rows = class_rows.loc[shuffled[:cutoff]]
        test_rows = class_rows.loc[shuffled[cutoff:]]
        train_list.append(train_rows)
        test_list.append(test_rows)

    train_data = pd.concat(train_list, ignore_index=True)
    test_data = pd.concat(test_list, ignore_index=True)

    return train_data, test_data


# TODO: Standardize training and testing data using train mean and std


def normalize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    
    mean_vec = np.mean(X_train, axis=0)
    std_vec = np.std(X_train, axis=0)
    std_vec[std_vec == 0] = 1  # avoid division by zero

    X_train_std = (X_train - mean_vec) / std_vec
    X_test_std = (X_test - mean_vec) / std_vec

    return X_train_std, X_test_std, mean_vec, std_vec


# TODO: Create manual confusion matrix (TP, TN, FP, FN)

def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1):
   
    TP = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    TN = int(np.sum((y_true != pos_label) & (y_pred != pos_label)))
    FP = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
    FN = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))

    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

# -------------------------
# Streamlit GUI
# -------------------------
st.title("Task 1: Perceptron & Adaline ")

df = load_penguins()
df_proc, info = preprocess(df)

with st.sidebar:
    st.header("Experiment Settings")

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
    st.info("Please select exactly 2 features and 2 classes to continue.")

# ===============================================================
# MAIN LOGIC
# ===============================================================
if run_button:
    if len(selected_features) != 2 or len(class_pair) != 2:
        st.error("Please select exactly 2 features and 2 classes in the sidebar.")
    else:
        # -----------------------------------------------------------
        # ✅ STEP 1 — Data Preparation
        # -----------------------------------------------------------
        st.subheader("Step 1 – Data Preparation")
        train_df, test_df = split_by_class(df_proc, "Species", tuple(class_pair), seed=seed)

        X_train = train_df[selected_features].to_numpy(dtype=float)
        X_test = test_df[selected_features].to_numpy(dtype=float)

        label_map = {class_pair[0]: -1, class_pair[1]: 1}
        y_train = train_df["Species"].map(label_map).to_numpy(dtype=int)
        y_test = test_df["Species"].map(label_map).to_numpy(dtype=int)

        X_train_std, X_test_std, mu, sigma = normalize_train_test(X_train, X_test)

        st.write(f"Training samples: {len(y_train)}, Testing samples: {len(y_test)}")
        st.success("Data prepared successfully!")

        # -----------------------------------------------------------
        # ✅ STEP 2 — Initialize weights
        # -----------------------------------------------------------
        n_features = 2
        weights, bias = initialize_weights(n_features=n_features, use_bias=use_bias, seed=seed, scale=0.01)
        st.write(f"**Initial Weights:** {np.round(weights, 6).tolist()}")
        st.write(f"**Bias:** {round(bias, 6) if use_bias else 'None (disabled)'}")

        # -----------------------------------------------------------
        # ✅ STEP 3 — Training
        # -----------------------------------------------------------
        st.subheader("Step 3 – Training the Model")
        if algorithm == "Perceptron":
            trained_weights, trained_bias, errors_history = perceptron_train(
                X_train_std, y_train, weights, bias, eta=eta, epochs=epochs, use_bias=use_bias
            )
            st.success("Perceptron training complete!")
            st.line_chart(errors_history, y_label="Misclassifications per Epoch")
        else:
            trained_weights, trained_bias, mse_history = adaline_train(
                X_train_std, y_train, weights, bias, eta=eta, epochs=epochs,
                use_bias=use_bias, mse_threshold=mse_threshold
            )
            st.success("Adaline training complete!")
            st.line_chart(mse_history, y_label="Mean Squared Error per Epoch")

        st.write(f"**Final Weights:** {np.round(trained_weights, 6).tolist()}")
        st.write(f"**Final Bias:** {round(trained_bias, 6) if use_bias else 'None'}")

        # Save model to session state
        st.session_state["weights"] = trained_weights
        st.session_state["bias"] = trained_bias
        st.session_state["mu"] = mu
        st.session_state["sigma"] = sigma
        st.session_state["use_bias"] = use_bias
        st.session_state["class_pair"] = class_pair
        st.session_state["selected_features"] = selected_features
        st.session_state["algorithm"] = algorithm
        st.success("Model saved to session.")

        # -----------------------------------------------------------
        # ✅ STEP 4 — Test & Evaluation
        # -----------------------------------------------------------
        st.subheader("Step 4 – Testing & Evaluation")
        y_pred = np.where(np.dot(X_test_std, trained_weights) + (trained_bias if use_bias else 0) >= 0, 1, -1)
        cm = compute_confusion(y_test, y_pred, pos_label=1)
        accuracy = (cm["TP"] + cm["TN"]) / len(y_test) * 100
        st.json(cm)
        st.write(f"Accuracy: **{accuracy:.2f}%**")

        # -----------------------------------------------------------
        # ✅ STEP 5 — Decision Boundary
        # -----------------------------------------------------------
        st.subheader("Step 5 – Decision Boundary Visualization")
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])

        fig, ax = plt.subplots(figsize=(7, 6))
        x_vals = np.linspace(X_all[:, 0].min() - 1, X_all[:, 0].max() + 1, 200)
        mu0, mu1 = mu[0], mu[1]
        s0, s1 = sigma[0], sigma[1]
        weights, bias = trained_weights, trained_bias

        if abs(weights[1]) < 1e-12:
            if abs(weights[0]) < 1e-12:
                st.warning("Degenerate weights: cannot draw decision boundary.")
            else:
                x_decision = mu0 - (bias * s0 / weights[0])
                ax.axvline(x=x_decision, linestyle="--", color="k")
        else:
            y_vals = mu1 + s1 * (-bias - weights[0] * (x_vals - mu0) / s0) / weights[1]
            ax.plot(x_vals, y_vals, linestyle="--", color="k", label="Decision boundary")

        inv_label_map = {-1: class_pair[0], 1: class_pair[1]}
        for lab_val, lab_name in inv_label_map.items():
            mask_tr = (y_train == lab_val)
            ax.scatter(X_train[mask_tr, 0], X_train[mask_tr, 1], marker="o", s=60, label=f"Train {lab_name}")
            mask_te = (y_test == lab_val)
            ax.scatter(X_test[mask_te, 0], X_test[mask_te, 1], marker="x", s=80, label=f"Test {lab_name}")

        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        ax.legend()
        st.pyplot(fig)

        # -----------------------------------------------------------
        # ✅ STEP 6 – Notes
        # -----------------------------------------------------------
        st.subheader("Step 6 – Notes & Dataset Preview")
        st.info("Features standardized using training mean & std. Circle = train, × = test.")
        with st.expander("Show train/test sample data"):
            st.dataframe(train_df.head(10))
            st.dataframe(test_df.head(10))

# -----------------------------------------------------------
# ✅ STEP 7 — Predict New Sample (Persistent)
# -----------------------------------------------------------
if "weights" in st.session_state:
    st.subheader("Step 7 – Predict a New Sample")

    selected_features = st.session_state["selected_features"]
    col1, col2 = st.columns(2)
    with col1:
        feat1_val = st.number_input(f"Enter {selected_features[0]}:", value=0.0)
    with col2:
        feat2_val = st.number_input(f"Enter {selected_features[1]}:", value=0.0)

    if st.button("Predict Class"):
        X_new = np.array([[feat1_val, feat2_val]])
        X_new_std = (X_new - st.session_state["mu"]) / st.session_state["sigma"]

        weights = st.session_state["weights"]
        bias = st.session_state["bias"]
        use_bias = st.session_state["use_bias"]
        class_pair = st.session_state["class_pair"]

        y_pred_new = np.dot(X_new_std, weights) + (bias if use_bias else 0.0)
        y_class = 1 if y_pred_new >= 0 else -1
        inv_label_map = {-1: class_pair[0], 1: class_pair[1]}
        predicted_label = inv_label_map[y_class]

        st.success(f"Predicted Class: **{predicted_label}**")
        st.write(f"Raw model output: {float(y_pred_new):.4f}")
else:
    st.info("👆 Train the model first, then enter values to predict.")