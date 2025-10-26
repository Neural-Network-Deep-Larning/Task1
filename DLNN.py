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
    
    path = r"D:\Downloads\Lab3\penguins.csv"
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

# TODO: Implement Perceptron training algorithm (update weights per sample, track errors per epoch)

# TODO: Implement Adaline training algorithm (update using error, track MSE per epoch)


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
            st.warning(f"⚠️ Class '{c}' has only {total} samples (expected ~50). Using all available data.")

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
        st.subheader("Step 1 – Data Preparation")

        # Split the two chosen classes into train and test
        train_df, test_df = split_by_class(df_proc, "Species", tuple(class_pair), seed=seed)

        # Extract only the chosen 2 features
        X_train = train_df[selected_features].to_numpy(dtype=float)
        X_test = test_df[selected_features].to_numpy(dtype=float)

        # Map species → numeric labels  (-1 for first class, +1 for second)
        label_map = {class_pair[0]: -1, class_pair[1]: 1}
        y_train = train_df["Species"].map(label_map).to_numpy(dtype=int)
        y_test = test_df["Species"].map(label_map).to_numpy(dtype=int)

        # Standardize data (fit on train)
        X_train_std, X_test_std, mu, sigma = normalize_train_test(X_train, X_test)

        st.write(f"Training samples: {len(y_train)}, Testing samples: {len(y_test)}")
        st.success("Data prepared successfully!")

        # -----------------------------------------------------------
        # ✅ STEP 2 — Initialize weights
        # -----------------------------------------------------------
        # • Initialize random weights and bias (if enabled).
        # • Use a random seed for reproducibility.
        # TODO: Implement Step 2
        n_features = 2  # always two because the UI enforces selecting exactly 2 features
        weights, bias = initialize_weights(n_features=n_features,
                                        use_bias=use_bias,
                                        seed=seed,
                                        scale=0.01)

        # Display initialized parameters
        st.write(f"**Weights shape:** {weights.shape}")
        st.write(f"**Initial weights:** {np.round(weights, 6).tolist()}")
        st.write(f"**Bias:** {round(bias, 6) if use_bias else 'None (disabled)'}")
        
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
        st.subheader("Step 4 – Testing & Evaluation")

        # Temporary dummy weights for demo (to be replaced after training)
        # Use w_init, b_init for now to keep consistent structure
        y_pred = np.where(np.dot(X_test_std, weights) + (bias if use_bias else 0) >= 0, 1, -1)

        # Confusion matrix + accuracy
        cm = compute_confusion(y_test, y_pred, pos_label=1)
        accuracy = (cm["TP"] + cm["TN"]) / len(y_test) * 100

        st.write("Confusion matrix:")
        st.json(cm)
        st.write(f"Accuracy (with initial weights): **{accuracy:.2f}%**")

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

