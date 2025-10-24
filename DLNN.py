# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
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




# -------------------------
# Algorithms Section 
# -------------------------

# TODO: Implement weight initialization (small random values, optional bias)

def initialize_weights(n_features, use_bias=True, seed=42, scale=0.01):
    np.random.seed(seed)
    weights = np.random.normal(0.0, scale, n_features)
    bias = np.random.normal(0.0, scale) if use_bias else 0.0
    return weights, bias
    

# TODO: Implement Perceptron training algorithm (update weights per sample, track errors per epoch)

# TODO: Implement Adaline training algorithm (update using error, track MSE per epoch)


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
        pass

        # -----------------------------------------------------------
        # ✅ STEP 5 — Plot decision boundary
        # -----------------------------------------------------------
        # • Plot the 2D decision boundary using the two selected features.
        # • Show training and test samples with different markers.
        # • Add labels and legend for both classes.
        # • Display the figure in Streamlit.
        # TODO: Implement Step 5
        def predict_linear(X, weights, bias=None):
            """
            X: (N,2) numpy array (standardized features)
            weights: np.array shape (2,) or (3,) if augmented (last element bias)
            bias: float or None (ignored if weights length==3)
            returns predictions in {-1, +1}
            """
            w = np.asarray(weights) 
            if w.ndim != 1:
                w = w.ravel()
            if w.size == 3:
                # augmented: last element is bias
                b = float(w[-1])
                w = w[:2]
            else:
                b = 0.0 if bias is None else float(bias)
            scores = X.dot(w) + b
            preds = np.where(scores >= 0.0, 1, -1)
            return preds, scores

        def plot_decision_boundary_streamlit(X_train, y_train, X_test, y_test, weights, bias,
                                            class_names, feature_names, show_test=True,
                                            grid_steps=200, ax_size=(6,6)):
            """
            X_train/X_test: standardized numpy arrays shape (N,2)
            y_train/y_test: labels in {-1, +1}
            weights: np.array (2,) or (3,) augmented
            bias: float or None
            class_names: tuple/list like (class_for_-1, class_for_+1)
            feature_names: tuple/list (feat1_name, feat2_name)
            """
            # Validate shapes
            X_train = np.asarray(X_train)
            X_test = np.asarray(X_test) if X_test is not None else None

            # compute grid bounds in standardized feature space
            all_X = X_train if X_test is None else np.vstack([X_train, X_test])
            margin = 0.6
            x_min, x_max = all_X[:,0].min() - margin, all_X[:,0].max() + margin
            y_min, y_max = all_X[:,1].min() - margin, all_X[:,1].max() + margin

            xx = np.linspace(x_min, x_max, grid_steps)
            yy = np.linspace(y_min, y_max, grid_steps)
            XX, YY = np.meshgrid(xx, yy)
            grid = np.column_stack([XX.ravel(), YY.ravel()])

            preds_grid, scores_grid = predict_linear(grid, weights, bias)
            Z = preds_grid.reshape(XX.shape)

            cmap_light = ListedColormap(['#FFEEEE', '#EEFFEE'])
            cmap_points = ListedColormap(['#DD4444', '#22AA22'])

            fig, ax = plt.subplots(figsize=ax_size)
            ax.contourf(XX, YY, Z, alpha=0.3, cmap=cmap_light)

            # plot training points
            for lab, marker, edgecolor in [(-1, 'o', 'k'), (1, 's', 'k')]:
                sel = (y_train == lab)
                ax.scatter(X_train[sel, 0], X_train[sel, 1],
                        marker=marker, s=60, label=f"Train: {class_names[0] if lab==-1 else class_names[1]}",
                        edgecolor=edgecolor, linewidth=0.6)

            # plot test points if available
            if show_test and (X_test is not None) and (y_test is not None):
                for lab, markerface in [(-1, 'none'), (1, 'none')]:
                    sel = (y_test == lab)
                    ax.scatter(X_test[sel, 0], X_test[sel, 1],
                            marker='x', s=60, label=f"Test: {class_names[0] if lab==-1 else class_names[1]}",
                            linewidth=1.2)

            # decision boundary line (optional): derive from w
            w = np.asarray(weights).ravel()
            if w.size == 3:
                b = float(w[-1]); w = w[:2]
            else:
                b = 0.0 if bias is None else float(bias)
            if abs(w[1]) > 1e-8:
                # x2 = -(w0/w1) x1 - b/w1
                x_vals = np.array([x_min, x_max])
                y_vals = -(w[0]/w[1]) * x_vals - (b / w[1])
                ax.plot(x_vals, y_vals, 'k--', linewidth=1.2, label='Decision boundary')
            else:
                # vertical boundary x = -b/w0
                if abs(w[0]) > 1e-8:
                    x0 = -b / w[0]
                    ax.axvline(x=x0, linestyle='--', color='k', label='Decision boundary')

            ax.set_xlabel(feature_names[0] + " (standardized)")
            ax.set_ylabel(feature_names[1] + " (standardized)")
            ax.legend(loc='upper left', fontsize='small', framealpha=0.9)
            ax.set_title("2D Decision Boundary — Linear Classifier")
            plt.tight_layout()
            return fig

        # -------------- usage inside run_button ----------------
        # Expected variables (from earlier steps):
        # X_train, y_train, X_test, y_test  -> numpy arrays with standardized features shape (N,2)
        # weights, bias                     -> initialized/trained weights (np.array) and bias float (or augmented weights)
        # classes                            -> list like [classA_name, classB_name] corresponding to -1, +1 mapping
        # selected_features                  -> list of two selected feature names (strings)
        # history                            -> training history dict/list (optional)

        # Defensive checks & fallbacks
        _missing = []
        if 'X_train' not in globals():
            _missing.append("X_train")
        if 'X_test' not in globals():
            _missing.append("X_test")
        if 'y_train' not in globals():
            _missing.append("y_train")
        if 'y_test' not in globals():
            _missing.append("y_test")
        if 'weights' not in globals():
            _missing.append("weights")

        if _missing:
            st.warning("Cannot plot decision boundary: missing variables: " + ", ".join(_missing))
        else:
            # Ensure numpy arrays
            X_train_arr = np.asarray(X_train)
            X_test_arr = np.asarray(X_test) if 'X_test' in globals() else None
            y_train_arr = np.asarray(y_train)
            y_test_arr = np.asarray(y_test) if 'y_test' in globals() else None

            # class names fallback: try classes variable else use mapping from sidebar selection
            if 'classes' in globals():
                class_names = classes
            else:
                # fallback uses class_pair from your sidebar selection where first->-1, second->+1
                class_names = tuple(class_pair) if 'class_pair' in globals() else ("Class -1", "Class +1")

            # feature names
            feat_names = tuple(selected_features) if 'selected_features' in globals() else ("Feature1", "Feature2")

            fig = plot_decision_boundary_streamlit(X_train_arr, y_train_arr, X_test_arr, y_test_arr,
                                                weights, bias, class_names, feat_names)
            st.pyplot(fig)

            # Also show simple 1D metric (accuracy) and confusion matrix if y_test present
            if (y_test_arr is not None) and (X_test_arr is not None):
                preds_test, _ = predict_linear(X_test_arr, weights, bias)
                # manual confusion matrix for positive class = +1 (second selected class)
                TP = int(np.sum((preds_test == 1) & (y_test_arr == 1)))
                TN = int(np.sum((preds_test == -1) & (y_test_arr == -1)))
                FP = int(np.sum((preds_test == 1) & (y_test_arr == -1)))
                FN = int(np.sum((preds_test == -1) & (y_test_arr == 1)))
                acc = (TP + TN) / (TP + TN + FP + FN) if (TP+TN+FP+FN)>0 else 0.0

                st.write("### Test evaluation")
                st.write(f"Accuracy: **{acc*100:.2f}%**")
                cm_df = pd.DataFrame([[TP, FP],[FN, TN]],
                                    index=[f"Pred {class_names[1]}", f"Pred {class_names[0]}"],
                                    columns=[f"Actual {class_names[1]}", f"Actual {class_names[0]}"])
                st.table(cm_df)

        # -----------------------------------------------------------
        # ✅ STEP 6 — Display notes and dataset preview
        # -----------------------------------------------------------
        # • Add notes about label mapping and preprocessing.
        # • Optionally show sample rows from the training and testing datasets.
        # TODO: Implement Step 6
        st.write("---")
        st.write("### Notes & dataset preview")

        # Note about label mapping and preprocessing
        st.markdown(
            """
            **Label mapping:**  
            - The first class you selected (left in sidebar) is mapped to **-1**.  
            - The second class is mapped to **+1**.  

            **Preprocessing:**  
            - Features were standardized using the *training set* mean and std (train mean/std used to transform test).  
            - The decision boundary plot shows features in **standardized** space (mean=0, std=1).  
            - If you want axis ticks in original units, we can invert the standardization (need train mean/std).
            """
        )

        # Show a few rows from train/test if available
        if 'train_df' in globals() and isinstance(train_df, pd.DataFrame):
            st.write("#### Sample rows from training set")
            st.dataframe(train_df.sample(min(10, len(train_df))).reset_index(drop=True))
        elif 'X_train' in globals() and 'y_train' in globals():
            # create a simple preview DataFrame from X_train/y_train (standardized)
            preview = pd.DataFrame(X_train_arr[:10, :], columns=[f"{feat_names[0]}_std", f"{feat_names[1]}_std"])
            preview['label'] = y_train_arr[:10]
            # convert labels to class names for readability
            preview['label_name'] = preview['label'].apply(lambda v: class_names[1] if v==1 else class_names[0])
            st.write("#### Sample rows from (standardized) training set")
            st.dataframe(preview)

        if 'test_df' in globals() and isinstance(test_df, pd.DataFrame):
            st.write("#### Sample rows from testing set")
            st.dataframe(test_df.sample(min(10, len(test_df))).reset_index(drop=True))

        st.write("### Quick tips")
        st.markdown("""
        - If the boundary looks strange (all points on one side), check that labels mapping (-1/+1) matches class order.
        - To show axes in original units, provide `train_mean` and `train_std` or `scaler` so we can inverse-transform ticks.
        - If you want a filled contour colored exactly by model score magnitude, I can add a colorbar with the raw scores.
        """)