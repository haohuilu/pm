import numpy as np
import streamlit as st


from models.NaiveBayes import nb_param_selector
from models.NeuralNetwork import nn_param_selector
from models.RandomForet import rf_param_selector
from models.DecisionTree import dt_param_selector
from models.LogisticRegression import lr_param_selector
from models.KNearesNeighbors import knn_param_selector
from models.SVC import svc_param_selector
from models.GradientBoosting import gb_param_selector

from models.utils import model_imports
from utils.functions import img_to_bytes


def introduction():
    st.title("**Welcome to playground for project analytics**")
    st.subheader(
        """
        This is a place where we can get familiar with machine learning models directly from the browser
        """
    )

    st.markdown(
        """
    - 🗂️ We use the road construction deley survey as an example (as above). Question 1 - 44 are independent variables (features) and "How often do you experience cost overrun" is dependent variable.
    - ✅ We would like to predict the risk of cost overrun (1 or 2)
    - ⚙️ Pick a model and set its hyper-parameters
    - 📉 Train it (use 50% of the dataset) and check its performance metrics on test data (50% of the dataset)
    - ✉️ If you have any question, please feel free to email me at haohui.lu@sydney.edu.au
    -----
    """

    )


def model_selector():
    model_training_container = st.sidebar.beta_expander("Train a model", True)
    with model_training_container:
        model_type = st.selectbox(
            "Choose a model",
            (
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "K Nearest Neighbors",
                "Support Vector Machine"
            ),
        )

        if model_type == "Logistic Regression":
            model = lr_param_selector()

        elif model_type == "Decision Tree":
            model = dt_param_selector()

        elif model_type == "Random Forest":
            model = rf_param_selector()

        elif model_type == "K Nearest Neighbors":
            model = knn_param_selector()
        elif model_type == "Support Vector Machine":
            model = svc_param_selector()



    return model_type, model


def generate_snippet(
    model, model_type, 
):


    model_text_rep = repr(model)
    model_import = model_imports[model_type]


    snippet = f"""
    >>> {model_import}
    >>> from sklearn.metrics import accuracy_score, f1_score


    >>> model.fit(x_train, y_train)
    
    >>> y_train_pred = model.predict(x_train)
    >>> y_test_pred = model.predict(x_test)
    >>> train_accuracy = accuracy_score(y_train, y_train_pred)
    >>> test_accuracy = accuracy_score(y_test, y_test_pred)
    """
    return snippet




def footer():
    st.sidebar.markdown("---")
