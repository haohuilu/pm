import numpy as np
import streamlit as st
from utils.functions import (
    get_model_tips,
    get_model_url,    train_model,
    plot_decision_boundary_and_metrics,
)

from utils.ui import (
    footer,
    generate_snippet,
    introduction,
    model_selector,
)

st.set_page_config(
    page_title="Playground for project analytics", layout="wide", page_icon="./images/flask.png"
)

import pandas as pd
import streamlit as st

df = pd.read_csv("pm.csv",index_col = 0) # read a CSV file inside the 'data" folder next to 'app.py'
# df = pd.read_excel(...)  # will work for Excel files
df = df.dropna()
df = df.drop(df[df['How often do you experience cost overrun']==3].index)
st.write(df)  # visualize my dataframe in the Streamlit app

X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
from sklearn import model_selection

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    X,y, train_size=0.5, test_size=None, random_state = 2
)



def sidebar_controllers():
    model_type, model = model_selector()
    footer()

    return (

        model_type,
        model

    )


def body(model, model_type, 
):
    introduction()
    col1, col2 = st.beta_columns((1, 1))

    with col1:
        plot_placeholder = st.empty()

    with col2:
        duration_placeholder = st.empty()
        model_url_placeholder = st.empty()
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        tips_header_placeholder = st.empty()
        tips_placeholder = st.empty()

    model_url = get_model_url(model_type)

    (
        model,
        train_accuracy,
        train_f1,
        test_accuracy,
        test_f1,
        duration,
    ) = train_model(model, x_train, y_train, x_test, y_test)

    metrics = {
        "train_accuracy": train_accuracy,
        "train_f1": train_f1,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
    }

    snippet = generate_snippet(
        model, model_type
    )

    model_tips = get_model_tips(model_type)
    
    fig = plot_decision_boundary_and_metrics(
        model, x_train, y_train, x_test, y_test, metrics
    )

    plot_placeholder.plotly_chart(fig, True)

    duration_placeholder.warning(f"Training took {duration:.3f} seconds")
    model_url_placeholder.markdown(model_url)
    #code_header_placeholder.header("**Retrain the same model in Python**")
    #snippet_placeholder.code(snippet)
    tips_header_placeholder.header(f"**Tips on the {model_type} 💡 **")
    tips_placeholder.info(model_tips)



if __name__ == "__main__":
    (

        model_type,
        model,
    ) = sidebar_controllers()
    body(

        model,
        model_type,

    )
