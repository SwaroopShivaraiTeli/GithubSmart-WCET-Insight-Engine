import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart WCET Insight Engine", layout="wide")
st.title("Smart WCET Insight Engine")
st.markdown("🔍 Predict `loopQty` and WCET with static code metrics + SHAP explanations.")

# Load models
loop_model = joblib.load("dt_loopQty_model.pkl")
wcet_model = joblib.load("dt_wcet_model.pkl")

# Define correct feature columns used for WCET prediction
wcet_features = [
    'refactoring', 'cbo', 'cboModified', 'fanin', 'fanout', 'wmc', 'dit', 'noc', 'rfc',
    'lcom', 'lcom*', 'tcc', 'lcc', 'totalMethodsQty', 'staticMethodsQty', 'publicMethodsQty',
    'privateMethodsQty', 'protectedMethodsQty', 'defaultMethodsQty', 'visibleMethodsQty',
    'abstractMethodsQty', 'finalMethodsQty', 'synchronizedMethodsQty', 'totalFieldsQty',
    'staticFieldsQty', 'publicFieldsQty', 'privateFieldsQty', 'protectedFieldsQty',
    'defaultFieldsQty', 'finalFieldsQty', 'synchronizedFieldsQty', 'nosi', 'loc', 'returnQty',
    'loopQty', 'comparisonsQty', 'tryCatchQty', 'parenthesizedExpsQty', 'stringLiteralsQty',
    'numbersQty', 'assignmentsQty', 'mathOperationsQty', 'variablesQty', 'maxNestedBlocksQty',
    'anonymousClassesQty', 'innerClassesQty', 'lambdasQty', 'uniqueWordsQty', 'modifiers',
    'logStatementsQty'
]

uploaded_file = st.file_uploader("Upload your code metrics CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(5))

    # Drop non-feature columns
    df_model = df.copy()
    for col in ['file', 'class', 'type', 'WCET']:
        if col in df_model.columns:
            df_model.drop(columns=col, inplace=True)

    # Fill missing values
    for col in ['lcom*', 'tcc', 'lcc']:
        if col in df_model.columns:
            df_model[col].fillna(df_model[col].median(), inplace=True)

    # === STAGE 1: Predict loopQty ===
    loop_features = df_model.drop(columns=['loopQty'], errors='ignore')
    loop_preds = loop_model.predict(loop_features)
    df['Predicted_LoopQty'] = loop_preds
    df_model['loopQty'] = loop_preds  # For WCET model

    # === STAGE 2: Predict WCET ===
    df_wcet = df_model[wcet_features]  # Exact match to trained model
    wcet_preds = wcet_model.predict(df_wcet)
    df['Predicted_WCET'] = wcet_preds

    st.subheader("Predictions")
    st.dataframe(df[['Predicted_LoopQty', 'Predicted_WCET']].head(10))

    csv_out = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results", csv_out, "wcet_predictions.csv", "text/csv")

    # === Explainability ===
    st.subheader("Explainability (SHAP)")
    explainer = shap.Explainer(loop_model, loop_features)
    shap_values = explainer(loop_features)

    st.markdown("**Global Impact - Top Features (Beeswarm)**")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    st.pyplot(fig1)

    st.markdown("**Waterfall Plot for Most Complex Prediction**")
    idx = np.argmax(loop_preds)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_values[idx], show=False)
    st.pyplot(fig2)

else:
    st.info("👆 Upload a valid CSV to begin.")
