import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="German Credit Risk Tahmin", layout="wide", page_icon="🛡️")

page_bg_img = '''
<style>
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stButton>button {
    background-color: #6c5ce7;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-weight: 700;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #341f97;
    color: #dfe6e9;
}
.stSidebar {
    background: #2d3436;
    color: white;
    padding: 20px;
}
h1, h2, h3 {
    font-weight: 700;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, 'rf_model_smote.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    le_sex = joblib.load(os.path.join(BASE_DIR, 'le_sex.pkl'))
    le_housing = joblib.load(os.path.join(BASE_DIR, 'le_housing.pkl'))
    le_saving = joblib.load(os.path.join(BASE_DIR, 'le_saving.pkl'))
    le_checking = joblib.load(os.path.join(BASE_DIR, 'le_checking.pkl'))
    le_purpose = joblib.load(os.path.join(BASE_DIR, 'le_purpose.pkl'))
    feature_cols = joblib.load(os.path.join(BASE_DIR, 'feature_cols.pkl'))
    return model, scaler, le_sex, le_housing, le_saving, le_checking, le_purpose, feature_cols

model, scaler, le_sex, le_housing, le_saving, le_checking, le_purpose, feature_cols = load_artifacts()

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'german_credit_data.csv'))
    return df

df = load_data()

st.title("🛡️ German Credit Risk Tahmin Uygulaması")
st.markdown("""
Bu uygulama, Almanya kredi veri seti kullanılarak geliştirilmiş **Random Forest** modeli ile bireylerin kredi riskini tahmin eder.
""")

# Sidebar inputs with help text
st.sidebar.header("Kredi Başvuru Bilgileri")

age = st.sidebar.slider('Yaş', 18, 100, 30, help="Kredi başvuru sahibinin yaşı")
credit_amount = st.sidebar.slider('Kredi Miktarı (€)', 100, 1000000, 1000, step=100, help="Talep edilen kredi miktarı")
duration = st.sidebar.slider('Kredi Süresi (ay)', 1, 100, 12, help="Kredi geri ödeme süresi ay cinsinden")

sex_label = st.sidebar.selectbox('Cinsiyet', options=le_sex.classes_, help="Başvuru sahibinin cinsiyeti")
housing_label = st.sidebar.selectbox('Konut Durumu', options=le_housing.classes_, help="Başvuru sahibinin konut durumu")
saving_label = st.sidebar.selectbox('Tasarruf Hesabı', options=le_saving.classes_, help="Başvuru sahibinin tasarruf hesabı durumu")
checking_label = st.sidebar.selectbox('Vadesiz Hesap', options=le_checking.classes_, help="Başvuru sahibinin vadesiz hesap durumu")
purpose_label = st.sidebar.selectbox('Kredi Amacı', options=le_purpose.classes_, help="Kredi talep sebebi")

sex_encoded = le_sex.transform([sex_label])[0]
housing_encoded = le_housing.transform([housing_label])[0]
saving_encoded = le_saving.transform([saving_label])[0]
checking_encoded = le_checking.transform([checking_label])[0]
purpose_encoded = le_purpose.transform([purpose_label])[0]

if st.sidebar.button('Tahmin Et'):
    input_dict = {
        'Age': age,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Sex_encoded': sex_encoded,
        'Housing_encoded': housing_encoded,
        'Saving_encoded': saving_encoded,
        'Checking_encoded': checking_encoded,
        'Purpose_encoded': purpose_encoded
    }
    input_df = pd.DataFrame([input_dict])
    input_df[['Age', 'Credit amount', 'Duration']] = scaler.transform(input_df[['Age', 'Credit amount', 'Duration']])
    pred = model.predict(input_df[feature_cols])[0]
    proba = model.predict_proba(input_df[feature_cols])[0][pred]

    risk_map = {0: 'Good Risk ✅', 1: 'Bad Risk ⚠️'}
    st.markdown(f"### Tahmin Sonucu: {risk_map[pred]}")
    st.write(f"Model Güven Skoru: **{proba:.2f}**")

with st.expander("📊 Veri Seti Keşfi ve İstatistikler", expanded=True):
    st.markdown("### Sayısal Değişkenlerin Dağılımı ve Özet İstatistikleri")
    numeric_cols = ['Age', 'Credit amount', 'Duration']
    col1, col2, col3 = st.columns(3)
    for col, c in zip(numeric_cols, [col1, col2, col3]):
        with c:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, color='#6c5ce7', ax=ax)
            ax.set_title(f"{col} Dağılımı")
            st.pyplot(fig)
            st.write(df[col].describe())

    st.markdown("### Kategorik Değişkenlerin Dağılımı")
    cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    for col in cat_cols:
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, palette="viridis", ax=ax)
        ax.set_title(f"{col} Dağılımı")
        ax.set_xlabel("")
        ax.set_ylabel("Sayı")
        st.pyplot(fig)

st.markdown("---")
st.caption("Created by Yasin Aslan | Powered by Streamlit & Scikit-learn")
