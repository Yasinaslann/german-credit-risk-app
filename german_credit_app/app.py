import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Sayfa ayarları
st.set_page_config(page_title="German Credit Risk Tahmin", layout="wide", page_icon="🛡️")

# Stil (CSS) - arka plan ve genel tema
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

# Logo gösterimi - 2 seçenek:

# 1) Eğer logo dosyan varsa burayı aç (logo.png dosyasını projeye koymalısın)
# st.sidebar.image("logo.png", width=150)

# 2) Ya da internetten logo URL ile göster
st.sidebar.image("https://raw.githubusercontent.com/yasinaslann/german-credit-risk-app/main/logo.png", width=150)

# Dosya yolları (Streamlit Cloud ve local uyumlu)
BASE_DIR = os.path.dirname(__file__)

# Model ve encoders yükleme fonksiyonu
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

# Veri seti yükleme (EDA için)
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'german_credit_data.csv'))
    return df

df = load_data()

# Başlık
st.title("🛡️ German Credit Risk Tahmin Uygulaması")
st.markdown("""
Bu uygulama, Almanya kredi veri seti kullanılarak geliştirilmiş **Random Forest** modeli ile bireylerin kredi riskini tahmin eder.
""")

# Sidebar inputlar
st.sidebar.header("Kredi Başvuru Bilgileri")

age = st.sidebar.slider('Yaş', 18, 100, 30)
credit_amount = st.sidebar.slider('Kredi Miktarı (€)', 100, 1000000, 1000, step=100)
duration = st.sidebar.slider('Kredi Süresi (ay)', 1, 100, 12)

sex_label = st.sidebar.selectbox('Cinsiyet', options=le_sex.classes_)
housing_label = st.sidebar.selectbox('Konut Durumu', options=le_housing.classes_)
saving_label = st.sidebar.selectbox('Tasarruf Hesabı', options=le_saving.classes_)
checking_label = st.sidebar.selectbox('Vadesiz Hesap', options=le_checking.classes_)
purpose_label = st.sidebar.selectbox('Kredi Amacı', options=le_purpose.classes_)

# Encode et
sex_encoded = le_sex.transform([sex_label])[0]
housing_encoded = le_housing.transform([housing_label])[0]
saving_encoded = le_saving.transform([saving_label])[0]
checking_encoded = le_checking.transform([checking_label])[0]
purpose_encoded = le_purpose.transform([purpose_label])[0]

# Tahmin butonu
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

    # Ölçeklendir
    input_df[['Age', 'Credit amount', 'Duration']] = scaler.transform(input_df[['Age', 'Credit amount', 'Duration']])

    # Tahmin
    pred = model.predict(input_df[feature_cols])[0]
    proba = model.predict_proba(input_df[feature_cols])[0][pred]

    risk_map = {0: 'Good Risk ✅', 1: 'Bad Risk ⚠️'}
    st.markdown(f"### Tahmin Sonucu: {risk_map[pred]}")
    st.write(f"Model Güven Skoru: **{proba:.2f}**")

# Veri Keşfi (EDA) bölümü
with st.expander("📊 Veri Seti Keşfi ve İstatistikler", expanded=True):
    st.markdown("### Sayısal Değişkenlerin Dağılımı")
    numeric_cols = ['Age', 'Credit amount', 'Duration']
    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df['Age'], kde=True, color='#6c5ce7', ax=ax)
        ax.set_title("Yaş Dağılımı")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(df['Credit amount'], kde=True, color='#6c5ce7', ax=ax)
        ax.set_title("Kredi Miktarı Dağılımı")
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots()
        sns.histplot(df['Duration'], kde=True, color='#6c5ce7', ax=ax)
        ax.set_title("Kredi Süresi Dağılımı")
        st.pyplot(fig)

    st.markdown("### Kategorik Değişkenlerin Dağılımı")
    cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    for col in cat_cols:
        fig, ax = plt.subplots()
        df[col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, startangle=90, colors=sns.color_palette("viridis"))
        ax.set_ylabel('')
        ax.set_title(f"{col} Dağılımı")
        st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Created by Yasin Aslan | Powered by Streamlit & Scikit-learn")
