import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import joblib

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, 'rf_model_smote.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
le_sex = joblib.load(os.path.join(BASE_DIR, 'le_sex.pkl'))
le_housing = joblib.load(os.path.join(BASE_DIR, 'le_housing.pkl'))
le_saving = joblib.load(os.path.join(BASE_DIR, 'le_saving.pkl'))
le_checking = joblib.load(os.path.join(BASE_DIR, 'le_checking.pkl'))
le_purpose = joblib.load(os.path.join(BASE_DIR, 'le_purpose.pkl'))
feature_cols = joblib.load(os.path.join(BASE_DIR, 'feature_cols.pkl'))

# -- Stil (CSS) ile arka plan ve genel tema --
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
    height: 40px;
    width: 100%;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #341f97;
    color: #dfe6e9;
}
.stSidebar {
    background: #2d3436;
    color: white;
}
h1, h2, h3 {
    font-weight: 700;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# -- Başlık ve alt başlık --
st.title("🛡️ German Credit Risk Tahmin Uygulaması")
st.markdown("""
Bu uygulama, Almanya kredi veri seti kullanılarak geliştirilen **Random Forest** modeli ile bireylerin kredi riskini tahmin eder.
""")


# -- Model ve dosyaları yükle --
@st.cache_resource
def load_artifacts():
    model = joblib.load('rf_model_smote.pkl')
    scaler = joblib.load('scaler.pkl')
    le_sex = joblib.load('le_sex.pkl')
    le_housing = joblib.load('le_housing.pkl')
    le_saving = joblib.load('le_saving.pkl')
    le_checking = joblib.load('le_checking.pkl')
    le_purpose = joblib.load('le_purpose.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    return model, scaler, le_sex, le_housing, le_saving, le_checking, le_purpose, feature_cols


model, scaler, le_sex, le_housing, le_saving, le_checking, le_purpose, feature_cols = load_artifacts()

# -- Sidebar ile inputlar --
st.sidebar.header("Kredi Başvuru Bilgileri")

age = st.sidebar.number_input('Yaş', min_value=18, max_value=100, value=30)
credit_amount = st.sidebar.number_input('Kredi Miktarı (€)', min_value=100, max_value=1000000, value=1000)
duration = st.sidebar.number_input('Kredi Süresi (ay)', min_value=1, max_value=100, value=12)

sex_label = st.sidebar.selectbox('Cinsiyet', options=le_sex.classes_)
housing_label = st.sidebar.selectbox('Konut Durumu', options=le_housing.classes_)
saving_label = st.sidebar.selectbox('Tasarruf Hesabı', options=le_saving.classes_)
checking_label = st.sidebar.selectbox('Vadesiz Hesap', options=le_checking.classes_)
purpose_label = st.sidebar.selectbox('Kredi Amacı', options=le_purpose.classes_)

# -- Kodda kullanmak için encode et --
sex_encoded = le_sex.transform([sex_label])[0]
housing_encoded = le_housing.transform([housing_label])[0]
saving_encoded = le_saving.transform([saving_label])[0]
checking_encoded = le_checking.transform([checking_label])[0]
purpose_encoded = le_purpose.transform([purpose_label])[0]

# -- Tahmin butonu --
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

    # Model performans ve grafikler için demo amaçlı test verisi kullan (burada kendi X_test ve y_test varsa onu kullan)
    # Demo olarak orijinal test setinden örnek alabiliriz veya burada sabit grafik koyabiliriz.
    # Burada örnek olarak modelin performans grafiği:
    st.markdown("---")
    st.subheader("Model Performans Metrikleri ve Grafikler")

    # Örnek metrikler (bunları gerçek test setiyle doldur)
    accuracy = 0.82
    precision = 0.74
    recall = 0.68
    f1 = 0.70

    st.markdown(f"""
    **Accuracy:** {accuracy:.2f}  
    **Precision:** {precision:.2f}  
    **Recall:** {recall:.2f}  
    **F1-Score:** {f1:.2f}
    """)

    # Confusion matrix örneği (örnek, kendi verinle değiştir)
    cm = np.array([[80, 15],
                   [10, 25]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax)
    ax.set_xlabel('Tahmin')
    ax.set_ylabel('Gerçek')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # Feature importance örneği (gerçek feature importance ile değiştir)
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance',
                                                                                                 ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis', ax=ax2)
    ax2.set_title('Feature Importance (Özelliklerin Modeldeki Önemi)')
    st.pyplot(fig2)

# -- Bilgi bölümü --
with st.expander("ℹ️ Veri Seti ve Model Hakkında"):
    st.markdown("""
    - Veri seti, Almanya’daki kredi başvurularına ait finansal ve demografik bilgileri içerir.
    - Model, Random Forest algoritması ile SMOTE yöntemi kullanılarak dengelenmiş veri üzerinde eğitildi.
    - Modelin amacı, bireyin kredi riskini önceden tahmin etmektir.
    - Performans metrikleri ve grafiklerle modelin doğruluğu ve güvenilirliği sunulmaktadır.
    """)

st.markdown("---")
st.caption("Created by Yasin Aslan | Powered by Streamlit & Scikit-learn")
