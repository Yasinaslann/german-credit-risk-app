import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Sayfa yapılandırması
st.set_page_config(
    page_title="🛡️ German Credit Risk Tahmin Uygulaması",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Dosya yolu
BASE_DIR = os.path.dirname(__file__)

# Sayfa stil ayarları (arka plan, renkler, butonlar vs.)
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
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #341f97;
    color: #dfe6e9;
    cursor: pointer;
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

# Başlık
st.title("🛡️ German Credit Risk Tahmin Uygulaması")
st.markdown(
    """
    Bu uygulama, Almanya kredi veri seti kullanılarak geliştirilmiş **Random Forest** modeli ile kredi riskinizi tahmin eder.
    Sağdaki panelden başvuru bilgilerinizi girip **Tahmin Et** butonuna basınız.
    """
)

# Cache'li dosya yükleme fonksiyonu
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

# Sidebar - Kullanıcıdan input alma
st.sidebar.header("Kredi Başvuru Bilgileri")

age = st.sidebar.slider('Yaş', 18, 100, 30, help="Başvuru sahibinin yaşı")
credit_amount = st.sidebar.slider('Kredi Miktarı (€)', 100, 100000, 1000, step=100, help="Talep edilen kredi miktarı")
duration = st.sidebar.slider('Kredi Süresi (Ay)', 1, 60, 12, help="Kredi geri ödeme süresi")

sex_label = st.sidebar.selectbox('Cinsiyet', le_sex.classes_)
housing_label = st.sidebar.selectbox('Konut Durumu', le_housing.classes_)
saving_label = st.sidebar.selectbox('Tasarruf Hesabı', le_saving.classes_)
checking_label = st.sidebar.selectbox('Vadesiz Hesap', le_checking.classes_)
purpose_label = st.sidebar.selectbox('Kredi Amacı', le_purpose.classes_)

# Encoding
sex_encoded = le_sex.transform([sex_label])[0]
housing_encoded = le_housing.transform([housing_label])[0]
saving_encoded = le_saving.transform([saving_label])[0]
checking_encoded = le_checking.transform([checking_label])[0]
purpose_encoded = le_purpose.transform([purpose_label])[0]

# Butona basılınca tahmin et
if st.sidebar.button("Tahmin Et"):
    input_data = {
        'Age': age,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Sex_encoded': sex_encoded,
        'Housing_encoded': housing_encoded,
        'Saving_encoded': saving_encoded,
        'Checking_encoded': checking_encoded,
        'Purpose_encoded': purpose_encoded
    }
    input_df = pd.DataFrame([input_data])

    # Ölçeklendirme
    input_df[['Age', 'Credit amount', 'Duration']] = scaler.transform(input_df[['Age', 'Credit amount', 'Duration']])

    # Tahmin ve olasılık
    prediction = model.predict(input_df[feature_cols])[0]
    proba = model.predict_proba(input_df[feature_cols])[0][prediction]
    risk_map = {0: "İyi Risk ✅", 1: "Kötü Risk ⚠️"}

    # Sonucu göster
    st.markdown(f"## Tahmin Sonucu: **{risk_map[prediction]}**")
    st.write(f"Model Güven Skoru: **{proba:.2f}**")

    # Model performansını göster (örnek metrikler, kendi test verinle güncelle)
    st.markdown("---")
    st.subheader("Model Performans Metrikleri")
    accuracy = 0.82
    precision = 0.74
    recall = 0.68
    f1 = 0.70
    st.markdown(f"""
    - **Doğruluk (Accuracy):** {accuracy:.2f}  
    - **Kesinlik (Precision):** {precision:.2f}  
    - **Duyarlılık (Recall):** {recall:.2f}  
    - **F1 Skoru:** {f1:.2f}
    """)

    # Confusion Matrix çizimi
    cm = np.array([[80, 15],
                   [10, 25]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Feature Importance
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({"Özellik": feature_cols, "Önem": importances}).sort_values(by="Önem", ascending=False)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Önem", y="Özellik", data=feat_imp_df, palette="viridis", ax=ax2)
    ax2.set_title("Özelliklerin Modeldeki Önemi")
    st.pyplot(fig2)

# Uygulama hakkında detay
with st.expander("ℹ️ Veri Seti ve Model Hakkında"):
    st.markdown("""
    - Veri seti, Almanya’daki kredi başvurularına ait finansal ve demografik bilgileri içerir.
    - Model, Random Forest algoritması ile SMOTE yöntemi kullanılarak dengelenmiş veri üzerinde eğitildi.
    - Modelin amacı, bireyin kredi riskini önceden tahmin etmektir.
    - Performans metrikleri ve grafiklerle modelin doğruluğu ve güvenilirliği sunulmaktadır.
    """)

st.markdown("---")
st.caption("Created by Yasin Aslan | Powered by Streamlit & Scikit-learn")
