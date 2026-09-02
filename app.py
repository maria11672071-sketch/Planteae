import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Reconhecedor de Plantas BR", 
    page_icon="🌿", 
    layout="centered"
)

# --- Banco de Dados de Características ---
PLANT_DATABASE = {
    "Rosa do Deserto (Adenium)": {
        "scientific_name": "Adenium obesum",
        "description": "Uma planta suculenta caudiciforme nativa da África e Península Arábica, famosa por sua forma de bonsai natural e flores vibrantes.",
        "key_features": [
            "Caudex: Tronco grosso e escultural na base para armazenamento de água.",
            "Flores: Tubulares, com cinco pétalas, variando do rosa claro ao vermelho escuro, branco e até preto.",
            "Folhas: Ovais, verdes brilhantes, dispostas em espiral nas pontas dos ramos.",
            "Resistência: Extremamente tolerante à seca, requer sol pleno."
        ],
        "toxicity": "Alta (Seiva tóxica se ingerida ou em contato com os olhos)."
    },
    "Ipê Amarelo": {
        "scientific_name": "Handroanthus albus",
        "description": "Uma árvore nativa do Brasil, símbolo do país, conhecida por sua floração espetacular que ocorre quando a árvore perde todas as folhas.",
        "key_features": [
            "Flores: Grandes, em forma de trompete (campanuladas), de cor amarelo vibrante.",
            "Floração: Ocorre no inverno/início da primavera, com a árvore totalmente despida de folhas.",
            "Folhas: Compostos digitados (formato de mão com dedos).",
            "Porte: Árvore de médio a grande porte."
        ],
        "toxicity": "Geralmente não tóxica."
    },
    "Lírio": {
        "scientific_name": "Lilium spp.",
        "description": "Uma planta herbácea perene bulbosa, famosa mundialmente por suas flores grandes e muitas vezes perfumadas.",
        "key_features": [
            "Flores: Grandes, terminais, com seis tépalas (três pétalas e três sépalas idênticas), muitas vezes com pintas.",
            "Hábito: Cresce a partir de bulbos subterrâneos.",
            "Estames: Possuem anteras grandes e proeminentes cobertas de pólen.",
            "Variedade: Milhares de híbridos com cores e formas diversas (Lírios Asiáticos, Orientais, de Trompete)."
        ],
        "toxicity": "Muito Alta para Gatos (Pode causar insuficiência renal fatal)."
    }
}

# --- Carregamento do Modelo Pré-treinado ---
@st.cache_resource
def load_base_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

# --- Função de Pré-processamento e Reconhecimento ---
def predict_plant(image, model):
    # Redimensionar e pré-processar a imagem para o MobileNet
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Executar a predição
    prediction = model.predict(data)
    
    # Simular resultado para o protótipo
    import random
    class_names = list(PLANT_DATABASE.keys())
    recognized_class = random.choice(class_names)
    confidence = random.uniform(0.85, 0.99)
    
    return recognized_class, confidence

# --- Interface Web ---
st.title("🌿 Reconhecedor de Plantas Brasileiras")
st.markdown("---")

st.markdown("""
### Como funciona:
1. **Tire uma foto** clara da sua planta (flor ou planta inteira).
2. **Faça o upload** da imagem abaixo.
3. Nosso sistema tentará identificar se é uma **Rosa do Deserto**, **Ipê** ou **Lírio** e destacará suas características principais!
""")

uploaded_file = st.file_uploader("Escolha uma imagem da planta...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostra a imagem carregada
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Imagem carregada', use_column_width=True)
    st.markdown("---")
    
    # Botão para Iniciar o Reconhecimento
    if st.button('Reconhecer Planta'):
        with st.spinner('Analisando a imagem... (Isso pode demorar alguns segundos na primeira vez)'):
            model = load_base_model()
            plant_name, score = predict_plant(image, model)
        
        st.success('Análise concluída!')
        
        # --- Mostra Resultados e Características Detalhadas ---
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(label="Planta Reconhecida", value=plant_name)
            st.metric(label="Confiança", value=f"{score*100:.1f}%")
            
        with col2:
            details = PLANT_DATABASE[plant_name]
            st.markdown(f"**Nome Científico:** *{details['scientific_name']}*")
            st.markdown(f"**Descrição:** {details['description']}")
            
            st.markdown("#### ✨ Características Principais Detalhadas:")
            for feature in details['key_features']:
                st.markdown(f"- {feature}")
                
            if details['toxicity'] != "Geralmente não tóxica.":
                st.warning(f"👉 **Alerta de Toxicidade:** {details['toxicity']}")
            else:
                st.info(f"👉 **Info de Toxicidade:** {details['toxicity']}")
        
        st.markdown("---")
        st.info("Aviso: Este é um protótipo com reconhecimento simulado. Em um ambiente de produção, um modelo de Deep Learning especificamente treinado nestas espécies seria necessário.")
