import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

@st.cache_resource
def carrega_modelo():
    # https://drive.google.com/file/d/1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp/view?usp=sharing
    url = 'https://drive.google.com/uc?id=1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp'

    # baixa o arquivo
    gdown.download(url, 'modelo_quantizado16bits.tflite') 

    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')

    # disponibiliza para uso
    interpreter.allocate_tensors()  # Corrigido para o método correto

    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader("Arraste e solte a uma imagem ou clique para selecionar uma", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()

        # abrir a imagem
        image = Image.open(io.BytesIO(image_data))  # Corrigido Image e BytesIO

        # exibir a imagem na página
        st.image(image)
        st.success('Imagem foi carregada com sucesso!')  # Corrigido st.success

        # converter a imagem em ponto flutuante
        image = np.array(image, dtype=np.float32)

        # normalizar a imagem
        image = image / 255.0

        # adicionar uma dimensão extra
        image = np.expand_dims(image, axis=0)

        return image
    return None  # Retorna None se nenhuma imagem for carregada

def main():
    st.set_page_config(
        page_title="Classifica folhas de videiras!"
    )
    st.write("# Classifica folhas de videiras!")

    # Carregar o modelo
    interpreter = carrega_modelo()

    # Carregar a imagem
    imagem = carrega_imagem()
    
    if imagem is not None:
        # Aqui você precisará adicionar a lógica para classificar a imagem
        # usando o interpretador do modelo
        st.write("Imagem pronta para classificação!")
        # Exemplo:
        # input_details = interpreter.get_input_details()
        # interpreter.set_tensor(input_details[0]['index'], imagem)
        # interpreter.invoke()
        # output_details = interpreter.get_output_details()
        # output_data = interpreter.get_tensor(output_details[0]['index'])
        # st.write(f"Resultado: {output_data}")

if __name__ == "__main__":
    main()
Observações adicionais: