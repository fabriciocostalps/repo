import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np

@st.cache_resource
def carrega_modelo():
    url = 'https://drive.google.com/uc?id=1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp'
    gdown.download(url, 'modelo_quantizado16bits.tflite') 
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()
    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader("Arraste e solte uma imagem ou clique para selecionar", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        st.image(image)
        st.success('Imagem carregada com sucesso!')
        
        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    return None

def main():
    st.set_page_config(page_title="Classificador de folhas de videiras")
    st.write("# Classificador de folhas de videiras")

    interpreter = carrega_modelo()
    imagem = carrega_imagem()
    
    if imagem is not None:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        if imagem.shape[1:3] != tuple(input_details[0]['shape'][1:3]):
            st.warning(f"A imagem precisa ser redimensionada para {input_details[0]['shape'][1:3]}")
            return
            
        interpreter.set_tensor(input_details[0]['index'], imagem)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        st.write("Resultado da classificação:", output_data)

if __name__ == "__main__":
    main()
