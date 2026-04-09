from io import BytesIO
from pathlib import Path

import numpy as np
import plotly.express as px
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


IMG_HEIGHT = 180
IMG_WIDTH = 180
CLASS_NAMES = ["Margaritas", "Diente de Leon", "Rosas", "Girasoles", "Tulipanes"]
MODEL_PATH = Path(__file__).with_name("flores.keras")


st.set_page_config(
    page_title="Clasificador de Flores con IA",
    page_icon="🌼",
    layout="wide",
)


@st.cache_resource
def load_classifier() -> tf.keras.Model:
    """Load the trained Keras model once per session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontro el archivo flores.keras en la misma carpeta de app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def load_image_from_upload(uploaded_file) -> Image.Image:
    """Read an uploaded image and convert it to RGB."""
    return Image.open(uploaded_file).convert("RGB")


def load_image_from_url(image_url: str) -> Image.Image:
    """Download an image from a URL and convert it to RGB."""
    response = requests.get(image_url, timeout=15)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_image_from_camera(camera_file) -> Image.Image:
    """Read a camera snapshot and convert it to RGB."""
    return Image.open(camera_file).convert("RGB")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize image to the input size expected by the model and build a batch."""
    resized_image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(resized_image)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def predict_image(model: tf.keras.Model, image: Image.Image) -> tuple[str, float, np.ndarray]:
    """Run inference and return the predicted class, confidence and all scores."""
    input_batch = preprocess_image(image)
    predictions = model.predict(input_batch, verbose=0)
    scores = tf.nn.softmax(predictions[0]).numpy()
    best_index = int(np.argmax(scores))
    best_label = CLASS_NAMES[best_index]
    best_score = float(scores[best_index]) * 100
    return best_label, best_score, scores


def render_probability_chart(scores: np.ndarray) -> None:
    """Render a comparative probability bar chart."""
    figure = px.bar(
        x=CLASS_NAMES,
        y=(scores * 100).round(2),
        labels={"x": "Tipo de flor", "y": "Probabilidad (%)"},
        color=CLASS_NAMES,
        color_discrete_sequence=["#e85d75", "#f4a261", "#d62828", "#f6bd60", "#84a59d"],
        text_auto=".2f",
    )
    figure.update_layout(
        showlegend=False,
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(figure, use_container_width=True)


def get_selected_image() -> Image.Image | None:
    """Return the image selected by the user from one of the supported sources."""
    source = st.radio(
        "Selecciona la fuente de la imagen",
        ["Subir archivo", "Usar URL", "Tomar foto"],
        horizontal=True,
    )

    try:
        if source == "Subir archivo":
            uploaded_file = st.file_uploader(
                "Sube una imagen de una flor",
                type=["png", "jpg", "jpeg", "webp"],
            )
            if uploaded_file is not None:
                return load_image_from_upload(uploaded_file)

        if source == "Usar URL":
            image_url = st.text_input("Pega la URL de la imagen")
            if image_url:
                return load_image_from_url(image_url)

        if source == "Tomar foto":
            camera_file = st.camera_input("Toma una foto")
            if camera_file is not None:
                return load_image_from_camera(camera_file)

    except requests.RequestException:
        st.error("No fue posible descargar la imagen desde la URL proporcionada.")
    except UnidentifiedImageError:
        st.error("El archivo seleccionado no es una imagen valida.")
    except OSError:
        st.error("No se pudo abrir la imagen. Verifica el archivo o la URL.")

    return None


def render_header() -> None:
    """Draw the title and short description."""
    st.markdown(
        """
        <style>
            .main {
                background: linear-gradient(180deg, #fffdf7 0%, #fff7ed 100%);
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 980px;
            }
            .hero {
                text-align: center;
                padding: 1.5rem 1rem 2rem 1rem;
                border-radius: 24px;
                background: rgba(255, 255, 255, 0.78);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
                margin-bottom: 1.5rem;
            }
            .footer {
                text-align: center;
                color: #555;
                padding-top: 2rem;
                font-size: 0.95rem;
            }
        </style>
        <div class="hero">
            <h1>Clasificador de Flores con IA</h1>
            <p>
                Esta aplicacion usa inteligencia artificial para identificar flores a partir de
                una imagen cargada desde tu equipo, una URL o la camara.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Draw the footer text."""
    st.markdown(
        """
        <div class="footer">
            <div>Desarrollado por Alfredo Diaz</div>
            <div>UNAB © Derechos Reservados</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Streamlit entry point."""
    render_header()

    try:
        model = load_classifier()
    except Exception as exc:
        st.error(f"No se pudo cargar el modelo: {exc}")
        st.info("Coloca el archivo flores.keras en la misma carpeta que app.py.")
        return

    image = get_selected_image()

    if image is None:
        st.info("Carga una imagen para realizar la clasificacion.")
        render_footer()
        return

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("Imagen cargada")
        st.image(image, use_container_width=True)

    with right_col:
        with st.spinner("Analizando la imagen..."):
            predicted_label, probability, scores = predict_image(model, image)

        st.subheader("Resultado")
        st.success(
            f"Esta imagen probablemente pertenece a {predicted_label} con una probabilidad del {probability:.2f}%."
        )
        st.metric("Clase predicha", predicted_label)
        st.metric("Probabilidad", f"{probability:.2f}%")

    st.subheader("Comparativo de probabilidades")
    render_probability_chart(scores)
    render_footer()


if __name__ == "__main__":
    main()