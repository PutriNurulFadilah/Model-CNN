import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model  # Use tensorflow.keras.models instead of keras.models

def load_model():
    # Use an absolute path to the model
    path_model = r'C:\Model H5\best_model.h5'
    model = load_model(path_model)
    return model

# Function to process the image and make a prediction
def predict_disease(model, img_array):
    img_array = tf.image.resize(img_array, (150, 150))  # Removed the third dimension in the resize function
    img_array = tf.image.rgb_to_grayscale(img_array)
    img_array = tf.image.grayscale_to_rgb(img_array)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    return prediction

# Function to recommend treatment based on the disease
def recommend_treatment(disease):
    if disease == "Coccidiosis":
        return """
        - Vaksinasi coccidiosis
        - Perhatikan kondisi dan kebersihan kandang
        - Gunakan Koksidiostat, Ssulfadimetoksin, dan antibiotik (tetrasiklin, eritromisin, spektinomisin, dan tilosin)
        - Lakukan terapi antioksidan
        """
    elif disease == "New Castle Disease":
        return """
        - Ayam yang tertular harus dengan cepat dikarantina
        - Jika terlalu parah, ayam harus dimusnahkan untuk menghindari penularan
        - Vaksinasi melalui tetes mata
        """
    elif disease == "Salmonella":
        return """
        - Desinfeksi kandang dari salmonela
        - Rutin membersihkan tempat makan ayam
        - Berikan Antibiotik seperti amoksisilin dan kolistin
        """
    else:
        return "Rekomendasi pengobatan tidak tersedia."

# Main function for Streamlit display
def main():
    # Add title and background with image from Pexels
    st.title("Chicken Disease Classification")
    bg_image_url = "https://www.pexels.com/id-id/foto/empat-ayam-jantan-aneka-warna-1769279/"

    # Display image from URL
    st.markdown(
        f"""
        <style>
            body {{
                background-image: url('{bg_image_url}');
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add sidebar
    st.sidebar.title("Upload Chicken Feces Image")

    # Upload image from user
    uploaded_file = st.sidebar.file_uploader("Choose a chicken feces image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Chicken Feces Image.", use_column_width=True)

        # Load the model
        model = load_model()

        if model is not None:
            # Convert the image to an array
            img_array = np.array(image)

            # Make a prediction
            prediction = predict_disease(model, img_array)

            # Get the disease name based on the prediction
            disease_mapping = {0: "Coccidiosis", 1: "New Castle Disease", 2: "Salmonella"}
            predicted_disease = disease_mapping[np.argmax(prediction)]

            # Display the prediction result
            st.write("### Prediction:")
            st.write(f"The predicted disease class is: {predicted_disease}")

            # Provide treatment recommendation
            st.write("### Treatment Recommendation:")
            treatment_recommendation = recommend_treatment(predicted_disease)
            st.write(treatment_recommendation)
