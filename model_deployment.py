import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from pydub import AudioSegment
from pydub.playback import play
from skimage import exposure, io
from scipy import ndimage
import time

# Load model
model = load_model("googlenet_modelbest_on_deblurred_images.h5")

# Define categories
# Predictions of googLeNet model
activity_map = {'c0': 'Safe driving',
                'c1': 'Texting - right',
                'c2': 'Talking on the phone - right',
                'c3': 'Texting - left',
                'c4': 'Talking on the phone - left',
                'c5': 'Operating the radio',
                'c6': 'Drinking',
                'c7': 'Reaching behind',
                'c8': 'Hair and makeup',
                'c9': 'Talking to passenger'}

# Function to process the uploaded image
def process_uploaded_image(uploaded_image):
    processed_images = []
    img = io.imread(uploaded_image)

    # Deblurring and restoration using unsharp masking
    deblurred_img = unsharp_mask(img, radius=1, amount=2)  # Adjust parameters as needed
    deblurred_img = exposure.rescale_intensity(deblurred_img, out_range=(0, 255))
    deblurred_img = deblurred_img.astype(np.uint8)

    # Flip horizontally
    flipped_img = cv2.flip(deblurred_img, 1)

    # Rotate by 90 degrees
    rotated_img = ndimage.rotate(deblurred_img, 90)
    processed_images.extend([deblurred_img, flipped_img, rotated_img])
    return processed_images

# Function to make predictions on a single image
def predict_single_image(img, model):
    img_array = np.array(img.resize((64, 64)))  # Resize image to (300, 300, 3)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]

    return predicted_class, confidence

def main():
    st.set_page_config(
        page_title="Distracted Driver APP",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # # Header photo
    # header_image = "Drive Safe.jpg"
    # st.image(header_image, use_column_width=True)

    # Define image paths for each page
    home_image = "Drive Safe.jpg"
    image_predictor_image = "be_safe.png"
    about_us_image = "Drive Safe.jpg"

    # Create a sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Image Predictor", "About Us"])

    # Display the selected page
    if page == "Home":
        home_page(home_image)
    elif page == "Image Predictor":
        image_predictor_page(image_predictor_image)
    elif page == "About Us":
        about_us_page(about_us_image)

def home_page(image_path):
    st.image(image_path, use_column_width=True)
    st.title('Distracted Driver App')
    st.markdown("""
    ## Business Overview

    You're probably wondering why this APP? Well, road safety remains a critical concern around the world, with distracted driving claimed as being a leading cause of accidents. Distracted driving accounts for at least **9%** of annual car accidents in USA and is the leading cause of accidents worldwide.  

    According to an NTSA report on accidents in 2023, **1,072** people were killed on our roads, with the main causes being drunk driving, speeding and distracted driving. In Kenya we already have measures in place to tackle the first two: Alcoblow for drunk-driving, speed guns and speed governors for speeding. There seems to be nothing in place to tackle the third cause and that is where our project comes in.  

    This project aims to leverage computer vision and machine learning techniques to develop a system capable of detecting distracted drivers in real-time, contributing to enhanced road safety measures.

    ## Problem Statement

    Distracted driving poses significant risks, including accidents, injuries, and fatalities. Identifying and mitigating instances of distraction while driving is crucial to reducing road accidents.  

    The ballooning of car insurance claims led Directline Insurance, Kenya, to engage us in this project, with a vision to lower the rising claims from their customers.
    """, unsafe_allow_html=True)


def play_sound(sound_file="Distracted_driver_alert.aac"):
    st.markdown(f'<audio src="{sound_file}" autoplay="autoplay" controls="controls"></audio>', unsafe_allow_html=True)
# def play_sound(sound_file):
#     # Check if the predicted class is not c1
#     if categories[predicted_class] != 'c1':
#         # Play sound if the predicted class is not c1
#         st.markdown(f'<audio src="{sound_file}" autoplay="autoplay" controls="controls"></audio>', unsafe_allow_html=True)
    

def image_predictor_page(image_path):
    st.image(image_path, use_column_width=True)
    st.title("Driver Image Classification App")

    uploaded_file = st.file_uploader("Choose a driver image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        start_time = time.time()  # Start time counter

        # Make predictions
        predicted_class, confidence = predict_single_image(image, model)
        prediction = f"c{predicted_class}"
        end_time = time.time()  # End time counter
        elapsed_time = end_time - start_time  # Calculate elapsed time

        st.write(f"Predicted Class: {activity_map[prediction]}")
        st.write(f"Confidence: {confidence:.2%}")
        st.write(f"Time taken: {elapsed_time:.2f} seconds")

        # Check if the predicted class is not c1
        if activity_map[prediction] != 'Safe driving':
            # st.markdown(f'<audio src="{sound_file}" autoplay="autoplay" controls="controls"></audio>', unsafe_allow_html=True)
            # Play sound if the predicted class is not c1
            play_sound("Distracted_driver_alert.aac")

def about_us_page(image_path):
    st.image(image_path, use_column_width=True)
    st.title('About Us')

    st.subheader('Meet the Team')
    st.write("""
        We are all data science students from Flat Iron Moringa School, working on our capstone project.
    """)

    team_members = {
        "Leonard Gachimu": "https://github.com/leogachimu",
        "Rowlandson Kariuki": "https://github.com/RowlandsonK",
        "Francis Njenga": "https://github.com/GaturaN",
        "Mourine Mwangi": "https://github.com/Mourinem97",
        "Khadija Omar": "https://github.com/Khadija-Omar",
        "Victor Mawira": "https://github.com/Victormawira",
        "Onesphoro Kibunja": "https://github.com/Ones-Muiru"
    }

    for name, link in team_members.items():
        st.markdown(f"- [{name}]({link})")

if __name__ == "__main__":
    main()
