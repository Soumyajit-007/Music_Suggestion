import streamlit as st
import numpy as np
import pandas as pd
import cv2
from keras.models import load_model

# Load model and music data
classifier = load_model('emotion_detection.h5', compile=False)
mood_music = pd.read_csv("data_moods.csv")

def recommend_songs(pred_class):
    if pred_class in ['disgust', 'sadness']:
        mood = 'Sad'
    elif pred_class == 'happiness':
        mood = 'Happy'
    elif pred_class in ['fear', 'anger']:
        mood = 'Calm'
    elif pred_class in ['surprise', 'neutral']:
        mood = 'Energetic'
    else:
        return None

    recommended_songs = mood_music[mood_music['mood'] == mood]
    if recommended_songs.empty:
        recommended_songs = mood_music
    recommended_songs = recommended_songs.sample(frac=1).reset_index(drop=True)
    recommended_songs = recommended_songs.head(5)
    return recommended_songs

# Function to convert predicted values ​​into labels
def map_prediction_to_emotion(prediction):
    emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    max_prob_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[max_prob_index]
    return predicted_emotion

# Function to capture images from the camera
def capture_frame():
    # Initialize the camera
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        st.error("Cannot access camera. Please try again.")
        return None

    # Capture a frame
    ret, frame = capture.read()
    if ret:
        # Preprocess the frame for prediction
        frame_resized = cv2.resize(frame, (224, 224))
        frame_resized_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        st.image(frame_resized_bgr, caption='Picture Captured.', use_column_width=True)
        return frame_resized
    else:
        st.error("Failed to take picture. Please try again.")
        return None
    # Release the camera
    capture.release()

# Titles on Streamlit
st.title('Emotion Prediction and Music Recommendation')

# Option to upload or take a picture
option = st.radio("Select an option:", ("Select an image", "Take a picture from the camera"))

if option == "Select an image":
    # Upload image
    uploaded_file = st.file_uploader('Select an image...', type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        # Reading pictures
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        st.image(image_bgr, caption='Image Uploaded.', use_column_width=True)
        st.write("")
        st.write("Processing...")

        # Emotion prediction
        image_resized = cv2.resize(image, (224, 224))
        image_expanded = np.expand_dims(image_resized, axis=0)
        image_array = np.array(image_expanded)
        predictions = classifier.predict(image_array)
        predicted_emotion = map_prediction_to_emotion(predictions)

        # Elicit emotional predictions
        st.write(f'Emotion Prediction: {predicted_emotion}')
        
        # Music recommendations based on emotion prediction
        recommended_songs = recommend_songs(predicted_emotion)
        if recommended_songs is not None:
            st.write('Music Recommendations:')
            for index, row in recommended_songs.iterrows():
                st.write(f'{index + 1}. **{row["name"]}** by {row["artist"]}')
                spotify_link = f'[Open in Spotify](spotify:track:{row["id"]})'
                st.markdown(spotify_link, unsafe_allow_html=True)
        else:
            st.write('There are no music recommendations for this emotion.')
            
        st.markdown("<p style='text-align: center;'>Please click 'R' to get other music recommendations.</p>", unsafe_allow_html=True)


elif option == "Take a picture from the camera":
    st.write("Click the button to take a picture")
    if st.button("Take picture"):
        captured_frame = capture_frame()
        if captured_frame is not None:
            height, width, _ = captured_frame.shape

            # ROI (region of interest)
            zoom_factor = 1.3  
            zoomed_width = int(width / zoom_factor)
            zoomed_height = int(height / zoom_factor)
            roi = captured_frame[
                int((height - zoomed_height) / 2):int((height + zoomed_height) / 2),
                int((width - zoomed_width) / 2):int((width + zoomed_width) / 2)
            ]
            # Predict emotions from captured images
            image_resized = cv2.resize(roi, (224, 224))
            image_expanded = np.expand_dims(image_resized, axis=0)
            image_array = np.array(image_expanded)
            predictions = classifier.predict(image_array)
            predicted_emotion = map_prediction_to_emotion(predictions)

            # Elicit emotional predictions
            st.write(f'Emotion Prediction: {predicted_emotion}')
            
            # Music recommendations based on emotion prediction
            recommended_songs = recommend_songs(predicted_emotion)
            if recommended_songs is not None:
                st.write('Music Recommendations:')
                for index, row in recommended_songs.iterrows():
                    st.write(f'{index + 1}. **{row["name"]}** by {row["artist"]}')
                    spotify_link = f'[Open in Spotify](spotify:track:{row["id"]})'
                    st.markdown(spotify_link, unsafe_allow_html=True)
            else:
                st.write('There are no music recommendations for this emotion.')
            
            st.markdown("<p style='text-align: center;'>Please click 'R' to get other music recommendations.</p>",unsafe_allow_html=True) 
