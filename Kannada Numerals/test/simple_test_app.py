import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import random
import base64
import os
import time

st.set_page_config(page_title="Kannada Digit Learner Prototype", layout="centered")

def load_model():
    try:
        with st.spinner("Loading AI model... Please wait"):
            return tf.keras.models.load_model('best_kannada_model.h5')
    except:
        try:
            with st.spinner("Loading AI model... Please wait"):
                return tf.keras.models.load_model('digit_recognition_cnn.h5')
        except:
            st.error(" No model found")
            return None

def preprocess_image(image):
    try:
        img_array = np.array(image.image_data)
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4: 
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
            else:  
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img_array = cv2.resize(img_array, (28, 28))
        img_array = 255 - img_array
        img_array = img_array.astype('float32') / 255.0
        return img_array.reshape(1, 28, 28, 1)
    except Exception as e:
        st.error(f"Image error: {e}")
        return None

def play_audio(digit):
    """Play audio using pre-recorded files"""
    audio_file = f"../audio/kannada_digits/{digit}.mp3"
    if os.path.exists(audio_file):
        with open(audio_file, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}"></audio>'
            st.components.v1.html(md, height=0)
        return True
    return False

def is_canvas_empty(canvas_result):
    """Check if canvas has drawing - FIXED the array error"""
    if canvas_result is None:
        return True
    if canvas_result.image_data is None:
        return True
    
    img_array = np.array(canvas_result.image_data)
    return np.all(img_array == 255)  # All white pixels

def st_canvas(**kwargs):
    try:
        from streamlit_drawable_canvas import st_canvas as canvas
        return canvas(**kwargs)
    except ImportError:
        st.error("Install: pip install streamlit-drawable-canvas")
        return None

def main():
    st.title(" Kannada Digit Learner Prototype")
    
    # Initialize session state
    if 'target_digit' not in st.session_state:
        st.session_state.target_digit = random.randint(0, 9)
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'attempts' not in st.session_state:
        st.session_state.attempts = 0
    # CHANGED: Added canvas key state for clearing functionality
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    
    model = load_model()
    if not model:
        return
    
    # App mode selection
    mode = st.sidebar.radio("Mode", [" Learn with Audio", "Free Practice"])
    
    if mode == " Learn with Audio":
        audio_learning_mode(model)
    else:
        free_practice_mode(model)

def audio_learning_mode(model):
    st.header(" Learn with Audio")
    
    # Score display
    col1, col2, col3 = st.columns(3)
    with col1: 
        st.metric("Score", st.session_state.score)
    with col2: 
        st.metric("Attempts", st.session_state.attempts)
    with col3: 
        accuracy = (st.session_state.score / max(st.session_state.attempts, 1)) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")
    
    # Controls
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔊 Play Random Number", use_container_width=True):
            if play_audio(st.session_state.target_digit):
                st.success("Listen carefully!")
            else:
                st.error("Audio not found")
    
    with col_b:
        if st.button(" New Number", use_container_width=True):
            st.session_state.target_digit = random.randint(0, 9)
            # CHANGED: Reset canvas when manually getting new number
            st.session_state.canvas_key += 1
            st.rerun()
    
    # Show answer helper
    if st.checkbox("Show current number (for testing)"):
        kannada_digits = ["೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯"]
        st.info(f"Target: {kannada_digits[st.session_state.target_digit]} (Digit {st.session_state.target_digit})")
    
    # Drawing canvas
    st.subheader("✏️ Draw what you hear")
    # CHANGED: Added dynamic key to force canvas refresh
    canvas_result = st_canvas(
        stroke_width=19,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key=f"audio_canvas_{st.session_state.canvas_key}",  # CHANGED: Dynamic key for clearing
    )
    
    # Check answer - FIXED: Proper canvas empty check
    if st.button(" Check Answer", type="primary", use_container_width=True):
        if is_canvas_empty(canvas_result):
            st.warning("Please draw something first!")
            return
        
        processed_image = preprocess_image(canvas_result)
        if processed_image is not None:
            with st.spinner("🔍 Analyzing your drawing... Please wait"):
                predictions = model.predict(processed_image, verbose=0)
                predicted_digit = np.argmax(predictions)
                confidence = np.max(predictions)
            
            kannada_digits = ["೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯"]
            st.session_state.attempts += 1
            
            if predicted_digit == st.session_state.target_digit and confidence > 0.3:
                st.success(f"🎉 Correct! It was {kannada_digits[predicted_digit]}")
                st.balloons()
                st.session_state.score += 1
                
                # Auto-next after 5 seconds
                st.info("Next number in 5 seconds...")
                with st.spinner("Loading next number..."):
                    time.sleep(5)
                # CHANGED: Clear canvas by updating key and get new number
                st.session_state.canvas_key += 1
                st.session_state.target_digit = random.randint(0, 9)
                st.rerun()
            else:
                st.error(f"Try again! You drew {kannada_digits[predicted_digit]}")
                st.write(f"Confidence: {confidence:.1%}")

def free_practice_mode(model):
    st.header("Free Practice")
    
    # CHANGED: Added separate canvas key for practice mode
    if 'practice_canvas_key' not in st.session_state:
        st.session_state.practice_canvas_key = 0
    
    # Drawing canvas
    canvas_result = st_canvas(
        stroke_width=15,
        stroke_color="#000000", 
        background_color="#FFFFFF",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key=f"practice_canvas_{st.session_state.practice_canvas_key}",  # CHANGED: Dynamic key
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(" Predict", type="primary", use_container_width=True):
            if is_canvas_empty(canvas_result):
                st.warning("Draw a digit first!")
                return
            
            processed_image = preprocess_image(canvas_result)
            if processed_image is not None:
                with st.spinner(" Analyzing your drawing... Please wait"):
                    predictions = model.predict(processed_image, verbose=0)
                    predicted_digit = np.argmax(predictions)
                    confidence = np.max(predictions)
                
                kannada_digits = ["೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯"]
                
                st.success(f"**Predicted: {kannada_digits[predicted_digit]} ({predicted_digit})**")
                st.metric("Confidence", f"{confidence:.1%}")
                
                # Hear the predicted digit
                if st.button("🔊 Hear this digit"):
                    play_audio(predicted_digit)
    
    with col2:
        # CHANGED: Clear canvas by updating the key
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.practice_canvas_key += 1
            st.rerun()

if __name__ == "__main__":
    main()
