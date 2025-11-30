import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import random
import base64
import os
import time

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

st.set_page_config(page_title="Kannada Digit Learner Prototype", layout="centered")

def load_model():
    try:
        with st.spinner("Loading AI model... Please wait"):
            model = tf.keras.models.load_model('best_kannada_model.h5')
            # Compile the model to avoid warnings
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model
    except:
        try:
            with st.spinner("Loading AI model... Please wait"):
                model = tf.keras.models.load_model('digit_recognition_cnn.h5')
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                return model
        except:
            st.error("No model found. Please check if the model files exist.")
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
    # Try multiple possible audio file paths
    audio_paths = [
        f"audio/kannada_digits/{digit}.mp3",
        f"../audio/kannada_digits/{digit}.mp3",
        f"./audio/kannada_digits/{digit}.mp3",
        f"kannada_digits/{digit}.mp3",
        f"../kannada_digits/{digit}.mp3"
    ]
    
    for audio_file in audio_paths:
        if os.path.exists(audio_file):
            try:
                with open(audio_file, "rb") as f:
                    data = f.read()
                    b64 = base64.b64encode(data).decode()
                    md = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}"></audio>'
                    st.components.v1.html(md, height=0)
                st.success(f"Playing audio for digit {digit}")
                return True
            except Exception as e:
                continue
        else:
            st.warning(f"Audio file not found: {audio_file}")
    
    # If no audio file found
    st.error(f"Audio file for digit {digit} not found in any location")
    return False

def is_canvas_empty(canvas_result):
    """Check if canvas has drawing"""
    if canvas_result is None:
        return True
    if canvas_result.image_data is None:
        return True
    img_array = np.array(canvas_result.image_data)
    return np.all(img_array == 255)  

def st_canvas(**kwargs):
    try:
        from streamlit_drawable_canvas import st_canvas as canvas
        return canvas(**kwargs)
    except ImportError:
        st.error("Install: pip install streamlit-drawable-canvas")
        return None

def get_animated_numbers_css():
    return """
    <style>
    @keyframes draw {
        to {
            stroke-dashoffset: 0;
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0.3; }
        to { opacity: 1; }
    }
    
    .number-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
    
    .number-svg {
        width: 280px;
        height: 280px;
        background: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        display: block;
    }
    
    .number-base {
        fill: none;
        stroke: #e0e0e0;
        stroke-width: 4.5;
        stroke-linecap: round;
        stroke-linejoin: round;
        opacity: 0.6;
    }
    
    .number-path {
        fill: none;
        stroke: #4CAF50;
        stroke-width: 3;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-dasharray: 100;
        stroke-dashoffset: 100;
        animation: draw 2s ease-in-out forwards;
        filter: drop-shadow(0 2px 3px rgba(0,0,0,0.2));
    }
    
    .number-path-part2 {
        fill: none;
        stroke: #4CAF50;
        stroke-width: 3;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-dasharray: 100;
        stroke-dashoffset: 100;
        animation: draw 2s ease-in-out forwards;
        filter: drop-shadow(0 2px 3px rgba(0,0,0,0.2));
    }
    
    .progress {
        height: 8px;
        background: #e0e0e0;
        border-radius: 4px;
        margin: 15px 0;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        transition: width 0.5s ease;
    }
    
    .number-display {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: #333;
        margin: 10px 0;
    }
    
    .instruction {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin: 10px 0;
    }
    
    .main > div {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 20px !important;
        margin: 5px 0;
    }
    </style>
    """

def create_kannada_animated_number(digit):
    """Create animated Kannada number SVG with proper paths and animation delays"""
    kannada_numbers = {
        0: """
        <div class="number-container">
            <svg viewBox="0 0 70 80" class="number-svg">
                <!-- Light base version -->
                <path class="number-base" d="M 35,20 
                                            C 25,20 20,25 20,35 
                                            C 20,45 25,50 35,50 
                                            C 45,50 50,45 50,35 
                                            C 50,25 45,20 35,20" />
                <!-- Oval shape drawn counter-clockwise -->
                <path class="number-path" d="M 35,20 
                                            C 25,20 20,25 20,35 
                                            C 20,45 25,50 35,50 
                                            C 45,50 50,45 50,35 
                                            C 50,25 45,20 35,20" 
                      style="animation-delay: 0.1s">
                </path>
            </svg>
        </div>
        """,
        1: """
        <div class="number-container">
            <svg viewBox="0 0 70 80" class="number-svg">
                <!-- Light base version -->
                <path class="number-base" d="M 25,65 
                                            C 25,20 45,20 47,65" /> 
                <!-- Kannada 1 (೧): Inverted U shape -->
                <path class="number-path" d="M 25,65 
                                            C 25,20 45,20 47,65" 
                      style="animation-delay: 0.2s">
                </path>
            </svg>
        </div>
        """,
        2: """
        <div class="number-container">
            <svg viewBox="0 0 70 80" class="number-svg">
                <!-- Light base version -->
                <path class="number-base" d="M 25,45 L 55,45 M 55,45 
                                                  A 10,10 0 0,0 55,25
                                                  M 55,25 
                                                  A 10,10 0 0,0 55,45" />
                <path class="number-path" d="M 25,45 L 55,45" 
                      style="animation-delay: 0.3s">
                </path>
                <!-- Top semi-circle (anti-clockwise from 0° to 180°) -->
                <path class="number-path-part2" d="M 55,45 
                                                  A 10,10 0 0,0 55,25" 
                      style="animation-delay: 1.2s">
                </path>
                <!-- Bottom semi-circle to complete the full circle (anti-clockwise from 180° to 360°) -->
                <path class="number-path-part2" d="M 55,25 
                                                  A 10,10 0 0,0 55,45" 
                      style="animation-delay: 2.0s">
                </path>
            </svg>
        </div>
        """,
        3: """
        <div class="number-container">
            <svg viewBox="0 0 70 80" class="number-svg">
                <!-- Light base version -->
                <path class="number-base" d="M 38,30 
                                            C 25,30 25,20 35,20 
                                            C 45,20 45,30 41,30  
                                            M 43,25
                                            C 45,25 45,40 40,50
                                            M 40,50 
                                            C 35,60 25,60 25,50 
                                            C 25,40 35,40 40,45
                                            M 40,45  
                                            C 50,50 50,55 50,60" />
                <!-- Small top loop counterclockwise -->
                <path class="number-path" d="M 38,30 
                                            C 25,30 25,20 35,20 
                                            C 45,20 45,30 43,25" 
                      style="animation-delay: 0.4s">
                </path>
                <!-- Middle forward sweeping curve -->
                <path class="number-path-part2" d="M 43,25
                                                  C 45,25 45,40 40,50" 
                      style="animation-delay: 1.2s">
                </path>
                <!-- Bottom larger loop clockwise -->
                <path class="number-path-part2" d="M 40,50 
                                                  C 35,60 25,60 25,50 
                                                  C 25,40 35,40 40,45" 
                      style="animation-delay: 1.8s">
                </path>
                <!-- Sharp turn and downward right sweep -->
                <path class="number-path-part2" d="M 40,45  
                                                  C 50,50 50,55 50,60" 
                      style="animation-delay: 2.5s">
                </path>
            </svg>
        </div>
        """,
        4: """
        <div class="number-container">
            <svg viewBox="0 0 70 80" class="number-svg">
                <path class="number-base" d="M 30,25 
                                            C 35,25 35,20 30,20 
                                            C 19,20 25,25 25,20 
                                            M 25,20 
                                            C 20,25 25,30 30,35 
                                            C 40,45 50,55 25,55
                                            M 25,55 
                                                C 20,45 45,40 50,22" />  
                <!-- Small starting loop anti-clockwise -->
                <path class="number-path" d="M 30,25 
                                            C 35,25 35,20 30,20 
                                            C 19,20 25,25 25,20" 
                      style="animation-delay: 0.3s">
                </path>
                <!-- Starting hook and descending curve -->
                <path class="number-path" d="M 25,20 
                                            C 20,25 25,30 30,35 
                                            C 40,45 50,55 25,55" 
                      style="animation-delay: 0.9s">
                </path>
                <!-- Rising diagonal stroke -->
                <path class="number-path-part2" d="M 25,55 
                                                  C 20,45 45,40 50,22" 
                      style="animation-delay: 1.9s">
                </path> 
            </svg>
        </div>
        """,
        5: """
        <div class="number-container">
            <svg viewBox="0 0 70 80" class="number-svg">
                <path class="number-base" d= "M 40,30 
                                            C 25,30 25,20 35,20 
                                            C 45,20 45,30 35,30
                                            M 43,23 
                                                  C 40,30 45,35 45,45
                                                  M 45,45 
                                                  C 45,55 30,60 25,50
                                                  M 25,50 
                                                  C 20,45 30,40 30,45 
                                                  C 35,50 60,55 58,20
                                                  A 20,10 0 0,0 55,19
                                                  A 10,10 0 0,0 55,29
                                                  V 52  "/>
                <!-- Small top loop counterclockwise -->
                <path class="number-path" d="M 40,30 
                                            C 25,30 25,20 35,20 
                                            C 45,20 45,30 35,30" 
                      style="animation-delay: 0.4s">
                </path>
                <!-- Downward sweep slanted down-right -->
                <path class="number-path-part2" d="M 43,23 
                                                  C 40,30 45,35 45,45" 
                      style="animation-delay: 1.3s">
                </path>
                <!-- Leftward belly curve (wide C-shape) -->
                <path class="number-path-part2" d="M 45,45 
                                                  C 45,55 30,60 25,50" 
                      style="animation-delay: 1.9s">
                </path>
                <!-- Sharp turn and downward right sweep -->
                <path class="number-path-part2" d="M 25,50 
                                                  C 20,45 30,40 30,45 
                                                  C 35,50 60,55 58,20
                                                  A 20,10 0 0,0 55,19
                                                  A 10,10 0 0,0 55,29
                                                  V 52" 
                      style="animation-delay: 2.5s">
                </path>
            </svg>
        </div>
        """,
        6: """
        <div class="number-container">
            <svg viewBox="0 0 70 80" class="number-svg">
                <path class="number-base" d="M 42,20 
                                            C 35,15 20,25 25,40 
                                            C 30,55 45,60 50,50
                                            M 50,50 
                                                C 45,40 35,40 30,45
                                                M 30,45 
                                            C 30,80 40,60 55,65"/>
                <!-- Outer loop -->
                <path class="number-path" d="M 42,20 
                                            C 35,15 20,25 25,40 
                                            C 30,55 45,60 50,50" 
                      style="animation-delay: 0.7s">
                </path>
                <!-- Inner curl - starts after loop completes -->
                <path class="number-path-part2" d="M 50,50 
                                                C 45,40 35,40 30,45" 
                      style="animation-delay: 1.8s">
                </path>
                <!-- Outer loop -->
                <path class="number-path" d="M 30,45 
                                            C 30,80 40,60 55,65" 
                      style="animation-delay: 2.0s">
                </path>
            </svg>
        </div>
        """,
        7: """
        <div class="number-container">
            <svg viewBox="0 0 70 80" class="number-svg">
                <path class="number-base" d="M 40,30 
                                            C 30,30 25,20 35,20 
                                            C 45,20 45,30 35,30
                                            M 44,25
                                                C 40,35 45,40 30,50
                                                M 30,50
                                            H 50"/>
                <!-- Small top loop counterclockwise -->
                <path class="number-path" d="M 40,30 
                                            C 30,30 25,20 35,20 
                                            C 45,20 45,30 35,30" 
                      style="animation-delay: 0.4s">
                </path>
                <path class="number-path-part2" d="M 44,25
                                                  C 40,35 45,40 30,50" 
                      style="animation-delay: 1.2s">
                </path>
                <path class="number-path" d="M 30,50
                                            H 50"
                      style="animation-delay: 2.4s">
                </path>
            </svg>
        </div>
        """,
        8: """
        <div class="number-container">
            <svg viewBox="0 0 70 80" class="number-svg">
                <path class="number-base" d="M   30,28 
                                            C 30,28 30,25 29,30 
                                            C 35,35 40,30 35,19
                                            M 33,19 
                                            C 25,20 20,25 20,35 
                                            C 20,45 25,50 35,50 
                                            C 45,50 50,45 50,23
                                            M 50,23
                                            H 62"/>
                <path class="number-path" d="M  30,28 
                                            C 30,28 30,25 29,30 
                                            C 35,35 40,30 35,19"
                      style="animation-delay: 0.5s">
                </path>
                <path class="number-path" d="M 33,19 
                                            C 25,20 20,25 20,35 
                                            C 20,45 25,50 35,50 
                                            C 45,50 50,45 50,23"
                      style="animation-delay: 1.0s">
                </path>
                <path class="number-path" d="M 50,23
                                            H 62"
                      style="animation-delay: 2.4s">
                </path>
            </svg>
        </div>
        """,
        9: """
        <div class="number-container">
            <svg viewBox="0 0 70 80" class="number-svg">
                <path class="number-base" d=" M 60,20 
                                            C 35,15 20,25 25,40 
                                            C 30,55 45,50 50,40
                                            M 50,40 
                                                C 50,40 35,30 30,45
                                                M 30,45 
                                            C 20,80 45,60 45,65"/>
                <!-- Outer loop -->
                <path class="number-path" d="M 60,20 
                                            C 35,15 20,25 25,40 
                                            C 30,55 45,50 50,40" 
                      style="animation-delay: 0.4s">
                </path>
                <!-- Inner curl - starts after loop completes -->
                <path class="number-path-part2" d="M 50,40 
                                                C 50,40 35,30 30,45" 
                      style="animation-delay: 1.8s">
                </path>
                <!-- Outer loop -->
                <path class="number-path" d="M 30,45 
                                            C 20,80 45,60 45,65" 
                      style="animation-delay: 2.3s">
                </path>
            </svg>
        </div>
        """
    }
    
    return kannada_numbers.get(digit, "")

def main():
    st.title("Kannada Digit Learner")
    
    # Initialize session state
    if 'target_digit' not in st.session_state:
        st.session_state.target_digit = random.randint(0, 9)
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'attempts' not in st.session_state:
        st.session_state.attempts = 0
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    if 'learning_digit' not in st.session_state:
        st.session_state.learning_digit = 0
    if 'last_played_digit' not in st.session_state:
        st.session_state.last_played_digit = -1
    if 'animation_key' not in st.session_state:
        st.session_state.animation_key = 0
    if 'predicted_digit' not in st.session_state:
        st.session_state.predicted_digit = None
    if 'page_loaded' not in st.session_state:
        st.session_state.page_loaded = False
    
    model = load_model()
    if not model:
        st.warning("Running in demo mode without AI model")
    
    # App mode selection
    mode = st.sidebar.radio("Mode", ["Learn Numbers", "Learn with Audio", "Free Practice"])
    if mode == "Learn Numbers":
        step_by_step_learning(model)
    elif mode == "Learn with Audio":
        audio_learning_mode(model)
    else:
        free_practice_mode(model)

def step_by_step_learning(model):
    """Step-by-step learning mode with animations"""
    st.markdown(get_animated_numbers_css(), unsafe_allow_html=True)
    
    kannada_digits = ["೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯"]
    current_digit = st.session_state.learning_digit
    
    # Auto-play audio when digit changes
    if not st.session_state.page_loaded or st.session_state.last_played_digit != current_digit:
        # Play audio immediately for all digits including 0
        play_audio(current_digit)
        st.session_state.last_played_digit = current_digit
        st.session_state.page_loaded = True
    
    # Progress bar
    progress = (current_digit + 1) / 10
    st.markdown(f"""
        <div class="progress">
            <div class="progress-fill" style="width: {progress * 100}%"></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Current number display
    st.markdown(f'<div class="number-display">{kannada_digits[current_digit]} - {current_digit}</div>', unsafe_allow_html=True)
    
    # Animation display
    st.markdown(create_kannada_animated_number(current_digit), unsafe_allow_html=True)
    
    # Add JavaScript to restart animations on page load and navigation
    st.markdown(f"""
    <script>
    function restartAnimations() {{
        const paths = document.querySelectorAll('.number-path, .number-path-part2');
        paths.forEach(path => {{
            // Remove and re-add animation to trigger restart while preserving delays
            const originalAnimation = path.style.animation;
            const originalDelay = path.style.animationDelay || getComputedStyle(path).animationDelay;
            path.style.animation = 'none';
            void path.offsetWidth; // Trigger reflow
            path.style.animation = originalAnimation;
            path.style.animationDelay = originalDelay;
        }});
    }}
    
    // Restart animations when page loads
    document.addEventListener('DOMContentLoaded', function() {{
        setTimeout(restartAnimations, 100);
    }});
    
    // Also restart after a short delay to ensure components are loaded
    setTimeout(restartAnimations, 500);
    
    // Use MutationObserver to detect when Streamlit updates content
    const observer = new MutationObserver(function(mutations) {{
        mutations.forEach(function(mutation) {{
            if (mutation.addedNodes.length) {{
                setTimeout(restartAnimations, 200);
            }}
        }});
    }});
    
    observer.observe(document.body, {{
        childList: true,
        subtree: true
    }});
    
    // Force restart for this specific digit change
    setTimeout(restartAnimations, 300);
    </script>
    """, unsafe_allow_html=True)
    
    # Instruction
    st.markdown('<div class="instruction">Watch the animation, then try drawing it below</div>', unsafe_allow_html=True)
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔊 Hear Again", use_container_width=True):
            play_audio(current_digit)
    
    with col2:
        if st.button("➡️ Next Number", use_container_width=True):
            if current_digit < 9:
                st.session_state.learning_digit += 1
            else:
                st.session_state.learning_digit = 0  # Loop back to start
            st.session_state.canvas_key += 1
            st.session_state.animation_key += 1
            st.rerun()
    
    # Practice canvas
    st.markdown("### ✏️ Practice Drawing Here")
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key=f"learning_canvas_{st.session_state.canvas_key}",
    )
    
    # Check practice drawing
    if st.button("✓ Check My Drawing", type="primary", use_container_width=True):
        if is_canvas_empty(canvas_result):
            st.warning("Please draw the number first!")
            return
            
        processed_image = preprocess_image(canvas_result)
        if processed_image is not None and model is not None:
            with st.spinner("Checking..."):
                predictions = model.predict(processed_image, verbose=0)
                predicted_digit = np.argmax(predictions)
                confidence = np.max(predictions)
            
            if predicted_digit == current_digit and confidence > 0.3:
                st.success("🎉 Perfect! You got it right!")
                time.sleep(1)
                # Auto-advance to next number after success
                if current_digit < 9:
                    st.session_state.learning_digit += 1
                st.session_state.canvas_key += 1
                st.session_state.animation_key += 1
                st.rerun()
            else:
                st.error(f"Almost! Try again. You drew {kannada_digits[predicted_digit]}")
        else:
            st.info("Model not available - this is just for practice")

def audio_learning_mode(model):
    st.header("Learn with Audio")
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
                st.error("Audio file not found. Please check the audio file path.")
    with col_b:
        if st.button("🔄 New Number", use_container_width=True):
            st.session_state.target_digit = random.randint(0, 9)
            st.session_state.canvas_key += 1
            st.rerun()
    if st.checkbox("Show current number"):
        kannada_digits = ["೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯"]
        st.info(f"Target: {kannada_digits[st.session_state.target_digit]} (Digit {st.session_state.target_digit})")
    
    # Drawing canvas
    st.subheader("✏️ Draw what you hear")
    canvas_result = st_canvas(
        stroke_width=19,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key=f"audio_canvas_{st.session_state.canvas_key}",  
    )
    
    # Check answer 
    if st.button("✅ Check Answer", type="primary", use_container_width=True):
        if is_canvas_empty(canvas_result):
            st.warning("Please draw something first!")
            return
        processed_image = preprocess_image(canvas_result)
        if processed_image is not None and model is not None:
            with st.spinner("Analyzing your drawing..."):
                predictions = model.predict(processed_image, verbose=0)
                predicted_digit = np.argmax(predictions)
                confidence = np.max(predictions)
            kannada_digits = ["೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯"]
            st.session_state.attempts += 1
            if predicted_digit == st.session_state.target_digit and confidence > 0.3:
                st.success(f"🎉 Correct! It was {kannada_digits[predicted_digit]}")
                st.balloons()
                st.session_state.score += 1
                
                st.info("Next number in 3 seconds...")
                with st.spinner("Loading next number..."):
                    time.sleep(3)
                st.session_state.canvas_key += 1
                st.session_state.target_digit = random.randint(0, 9)
                st.rerun()
            else:
                st.error(f"Try again! You drew {kannada_digits[predicted_digit]}")
                st.write(f"Confidence: {confidence:.1%}")
        else:
            st.info("Model not available - this is just for practice")
                
def free_practice_mode(model):
    st.header("Free Practice")
    
    if 'practice_canvas_key' not in st.session_state:
        st.session_state.practice_canvas_key = 0
    if 'predicted_digit' not in st.session_state:
        st.session_state.predicted_digit = None
    
    canvas_result = st_canvas(
        stroke_width=15,
        stroke_color="#000000", 
        background_color="#FFFFFF",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key=f"practice_canvas_{st.session_state.practice_canvas_key}",  
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Predict", type="primary", use_container_width=True):
            if is_canvas_empty(canvas_result):
                st.warning("Draw a digit first!")
                return
            
            processed_image = preprocess_image(canvas_result)
            if processed_image is not None and model is not None:
                with st.spinner("Analyzing your drawing..."):
                    predictions = model.predict(processed_image, verbose=0)
                    predicted_digit = np.argmax(predictions)
                    confidence = np.max(predictions)
                
                kannada_digits = ["೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯"]
                st.session_state.predicted_digit = predicted_digit
                
                st.success(f"**Predicted: {kannada_digits[predicted_digit]} ({predicted_digit})**")
                st.metric("Confidence", f"{confidence:.1%}")  # FIXED: changed accuracy to confidence
            else:
                st.info("Model not available - this is just for practice")
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.practice_canvas_key += 1
            st.session_state.predicted_digit = None
            st.rerun()
    
    # Audio button for predicted digit
    if st.session_state.predicted_digit is not None:
        if st.button("🔊 Hear this digit", use_container_width=True):
            play_audio(st.session_state.predicted_digit)

if __name__ == "__main__":
    main()
