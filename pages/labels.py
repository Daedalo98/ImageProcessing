import streamlit as st
import os
import json
import numpy as np
from PIL import Image
import img_functions as fn
import random

EMOTIONS_FILE = "list_of_emotions.json"
DEFAULT_EMOTIONS = ["Happy", "Sad", "Angry", "Surprised", "Disgusted", "Fearful", "Neutral"]

# Initialize emotions in session state so it persists during reruns
if "emotion_options" not in st.session_state:
    st.session_state.emotion_options = fn.load_emotions(EMOTIONS_FILE, DEFAULT_EMOTIONS)


# ==========================================
# MAIN APP LOGIC
# ==========================================
st.set_page_config(layout="wide", page_title="Emotion Labeler Pro")

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("Settings & Labels")
    
    with st.expander("📁 Paths & Configuration", expanded=True):
        image_dir = st.text_input("Images Directory", placeholder="e.g., data/images/")

        # Initialize directories dynamically if user confirms
        if st.button("Initialize Directories"):
            os.makedirs(image_dir, exist_ok=True)
            st.success("Directories verified/created.")

    # Emotion & Annotation Parameters
    st.markdown("---")
    st.write("**Annotation Data**")

    # Emotion Selection
    main_emotion = st.selectbox("Main Emotion", st.session_state.emotion_options)
    
    # Add new emotion UI
    with st.expander("➕ Add Custom Emotion"):
        new_emotion = st.text_input("New Emotion Name")
        if st.button("Add Emotion"):
            if new_emotion and new_emotion not in st.session_state.emotion_options:
                st.session_state.emotion_options.append(new_emotion)
                # Save to disk for future app launches
                with open(EMOTIONS_FILE, "w") as f:
                    json.dump(st.session_state.emotion_options, f)
                st.success(f"Added '{new_emotion}'!")
                st.rerun() # Refresh to update the selectbox
    intensity = st.slider("Emotion Intensity", 1, 10, 5)
    
    facs_aus_upper_face = [
        # --- Upper Face ---
        "AU 01 - Inner Brow Raiser", 
        "AU 02 - Outer Brow Raiser (unilateral, right side)", 
        "AU 04 - Brow Lowerer",
        "AU 05 - Upper Lid Raiser", 
        "AU 06 - Cheek Raiser", 
        "AU 07 - Lid Tightener"
        ]

    facs_aus_eye_eyelid = [
        # --- Eye & Eyelid Positions ---
        "AU 41 - Lid Droop", 
        "AU 42 - Slit", 
        "AU 43 - Eyes Closed",
        "AU 44 - Squint", 
        "AU 45 - Blink", 
        "AU 46 - Wink",

        "AU 61 - Eyes Turn Left", 
        "AU 62 - Eyes Turn Right", 
        "AU 63 - Eyes Up",
        "AU 64 - Eyes Down", 
        "AU 65 - Walleye", 
        "AU 66 - Cross-eye"
        ]
    
    facs_aus_lower_face = [
        # --- Lower Face ---
        "AU 09 - Nose Wrinkler", 
        "AU 10 - Upper Lip Raiser", 
        "AU 11 - Nasolabial Deepener",
        "AU 12 - Lip Corner Puller", 
        "AU 13 - Cheek Puffer", 
        "AU 14 - Dimpler",
        "AU 15 - Lip Corner Depressor", 
        "AU 16 - Lower Lip Depressor", 
        "AU 17 - Chin Raiser",
        "AU 18 - Lip Puckerer", 
        
        "AU 20 - Lip Stretcher", 
        
        "AU 22 - Lip Funneler",
        "AU 23 - Lip Tightener", 
        "AU 24 - Lip Pressor", 
        "AU 25 - Lips Part",
        "AU 26 - Jaw Drop", 
        "AU 27 - Mouth Stretch", 
        "AU 28 - Lip Suck"
        ]
    
    facs_aus_misc_supp = [
        # --- Miscellaneous / Supplemental ---
        "AU 29 - Jaw Thrust", 
        "AU 30 - Jaw Sideways", 
        "AU 31 - Jaw Clencher",
        "AU 32 - Lip Bite", 
        "AU 33 - Cheek Blow", 
        "AU 34 - Cheek Puff",
        "AU 35 - Cheek Suck", 
        "AU 36 - Tongue Show", 
        "AU 37 - Lip Wipe",
        "AU 38 - Nostril Dilator", 
        "AU 39 - Nostril Compressor"
        ]

    facs_aus_head_orient = [    
        # --- Head Orientations ---
        "AU 51 - Head Turn Left", 
        "AU 52 - Head Turn Right", 
        "AU 53 - Head Up",
        "AU 54 - Head Down", 
        "AU 55 - Head Tilt Left", 
        "AU 56 - Head Tilt Right",
        "AU 57 - Head Forward", 
        "AU 58 - Head Back"
        ]

    action_units_upperface = st.multiselect("Action Units (Upper Face)", facs_aus_upper_face)
    action_units_eyes = st.multiselect("Action Units (Eyes/Eyelids)", facs_aus_eye_eyelid)
    action_units_lowerface = st.multiselect("Action Units (Lower Face)", facs_aus_lower_face)
    action_units_misc = st.multiselect("Action Units (Misc/Supplemental)", facs_aus_misc_supp)
    action_units_head = st.multiselect("Action Units (Head Orientations)", facs_aus_head_orient)
    
# --- State Management for Images ---
if "image_idx" not in st.session_state:
    st.session_state.image_idx = 0

# Initialize an image list state
if "image_list" not in st.session_state:
    st.session_state.image_list = []

# Check if path exists before listing
if not os.path.exists(image_dir):
    st.warning(f"Image directory '{image_dir}' not found. Please update the path in the sidebar or click 'Initialize Directories'.")
    st.stop()

if os.path.exists(image_dir):
    # Get all valid images
    raw_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # If the app just loaded or the directory changed, update the state list
    if not st.session_state.image_list or set(raw_images) != set(st.session_state.image_list):
        st.session_state.image_list = raw_images

if not st.session_state.image_list:
    st.info(f"No images found in '{image_dir}'. Add some images to begin labeling.")
    st.stop()

# Use the state list for the current image
images = st.session_state.image_list

# Load Current Image
current_image_name = images[st.session_state.image_idx]
image_path = os.path.join(image_dir, current_image_name)

if not images:
    st.info(f"No images found in '{image_dir}'. Add some images to begin labeling.")
    st.stop()

# --- State Management for Landmark Groups ---

# MediaPipe configuration (Persists across all images)
if "mp_groups" not in st.session_state:
    st.session_state.mp_groups = {
        "Face Oval": {"color": "white", "ids": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]},
        "Left Eye": {"color": "cyan", "ids": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]},
        "Right Eye": {"color": "cyan", "ids": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]},
        "Mouth": {"color": "red", "ids": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]}
    }

# Manual landmarks (Specific to the current image)
current_manual_key = f"manual_{current_image_name}"
if current_manual_key not in st.session_state:
    # Structure: {"Group Name": {"color": "hex", "points": [[x, y], [x, y]]}}
    st.session_state[current_manual_key] = {}

# Randomization Toggle in the Sidebar (Add this inside the sidebar block)
shuffle_toggle = st.sidebar.checkbox("🔀 Randomize Image Flow")
if shuffle_toggle and "is_shuffled" not in st.session_state:
    random.shuffle(st.session_state.image_list)
    st.session_state.is_shuffled = True
    st.session_state.image_idx = 0 # Reset index when shuffling
elif not shuffle_toggle and "is_shuffled" in st.session_state:
    # Revert to alphabetical if unchecked
    st.session_state.image_list = sorted(st.session_state.image_list)
    del st.session_state["is_shuffled"]
    st.session_state.image_idx = 0

# Convert PIL Image to Numpy Array for MediaPipe & Plotly compatibility
pil_image = Image.open(image_path).convert("RGB")
numpy_image = np.array(pil_image)

st.write(f"**Current Image:** {current_image_name} ({st.session_state.image_idx + 1}/{len(images)})")


# --- Main Visualization ---

# Render Plotly Preview using the ACTIVE (edited) landmarks and the new toggle state
show_coords_toggle = st.toggle("🔎 Enable Hover Coordinates (Slower rendering)", value=False)

landmarks = fn.get_face_landmarks(numpy_image)
if landmarks is None:
    st.warning("No face landmarks detected in the preview image.")
else:
    st.success(f"Detected {len(landmarks)} landmarks in the preview image.")

show_landmark_toggle = st.toggle("👁️ Show Detected Landmarks", value=False, key="show_landmark")

# We place this right above the chart so it's easy to adjust while looking at the image
zoom_factor = st.slider("🔍 Zoom Level", min_value=1.0, max_value=5.0, value=1.0, step=0.1)

# Generate Base Figure
if show_landmark_toggle:
    fig_orig = fn.create_main_preview(
        numpy_image, 
        landmarks=landmarks, 
        highlight_ids=None, 
        show_coords=show_coords_toggle
    )
else:
    fig_orig = fn.create_main_preview(
        numpy_image, 
        landmarks=None, 
        highlight_ids=None, 
        show_coords=show_coords_toggle
    )

# Modify Axes for Zooming
if zoom_factor > 1.0:
    h, w = numpy_image.shape[:2]
    
    # Calculate the dimensions of the new, smaller viewable window
    new_w = w / zoom_factor
    new_h = h / zoom_factor
    
    # Find the dead center of the original image
    center_x = w / 2
    center_y = h / 2
    
    # Restrict Plotly's X and Y axes to this new window, effectively zooming in
    fig_orig.update_xaxes(range=[center_x - (new_w / 2), center_x + (new_w / 2)])
    fig_orig.update_yaxes(range=[center_y - (new_h / 2), center_y + (new_h / 2)])

# Render the Chart
st.plotly_chart(fig_orig, width='stretch', height='content', key="preview")

# --- Navigation ---
col_prev, col_save, col_next = st.columns([1, 2, 1])

with col_prev:
    if st.button("⬅️ Previous", width='stretch') and st.session_state.image_idx > 0:
        st.session_state.image_idx -= 1
        st.rerun()

with col_save:
    if st.button("💾 Save Settings to JSON", width='stretch'):
        # Package everything together using the dynamically assigned sidebar variables
        if landmarks is None:
            output_data = {
                "filename": current_image_name,
                "emotion": main_emotion,
                "intensity": intensity,
                "action_units": action_units_upperface + action_units_eyes + action_units_lowerface + action_units_misc + action_units_head,
                "landmarks": None
            }
        else:
            output_data = {
                "filename": current_image_name,
                "emotion": main_emotion,
                "intensity": intensity,
                "action_units": action_units_upperface + action_units_eyes + action_units_lowerface + action_units_misc + action_units_head,
                "landmarks": landmarks
            }
        
        save_path = os.path.join(image_dir, f"labels_{os.path.splitext(current_image_name)[0]}.json")
        with open(save_path, "w") as f:
            json.dump(output_data, f, indent=4)
        st.success(f"Annotation saved to `{save_path}`!")

with col_next:
    if st.button("Next ➡️", width='stretch') and st.session_state.image_idx < len(images) - 1:
        st.session_state.image_idx += 1
        st.rerun()