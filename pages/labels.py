import streamlit as st
import os
import json
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import img_functions as fn
import random

# Note: If you are still using the drawable canvas for bounding boxes, keep this import:
# from streamlit_drawable_canvas import st_canvas 

# ==========================================
# MODULAR FUNCTIONS (Integrated & Updated)
# ==========================================

EMOTIONS_FILE = "list_of_emotions.json"
DEFAULT_EMOTIONS = ["Happy", "Sad", "Angry", "Surprised", "Disgusted", "Fearful", "Neutral"]

def load_emotions():
    """Loads emotions from a local file, or returns defaults if the file doesn't exist."""
    if os.path.exists(EMOTIONS_FILE):
        with open(EMOTIONS_FILE, "r") as f:
            return json.load(f)
    else:
        # If the file doesn't exist, create it with default emotions
        with open(EMOTIONS_FILE, "w") as f:
            json.dump(DEFAULT_EMOTIONS, f, indent=4)
            return json.load(f)

# Initialize emotions in session state so it persists during reruns
if "emotion_options" not in st.session_state:
    st.session_state.emotion_options = load_emotions()

def create_main_preview(img, landmarks=None, highlight_ids=None, preview_height=600):
    """Renders the image in Plotly and conditionally overlays the face mesh."""
    h, w = img.shape[:2]
    fig = go.Figure()
    
    # Add image to the background
    fig.add_layout_image(
        dict(source=Image.fromarray(img), x=0, y=h, xref="x", yref="y",
             sizex=w, sizey=h, sizing="stretch", opacity=1, layer="below")
    )
    
    # Conditionally add the mesh
    if landmarks:
        x = [pt[0] for pt in landmarks]
        y = [h - pt[1] for pt in landmarks] # Invert Y for Plotly Cartesian plane
        text = [f"ID: {i}" for i in range(len(landmarks))]
        
        # Base styling for all landmarks
        colors = ['red'] * len(landmarks)
        sizes = [4] * len(landmarks)
        
        # Highlight specifically selected landmarks
        if highlight_ids:
            for hid in highlight_ids:
                if 0 <= hid < len(colors):
                    colors[hid] = 'lawngreen'
                    sizes[hid] = 10
                    text[hid] = f"ID: {hid} (SELECTED)"

        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', text=text,
                                 marker=dict(size=sizes, color=colors), hoverinfo='text'))
                                 
    fig.update_layout(xaxis=dict(range=[0, w], visible=False),
                      yaxis=dict(range=[0, h], visible=False, scaleanchor="x"),
                      margin=dict(l=0, r=0, t=0, b=0), height=preview_height)
    return fig

# ==========================================
# MAIN APP LOGIC
# ==========================================
st.set_page_config(layout="wide", page_title="Emotion Labeler Pro")

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("Settings & Labels")
    
    with st.expander("📁 Paths & Configuration", expanded=True):
        image_dir = st.text_input("Images Directory", placeholder="e.g., data/images/")
        preview_height = st.slider("Preview Height (px)", 400, 1000, 600)
        
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
    
    action_units = st.multiselect("Action Units", [
        "AU 01 - Inner Brow Raiser", "AU 02 - Outer Brow Raiser", 
        "AU 04 - Brow Lowerer", "AU 09 - Nose Wrinkler", 
        "AU 10 - Upper Lip Raiser", "AU 12 - Lip Corner Puller"
    ])
    
# --- State Management for Images ---
if "image_idx" not in st.session_state:
    st.session_state.image_idx = 0

# Check if path exists before listing
if not os.path.exists(image_dir):
    st.warning(f"Image directory '{image_dir}' not found. Please update the path in the sidebar or click 'Initialize Directories'.")
    st.stop()

# Initialize an image list state
if "image_list" not in st.session_state:
    st.session_state.image_list = []

if os.path.exists(image_dir):
    # Get all valid images
    raw_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # If the app just loaded or the directory changed, update the state list
    if not st.session_state.image_list or set(raw_images) != set(st.session_state.image_list):
        st.session_state.image_list = raw_images

if not st.session_state.image_list:
    st.info(f"No images found in '{image_dir}'. Add some images to begin labeling.")
    st.stop()

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

# Use the state list for the current image
images = st.session_state.image_list

if not images:
    st.info(f"No images found in '{image_dir}'. Add some images to begin labeling.")
    st.stop()

# Load Current Image
current_image_name = images[st.session_state.image_idx]
image_path = os.path.join(image_dir, current_image_name)

# Convert PIL Image to Numpy Array for MediaPipe & Plotly compatibility
pil_image = Image.open(image_path).convert("RGB")
numpy_image = np.array(pil_image)

st.write(f"**Current Image:** {current_image_name} ({st.session_state.image_idx + 1}/{len(images)})")


# --- Main Visualization ---

# Use a session state key linked to the current image so we don't carry over old landmarks
current_lm_key = f"lm_{current_image_name}"

if current_lm_key not in st.session_state:
    # Run MediaPipe ONLY if we haven't already processed this image in the current session
    raw_lms = fn.get_face_landmarks(numpy_image)
    st.session_state[current_lm_key] = raw_lms if raw_lms else []

# Retrieve mutable landmarks from state
active_landmarks = st.session_state[current_lm_key]

if not active_landmarks:
    st.warning("No face detected. You can add manual landmarks below.")

# UI to modify landmarks
with st.expander("🛠️ Edit Landmarks (Add/Remove)"):
    col_add, col_rem = st.columns(2)
    
    with col_add:
        st.write("Add Landmark")
        new_x = st.number_input("X Coordinate", min_value=0, max_value=numpy_image.shape[1], value=0)
        new_y = st.number_input("Y Coordinate", min_value=0, max_value=numpy_image.shape[0], value=0)
        if st.button("Add Point"):
            st.session_state[current_lm_key].append([new_x, new_y])
            st.rerun()
            
    with col_rem:
        st.write("Remove Landmark")
        lm_to_remove = st.number_input("Landmark ID to remove", min_value=0, max_value=max(0, len(active_landmarks)-1), value=0)
        if st.button("Remove Point") and active_landmarks:
            st.session_state[current_lm_key].pop(lm_to_remove)
            st.rerun()

# Render Plotly Preview using the ACTIVE (edited) landmarks
fig = create_main_preview(numpy_image, active_landmarks, highlight_ids=None, preview_height=preview_height)
st.plotly_chart(fig, use_container_width=True)


# --- Navigation ---
col_prev, col_save, col_next = st.columns([1, 2, 1])

with col_prev:
    if st.button("⬅️ Previous") and st.session_state.image_idx > 0:
        st.session_state.image_idx -= 1
        st.rerun()

with col_save:
    if st.button("💾 Save Settings to JSON", width='stretch'):
        # Package everything together using the dynamically assigned sidebar variables
        output_data = {
            "filename": current_image_name,
            "emotion": main_emotion,
            "intensity": intensity,
            "action_units": action_units
        }
        
        save_path = os.path.join(image_dir, f"labels_{os.path.splitext(current_image_name)[0]}.json")
        with open(save_path, "w") as f:
            json.dump(output_data, f, indent=4)
        st.success(f"Annotation saved to `{save_path}`!")

with col_next:
    if st.button("Next ➡️") and st.session_state.image_idx < len(images) - 1:
        st.session_state.image_idx += 1
        st.rerun()