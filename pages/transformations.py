import streamlit as st
import cv2
import numpy as np
import os
import io
from PIL import Image
import uuid
import img_functions as fn
import json
import tempfile
from pathlib import Path

# ==========================================
# STREAMLIT UI ARCHITECTURE
# ==========================================
preview_width, preview_height = None, None # Global variables to store dimensions of the first uploaded image for dynamic parameter limits

st.set_page_config(page_title="Advanced Side-by-Side CV", layout="wide")

# Initialize Session States
if 'transform_pipeline' not in st.session_state:
    st.session_state.transform_pipeline = []
if 'info_pipeline' not in st.session_state:
    st.session_state.info_pipeline = []

INFO_CATEGORIES = {
    "Dimensional": {
        "label": "Dimensional & Spatial",
        "keys": ["Resolution", "Aspect Ratio", "Orientation"]
    },
    "Color": {
        "label": "Color & Depth",
        "keys": ["Channels", "Color Model", "Color Space"]
    },
    "Photographic": {
        "label": "Photographic",
        "keys": ["Luminance (Mean)", "Contrast (Std Dev)", "Sharpness (Variance of Laplacian)"]
    },
    "Structural": {
        "label": "Structural (Stats & Hist)",
        "keys": ["Entropy", "Histogram Data"]
    },
    "Metadata": {
        "label": "Metadata & Encapsulation",
        "keys": ["File Format", "Color Space", "Weight", "EXIF"]
    },
}

# Transformation Pipeline Configuration
TRANSFORM_CATEGORIES = {
    "Geometric": {
        "label": "Geometric",
        "keys": [
            "translate_image", "rotate_image", "scale_image", "resize_image",
            "shear_image", "square_image", "flip_image", "crop_image"
        ]
    },
    "Color": {
        "label": "Color & Photographic",
        "keys": [
            "adjust_brightness_contrast", "apply_gamma", 
            "apply_histogram_equalization", "apply_thresholding", 
            "apply_posterization", "invert_image"
        ]
    },
    "Filtering": {
        "label": "Filtering & Structural",
        "keys": [
            "apply_gaussian_blur", "apply_median_blur", 
            "apply_unsharp_mask", "apply_canny", "apply_morphology"
        ]
    },
    "Frequency": {
        "label": "Frequency & Noise",
        "keys": [
            "apply_dwt", "add_gaussian_noise", 
            "add_salt_pepper_noise", "add_jpeg_artifacts"
        ]
    },
    "Face": {
        "label": "Face Processing Studio",
        "keys": ["advanced_crop_face", "align_face"]
    }
}

# Helper Functions
def add_transform(op_name):
    if op_name:
        st.session_state.transform_pipeline.append({'id': str(uuid.uuid4()), 'op': op_name})
        st.toast(f"➕ Added '{op_name}' to pipeline", icon="✅")

def remove_transform(index):
    removed_op = st.session_state.transform_pipeline[index]['op']
    st.session_state.transform_pipeline.pop(index)
    st.toast(f"❌ Removed '{removed_op}' from pipeline", icon="🗑️")

def toggle_info(category):
    if category in st.session_state.info_pipeline:
        st.session_state.info_pipeline.remove(category)
    else:
        st.session_state.info_pipeline.append(category)

def move_transform_up(index):
    """Swaps a pipeline step with the one above it."""
    if index > 0:
        pipeline = st.session_state.transform_pipeline
        pipeline[index - 1], pipeline[index] = pipeline[index], pipeline[index - 1]
        st.toast(f"⬆️ Moved '{pipeline[index - 1]['op']}' up")

def move_transform_down(index):
    """Swaps a pipeline step with the one below it."""
    pipeline = st.session_state.transform_pipeline
    if index < len(pipeline) - 1:
        pipeline[index + 1], pipeline[index] = pipeline[index], pipeline[index + 1]
        st.toast(f"⬇️ Moved '{pipeline[index + 1]['op']}' down")

def process_and_clear_selections():
    """
    Callback function attached to the Add button.
    Reads the multiselects, adds to pipeline, and resets the widgets safely.
    """
    for cat_name in TRANSFORM_CATEGORIES.keys():
        widget_key = f"multiselect_transform_{cat_name}"
        # Get selected operations from this category's multiselect
        selected_ops = st.session_state.get(widget_key, [])
        
        # Add them to the pipeline
        for op in selected_ops:
            add_transform(op)
            
        # Clear the widget's session state safely before it redraws
        st.session_state[widget_key] = []

def get_val(step_dict, param_key, default_value):
    """Safely extracts a loaded parameter value, or returns the default."""
    # .get() won't crash if 'loaded_params' or the param_key doesn't exist
    return step_dict.get('loaded_params', {}).get(param_key, default_value)

def get_idx(step_dict, param_key, options_list, default_index=0):
    """Safely finds the index of a loaded string for st.selectbox."""
    val = step_dict.get('loaded_params', {}).get(param_key)
    if val in options_list:
        return options_list.index(val)
    return default_index

# --- SIDEBAR: ALL CONTROLS ---
with st.sidebar:
    st.header("1. Input & Output")
    
    # Define explicit tuples for categorization later
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    VIDEO_EXTS = (".mp4", ".mov", ".avi")
    
    # Add video extensions to the uploader type parameter
    uploaded_files = st.file_uploader(
        "Upload Media (Select multiple for batch)", 
        type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "mov", "avi"], 
        accept_multiple_files=True
    )

    output_folder = st.text_input("Output Path", value="./outputs", help="Directory where processed files will be saved relative to the project folder (e.g., ./outputs, ./results). Will be created if it doesn't exist.")

    if uploaded_files:
        st.success(f"Successfully loaded {len(uploaded_files)} image(s) or video(s).")
        preview_bytes = uploaded_files[0].getvalue()
        # if uploaded file is an image
        if uploaded_files[0].type.startswith("image/"):
            with Image.open(io.BytesIO(preview_bytes)) as sample_image:
                preview_width, preview_height = sample_image.size
    else:
        st.warning("Awaiting image uploads. Please select files to begin.")
    
    st.header("2. Build Pipelines")
    
    # --- Load Pipeline UI ---
    uploaded_pipeline = st.file_uploader(
        "Load Previous Pipeline (JSON)", 
        type=["json"], 
        help="Upload a pipeline_config.json file from a previous session."
    )
    
    if uploaded_pipeline is not None:
        # We use a button to confirm the load, preventing accidental overwrites
        if st.button("📂 Apply Uploaded Pipeline", use_container_width=True):
            fn.load_pipeline_from_json(uploaded_pipeline)
            st.rerun() # Refresh the UI to show the newly loaded steps
    
    st.divider() # Keep things visually clean
    
    # Information Pipeline
    with st.expander("📊 Select Information to Display"):
        info_labels = [cfg["label"] for cfg in INFO_CATEGORIES.values()]
        label_to_key = {cfg["label"]: key for key, cfg in INFO_CATEGORIES.items()}
        selected_labels = st.multiselect(
            "Choose which information categories to show",
            options=info_labels,
            default=info_labels,
            key="info_category_labels"
        )
        st.session_state.info_pipeline = [label_to_key[label] for label in selected_labels if label in label_to_key]

    # Transformation Pipeline UI
    with st.expander("➕ Select Transformations to Apply"):
        st.write("Choose operations across categories, then click add to send them to the pipeline.")
        
        # Dynamically generate a multiselect for each category
        for cat_name, cfg in TRANSFORM_CATEGORIES.items():
            valid_options = [k for k in cfg["keys"] if k is not None]
            
            # The widget automatically ties to st.session_state[key]
            st.multiselect(
                cfg["label"],
                options=valid_options,
                key=f"multiselect_transform_{cat_name}"
            )

        # Use on_click callback to process and clear BEFORE the rerun
        st.button(
            "Add Selected Transformations to Pipeline", 
            type="primary", 
            width='stretch',
            on_click=process_and_clear_selections
        )

    st.header("3. Configure Active Pipeline")

    if len(st.session_state.transform_pipeline) == 0:
        st.caption("Pipeline is empty.")
    
    # Store discrete configurations for batch processing
    pipeline_config = []

    for i, step in enumerate(st.session_state.transform_pipeline):
        op = step['op']
        step_id = step['id']
        
        with st.container(border=True):
            # Create a 4-column layout for Title, Up, Down, Delete
            cols = st.columns([0.7, 0.1, 0.1, 0.1])
            cols[0].markdown(f"**{i+1}. {op}**")
            
            # Move Up button (disabled if it's the first item)
            if cols[1].button("⬆️", key=f"up_{step_id}", disabled=(i == 0), help="Move step up"):
                move_transform_up(i)
                st.rerun()
                
            # Move Down button (disabled if it's the last item)
            if cols[2].button("⬇️", key=f"dw_{step_id}", disabled=(i == len(st.session_state.transform_pipeline) - 1), help="Move step down"):
                move_transform_down(i)
                st.rerun()
                
            # Delete button
            if cols[3].button("❌", key=f"del_{step_id}", help="Remove step"):
                remove_transform(i)
                st.rerun()

            params = {}
            max_translate_x = preview_width if preview_width else 10000
            max_translate_y = preview_height if preview_height else 10000
            max_crop_x = max(0, preview_width - 1) if preview_width else 10000
            max_crop_y = max(0, preview_height - 1) if preview_height else 10000
            max_crop_w = preview_width if preview_width else 10000
            max_crop_h = preview_height if preview_height else 10000
            max_ksize = min(preview_width, preview_height, 99) if preview_width and preview_height else 99

            p_w = preview_width if preview_width else 10000
            p_h = preview_height if preview_height else 10000

            if max_ksize % 2 == 0:
                max_ksize -= 1
            max_ksize = max(1, max_ksize)

            if op == "translate_image":
                params['tx'] = st.number_input(
                    "Tx", 
                    min_value=-max_translate_x, max_value=max_translate_x, 
                    value=get_val(step, 'tx', 0), # <--- Injecting saved text
                    step=10, key=f"tx_{step_id}",
                    help="Moves the image horizontally."
                )
                params['ty'] = st.number_input(
                    "Ty", 
                    min_value=-max_translate_y, max_value=max_translate_y, 
                    value=get_val(step, 'ty', 0), # <--- Injecting saved text
                    step=10, key=f"ty_{step_id}",
                    help="Moves the image vertically."
                )
                
            elif op == "resize_image":
                # Number inputs use get_val()
                params['width'] = st.number_input(
                    "New Width (Pixels)", 
                    min_value=1, max_value=20000, 
                    value=get_val(step, 'width', p_w), # <--- Injecting saved text
                    step=1, key=f"rw_{step_id}",
                    help="Sets the new width of the image."
                )
                params['height'] = st.number_input(
                    "New Height (Pixels)", 
                    min_value=1, max_value=20000, 
                    value=get_val(step, 'height', p_h), # <--- Injecting saved text
                    step=1, key=f"rh_{step_id}",
                    help="Sets the new height of the image."
                )
                # Selectboxes use get_idx()
                interp_options = ["INTER_LINEAR", "INTER_NEAREST", "INTER_CUBIC"]
                params['interpolation'] = st.selectbox(
                    "Interpolation Method", 
                    options=interp_options, 
                    index=get_idx(step, 'interpolation', interp_options, 0), # <--- Injecting saved selectbox choice
                    key=f"int_{step_id}",
                    help="Determines the algorithm used for resizing. INTER_LINEAR is a good default for most cases. INTER_NEAREST is faster but lower quality. INTER_CUBIC is slower but can produce smoother results."
                )

            elif op == "rotate_image":
                params['angle'] = st.number_input(
                    "Angle (degrees)", 
                    min_value=-360, 
                    max_value=360, 
                    value=get_val(step, 'angle', 0), 
                    step=5, 
                    key=f"rot_{step_id}",
                    help="Rotates the image around its exact center. Positive values rotate clockwise.")
                
            elif op == "scale_image":
                params['scale_factor'] = st.number_input("Scale Factor", min_value=0.1, max_value=10.0, value=get_val(step, 'scale_factor', 1.0), step=0.1, key=f"scale_{step_id}",
                    help="Scales the image by the specified factor.")
                
            elif op == "shear_image":
                params['shx'] = st.number_input("Shear X", min_value=-5.0, max_value=5.0, value=get_val(step, 'shx', 0.0), step=0.1, key=f"shx_{step_id}",
                    help="Shears the image along the X-axis.")
                params['shy'] = st.number_input("Shear Y", min_value=-5.0, max_value=5.0, value=get_val(step, 'shy', 0.0), step=0.1, key=f"shy_{step_id}",
                    help="Shears the image along the Y-axis.")
                
            elif op == "square_image":
                params['mode'] = st.selectbox(
                    "Squaring Strategy", 
                    ["Crop", "Pad"], 
                    index=get_idx(step, 'mode', ["Crop", "Pad"], 0),
                    key=f"sq_mode_{step_id}",
                    help="Choose how to force the 1:1 aspect ratio. 'Crop' finds the shortest side and chops off the excess from the center. 'Pad' finds the longest side and adds black bars to the shorter side to fill it out.")
                
            elif op == "flip_image":
                params['mode'] = st.selectbox("Flip Mode", ["Horizontal", "Vertical", "Both"], index=get_idx(step, 'mode', ["Horizontal", "Vertical", "Both"], 0), key=f"flip_{step_id}",
                    help="Horizontal flips left-to-right. Vertical flips top-to-bottom. Both flips in both directions.")
            
            elif op == "crop_image":
                params['x'] = st.number_input("X", min_value=0, max_value=max_crop_x, value=get_val(step, 'x', 0), step=1, key=f"cropx_{step_id}",
                    help="X coordinate of the top-left corner of the crop region.")
                params['y'] = st.number_input("Y", min_value=0, max_value=max_crop_y, value=get_val(step, 'y', 0), step=1, key=f"cropy_{step_id}",
                    help="Y coordinate of the top-left corner of the crop region.")
                params['w'] = st.number_input("Width", min_value=1, max_value=max_crop_w, value=get_val(step, 'w', min(100, max_crop_w)), step=1, key=f"cropw_{step_id}",
                    help="Width of the crop region.")
                params['h'] = st.number_input("Height", min_value=1, max_value=max_crop_h, value=get_val(step, 'h', min(100, max_crop_h)), step=1, key=f"croph_{step_id}",
                    help="Height of the crop region.")
            
            elif op == "adjust_brightness_contrast":
                params['alpha'] = st.number_input("Alpha (Contrast)", min_value=0.0, max_value=10.0, value=get_val(step, 'alpha', 1.0), step=0.1, key=f"alpha_{step_id}",
                    help="Values > 1 increase contrast (brights brighter, darks darker). Values < 1 decrease it. 1.0 is original.")
                params['beta'] = st.number_input("Beta (Brightness)", min_value=-255, max_value=255, value=get_val(step, 'beta', 0), step=1, key=f"beta_{step_id}",
                    help="Adds a flat value to every pixel. > 0 lightens the image, < 0 darkens it. Values cap strictly at 0 and 255.")
            
            elif op == "apply_gamma":
                params['gamma'] = st.number_input("Gamma", min_value=0.1, max_value=10.0, value=get_val(step, 'gamma', 1.0), step=0.1, key=f"gamma_{step_id}",
                    help="Adjusts the gamma correction of the image.")
            
            elif op == "apply_histogram_equalization":
                params['method'] = st.selectbox("Method", ["Global", "CLAHE"], index=get_idx(step, 'method', ["Global", "CLAHE"], 0), key=f"he_{step_id}",
                    help="Global applies histogram equalization to the entire image. CLAHE applies adaptive histogram equalization to local regions.")
            
            elif op == "apply_thresholding":
                params['thresh'] = st.number_input("Threshold Value", min_value=0, max_value=255, value=get_val(step, 'thresh', 127), step=1, key=f"thresh_{step_id}",
                    help="The threshold value for binarization.")
                params['type'] = st.selectbox("Threshold Type", ["Binary", "Binary Inverted", "Truncate", "To Zero", "To Zero Inverted"], index=get_idx(step, 'type', ["Binary", "Binary Inverted", "Truncate", "To Zero", "To Zero Inverted"], 0), key=f"thtype_{step_id}",
                    help="The type of thresholding to apply.")
            
            elif op == "apply_posterization":
                params['bits'] = st.number_input("bits", min_value=1, max_value=8, value=get_val(step, 'bits', 4), step=1, key=f"post_{step_id}",
                    help="Number of bits to keep for each color channel. Reduces the number of colors in the image.")
            
            elif op == "apply_gaussian_blur":
                params['ksize'] = st.number_input("Kernel Size", min_value=1, max_value=max_ksize, value=get_val(step, 'ksize', min(5, max_ksize)), step=2, key=f"gblur_{step_id}",
                    help="Size of the kernel. Must be an odd number.")
            
            elif op == "apply_median_blur":
                params['ksize'] = st.number_input("Kernel Size", min_value=1, max_value=max_ksize, value=get_val(step, 'ksize', min(5, max_ksize)), step=2, key=f"mblur_{step_id}",
                    help="Size of the kernel. Must be an odd number.")
            
            elif op == "apply_canny":
                params['t1'] = st.number_input("Threshold 1", min_value=0, max_value=255, value=get_val(step, 't1', 100), step=1, key=f"canny1_{step_id}",
                    help="The lower threshold for edge detection.")
                params['t2'] = st.number_input("Threshold 2", min_value=0, max_value=255, value=get_val(step, 't2', 200), step=1, key=f"canny2_{step_id}",
                    help="The upper threshold for edge detection.")
            
            elif op == "apply_morphology":
                params['operation'] = st.selectbox("Operation", ["Erosion", "Dilation", "Opening", "Closing", "Gradient"], index=get_idx(step, 'operation', ["Erosion", "Dilation", "Opening", "Closing", "Gradient"], 0), key=f"morph_op_{step_id}",
                    help="The morphological operation to apply.")
                params['ksize'] = st.number_input("Kernel Size", min_value=1, max_value=max_ksize, value=get_val(step, 'ksize', min(5, max_ksize)), step=2, key=f"morph_ksize_{step_id}",
                    help="Size of the kernel. Must be an odd number.")
                params['shape'] = st.selectbox("Kernel Shape", ["Rect", "Ellipse", "Cross"], index=get_idx(step, 'shape', ["Rect", "Ellipse", "Cross"], 0), key=f"morph_shape_{step_id}",
                    help="The shape of the kernel.")
            
            elif op == "apply_dwt":
                params['wavelet'] = st.selectbox("Wavelet", ["haar", "db1", "db2", "sym2", "coif1"], index=get_idx(step, 'wavelet', ["haar", "db1", "db2", "sym2", "coif1"], 0), key=f"dwt_{step_id}",
                    help="The wavelet to use for the discrete wavelet transform.")
            
            elif op == "add_gaussian_noise":
                params['mean'] = st.number_input("Mean", min_value=-100.0, max_value=100.0, value=get_val(step, 'mean', 0.0), step=0.1, key=f"noise_mean_{step_id}",
                    help="The mean of the Gaussian noise to add.")
                params['std'] = st.number_input("Standard Deviation", min_value=0.0, max_value=100.0, value=get_val(step, 'std', 0.1), step=0.1, key=f"noise_std_{step_id}",
                    help="The standard deviation of the Gaussian noise to add.")
            
            elif op == "add_salt_pepper_noise":
                params['salt_prob'] = st.number_input("Salt Probability", min_value=0.0, max_value=1.0, value=get_val(step, 'salt_prob', 0.01), step=0.01, key=f"sp_salt_{step_id}",
                    help="The probability of adding salt noise.")
                params['pepper_prob'] = st.number_input("Pepper Probability", min_value=0.0, max_value=1.0, value=get_val(step, 'pepper_prob', 0.01), step=0.01, key=f"sp_pepper_{step_id}",
                    help="The probability of adding pepper noise.")
            
            elif op == "add_jpeg_artifacts":
                params['quality'] = st.number_input("Quality", min_value=1, max_value=100, value=get_val(step, 'quality', 50), step=1, key=f"jpeg_{step_id}",
                    help="The quality of the JPEG image (1-100).")
            
            # --- FACE CONTROLS ---
            elif op == "align_face":
                params['id1'] = st.number_input("Landmark 1 ID", min_value=0, max_value=477, value=get_val(step, 'id1', 33), key=f"fa1_{step_id}",
                    help="Hover over the Main Output image on the right to find the Landmark ID (red dot) you want to target.")
                params['id2'] = st.number_input("Landmark 2 ID", min_value=0, max_value=477, value=get_val(step, 'id2', 263), key=f"fa2_{step_id}")
                params['t1_x'] = st.number_input("Target 1 X", value=get_val(step, 't1_x', int(p_w*0.3)), key=f"fat1x_{step_id}", help="Where Landmark 1 should move horizontally.")
                params['t1_y'] = st.number_input("Target 1 Y", value=get_val(step, 't1_y', int(p_h*0.4)), key=f"fat1y_{step_id}", help="Where Landmark 1 should move vertically.")
                params['t2_x'] = st.number_input("Target 2 X", value=get_val(step, 't2_x', int(p_w*0.7)), key=f"fat2x_{step_id}", help="Where Landmark 2 should move horizontally.")
                params['t2_y'] = st.number_input("Target 2 Y", value=get_val(step, 't2_y', int(p_h*0.4)), key=f"fat2y_{step_id}", help="Where Landmark 2 should move vertically.")
            
            elif op == "advanced_crop_face":
                params['bb_type'] = st.selectbox("Crop Shape", 
                    ["Minimum Rectangle", "Minimum Square", "Minimum Oval", "Polygonal"], 
                    index=get_idx(step, 'bb_type', ["Minimum Rectangle", "Minimum Square", "Minimum Oval", "Polygonal"], 0),
                    key=f"ac_bb_{step_id}",
                    help="Defines the geometry of the mask used to isolate the face.")
                
                # Show string input ONLY if Polygonal is selected
                if params['bb_type'] == "Polygonal":
                    params['poly_string'] = st.text_input("Landmark IDs (Comma separated)", 
                        value=get_val(step, 'poly_string', "10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109"), 
                        key=f"ac_poly_{step_id}",
                        help="Enter at least 3 Landmark IDs separated by commas. Hover over the main output mesh to find IDs. Example creates an octagon around the face.")
                else:
                    params['poly_string'] = "" # Pass empty if not used
                
                params['padding'] = st.number_input("Padding (Pixels)", min_value=0, max_value=500, value=get_val(step, 'padding', 0), step=5, key=f"ac_pad_{step_id}",
                    help="Puffs the shape outwards. 0 hugs the face exactly. Higher values include more of the background (hair, neck).")
                
                params['exterior_mode'] = st.selectbox("Exterior Action", ["Cut Out", "Set Exterior to 0"], index=get_idx(step, 'exterior_mode', ["Cut Out", "Set Exterior to 0"], 0), key=f"ac_ext_{step_id}",
                    help="'Cut Out' physically slices the image file down to the masked size. 'Set to 0' leaves the image resolution unchanged but blacks out the background.")
                
            pipeline_config.append({'op': op, 'params': params})
            
            # Clear loaded_params so the user can freely edit the injected values now
            if 'loaded_params' in step:
                del step['loaded_params']

    # ==========================================
    # BATCH PROCESSING TRIGGER
    # ==========================================

    st.divider()
    st.header("4. Media Settings")
    
    # Requirement 3: Filter for batch processing
    media_filter = st.selectbox(
        "Files to process in batch:",
        ["Both Images and Videos", "Only Images", "Only Videos"],
        help="Filters the files in the upload list or target directory."
    )
    
    # Requirement 2: Video output format
    video_output_mode = st.radio(
        "Video Processing Output:",
        [
            "Output processed video file (.mp4)", 
            "Save frames as individual images",
            "Both (Video file AND individual frames)" # <--- ADD THIS LINE
        ],
        help="Choose how processed videos are saved to the output folder."
    )

    if st.button("🚀 Process Loaded Files", type="primary", width='stretch'):
        if not uploaded_files:
            st.error("Please upload files first.")
        else:
            os.makedirs(output_folder, exist_ok=True)
            
            # 1. Filter files based on user choice
            files_to_process = []
            for f in uploaded_files:
                name_lower = f.name.lower()
                is_vid = name_lower.endswith(VIDEO_EXTS)
                is_img = name_lower.endswith(IMAGE_EXTS)
                
                if media_filter == "Only Images" and is_img:
                    files_to_process.append(f)
                elif media_filter == "Only Videos" and is_vid:
                    files_to_process.append(f)
                elif media_filter == "Both Images and Videos" and (is_img or is_vid):
                    files_to_process.append(f)
                    
            if not files_to_process:
                st.warning("No files matched the selected media filter.")
            else:
                progress_bar = st.progress(0, text="Overall Batch Progress")
                
                # 2. Iterate through filtered list
                for idx, file in enumerate(files_to_process):
                    is_vid = file.name.lower().endswith(VIDEO_EXTS)
                    
                    if is_vid:
                        # --- VIDEO LOGIC ---
                        print(f"[LOG] Routing {file.name} to video processor.")
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tfile.write(file.read())
                        tfile.close()
                        
                        # Call our helper function
                        fn.process_video_file(tfile.name, file.name, output_folder, pipeline_config, video_output_mode, fn)
                        os.remove(tfile.name) # Clean up
                    else:
                        # --- IMAGE LOGIC ---
                        print(f"[LOG] Routing {file.name} to image processor.")
                        img = np.array(Image.open(file))

                        # Apply pipeline
                        for step in pipeline_config:
                            op, p = step['op'], step['params']
                            if not op: continue
                            func = getattr(fn, op, None)
                            if func is not None:
                                try:
                                    img = func(img, **p)
                                except Exception as e:
                                    # Log the exact failure gracefully
                                    filename = file.name if hasattr(file, 'name') else f"Image {idx+1}"
                                    st.error(f"Failed to apply '{op}' on image '{filename}'. Error: {str(e)}")
                                    break # Optional: break out of the pipeline for this specific image if a step fails
                                    
                        # Save
                        save_path = os.path.join(output_folder, f"mod_{file.name}")
                        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape)==3 else img)

                    progress_bar.progress((idx + 1) / len(files_to_process), text=f"Overall Batch Progress: {idx + 1}/{len(files_to_process)}")

            # Save Log as JSON
            log_path = os.path.join(output_folder, "pipeline_config.json")
            try:
                with open(log_path, "w") as f:
                    # pipeline_config is already a list of dictionaries!
                    json.dump(pipeline_config, f, indent=4)
                st.success(f"Batch completed! Pipeline configuration saved to {log_path}")
            except Exception as e:
                st.error(f"Batch completed, but failed to save JSON config. Error: {e}")

    st.markdown("<div style='text-align: center; margin: 10px 0;'><b>OR</b></div>", unsafe_allow_html=True)

    # --- Process 2: Local Directory Images ---
    input_folder = st.text_input("Target Local Folder Path", value="./input_files", help="Enter the path to a local folder containing files to process.")
    
    if st.button("🚀 Process Local Folder", type="primary", width='stretch'):
        input_path = Path(input_folder)
        
        if not input_folder or not input_path.is_dir():
            st.error("Please provide a valid input folder path.")
        else:
            os.makedirs(output_folder, exist_ok=True)
            
            # 1. Filter files based on user choice across ALL subfolders
            files_to_process = []
            
            # .rglob('*') recursively searches all files and folders
            for file_path in input_path.rglob('*'):
                if file_path.is_file():
                    name_lower = file_path.name.lower()
                    is_vid = name_lower.endswith(VIDEO_EXTS)
                    is_img = name_lower.endswith(IMAGE_EXTS)
                    
                    if media_filter == "Only Images" and is_img:
                        files_to_process.append(file_path)
                    elif media_filter == "Only Videos" and is_vid:
                        files_to_process.append(file_path)
                    elif media_filter == "Both Images and Videos" and (is_img or is_vid):
                        files_to_process.append(file_path)
                        
            if not files_to_process:
                st.warning("No files matched the selected media filter in the given folder or its subfolders.")
            else:
                progress_bar = st.progress(0, text="Overall Batch Progress")
                
                # 2. Iterate through filtered list
                for idx, file_path in enumerate(files_to_process):
                    is_vid = file_path.name.lower().endswith(VIDEO_EXTS)
                    
                    # Determine relative path to recreate the subfolder structure in the output directory
                    rel_path = file_path.relative_to(input_path)
                    file_output_dir = Path(output_folder) / rel_path.parent
                    file_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    if is_vid:
                        # --- VIDEO LOGIC ---
                        print(f"[LOG] Routing {file_path.name} to video processor.")
                        
                        # No temporary file needed! We just pass the local file path directly.
                        fn.process_video_file(
                            str(file_path), 
                            file_path.name, 
                            str(file_output_dir), 
                            pipeline_config, 
                            video_output_mode, 
                            fn
                        )
                    else:
                        # --- IMAGE LOGIC ---
                        print(f"[LOG] Routing {file_path.name} to image processor.")
                        img = np.array(Image.open(file_path))

                        # Apply pipeline
                        for step in pipeline_config:
                            op, p = step['op'], step['params']
                            if not op: continue
                            func = getattr(fn, op, None)
                            if func is not None:
                                try:
                                    img = func(img, **p)
                                except Exception as e:
                                    # Log the exact failure gracefully
                                    st.error(f"Failed to apply '{op}' on image '{file_path.name}'. Error: {str(e)}")
                                    break 
                                    
                        # Save
                        save_path = file_output_dir / f"mod_{file_path.name}"
                        cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape)==3 else img)

                    progress_bar.progress((idx + 1) / len(files_to_process), text=f"Overall Batch Progress: {idx + 1}/{len(files_to_process)}")

            # Save Log as JSON
            log_path = os.path.join(output_folder, "pipeline_config.json")
            try:
                with open(log_path, "w") as f:
                    json.dump(pipeline_config, f, indent=4)
                st.success(f"Batch completed! Pipeline configuration saved to {log_path}")
            except Exception as e:
                st.error(f"Batch completed, but failed to save JSON config. Error: {e}")

    st.markdown("<div style='text-align: center; margin: 10px 0;'><b>OR</b></div>", unsafe_allow_html=True)
# --- MAIN LAYOUT (Side-by-Side View) ---

st.header("Live Preview & Information")
st.text("Apply transformations and see the results in real-time.")

# Render Plotly Preview using the ACTIVE (edited) landmarks and the new toggle state
show_coords_toggle = st.toggle("🔎 Enable Hover Coordinates (Slower rendering)", value=False)

if uploaded_files:
    preview_file = uploaded_files[0]
    file_name = preview_file.name.lower()
    
    # Check if the primary file is a video
    is_video = file_name.endswith(VIDEO_EXTS)
    
    if is_video:
        print(f"[LOG] Video detected for preview: {file_name}")
        
        preview_bytes = preview_file.getvalue() 
        pil_img = None # We set this to None because a video frame isn't a traditional PIL file

        # 1. Write the uploaded bytes to a temporary file so OpenCV can read it
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        
        tfile.write(preview_bytes)
        tfile.close() # Close so OpenCV can safely open it
        
        # 2. Open video and get total frame count
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            st.error("Could not read video frames. The file might be corrupted.")
            original_img = np.zeros((500, 500, 3), dtype=np.uint8) # Fallback blank image
        else:
            # 3. Ask user which frame to preview
            st.info(f"📹 Video loaded. Total frames: {total_frames}")
            selected_frame = st.slider("Select Frame for Live Preview", 0, total_frames - 1, 0)
            
            # 4. Seek to the selected frame and read it
            cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
            ret, frame = cap.read()
            
            if ret:
                # OpenCV reads in BGR, Streamlit/PIL expects RGB
                original_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                print(f"[LOG] Successfully extracted frame {selected_frame}")
            else:
                st.error("Failed to read the selected frame.")
                original_img = np.zeros((500, 500, 3), dtype=np.uint8)
                
        cap.release()
        try:
            os.remove(tfile.name) # Clean up temp file
        except OSError:
            pass

    else:
        # Existing Image Logic
        print(f"[LOG] Image detected for preview: {file_name}")
        preview_bytes = preview_file.getvalue()
        pil_img = Image.open(io.BytesIO(preview_bytes))
        original_img = np.array(pil_img)


    original_landmarks = fn.get_face_landmarks(original_img)
    if original_landmarks is None:
        st.warning("No face landmarks detected in the preview image. Face-related transformations will be disabled.")

    # Process the preview image based on active pipeline
    processed_img = original_img.copy()
    
    # Track which landmarks the user is actively manipulating so we can color them green
    highlight_landmarks = []
    
    for step in pipeline_config:
        op, p = step['op'], step['params']
        if not op: continue
        func = getattr(fn, op, None)
        if func is not None:
            processed_img = func(processed_img, **p)
            
        # Collect IDs for highlighting
        if op == "align_face":
            highlight_landmarks.extend([p['id1'], p['id2']])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        original_show_landmark_toggle = st.toggle("👁️ Show Detected Landmarks", value=False, key="original_show_landmark")

        # Render Plotly Preview using the toggle state
        if original_show_landmark_toggle:
            fig_orig = fn.create_main_preview(
                original_img, 
                landmarks=original_landmarks, 
                highlight_ids=highlight_landmarks, 
                show_coords=show_coords_toggle
            )
        else:
            fig_orig = fn.create_main_preview(
                original_img, 
                landmarks=None, 
                highlight_ids=highlight_landmarks, 
                show_coords=show_coords_toggle
            )
        st.plotly_chart(fig_orig, width='stretch', height='content', key="original_preview")
        
    with col2:
        st.subheader("Modified Preview (Interactive)")
        show_landmark_toggle = st.toggle("👁️ Show Detected Landmarks", value=False, key="show_landmark")
        
        # Run landmark detection on the CURRENT state of the modified image
        current_landmarks = fn.get_face_landmarks(processed_img)
        
        # Render Plotly Preview using the toggle state
        if show_landmark_toggle:
            fig = fn.create_main_preview(
                processed_img, 
                landmarks=current_landmarks, 
                highlight_ids=highlight_landmarks, 
                show_coords=show_coords_toggle
            )
        else:
            fig = fn.create_main_preview(
                processed_img, 
                landmarks=None, 
                highlight_ids=highlight_landmarks, 
                show_coords=show_coords_toggle
            )
        st.plotly_chart(fig, width='stretch', height='content', key="modified_preview")

    orig_stats = fn.get_image_stats(original_img, pil_img, preview_bytes)
    mod_stats = fn.get_image_stats(processed_img)
    selected_categories = st.session_state.info_pipeline

    if selected_categories:
        st.divider()
        st.subheader("Selected Image Information")
        colA, colB = st.columns(2)

        def render_info_block(title, stats_dict, keys_to_render, is_chart=False):
            st.markdown(f"**{title}**")
            if is_chart:
                st.bar_chart(stats_dict['Histogram Data'])
            else:
                for k in keys_to_render:
                    st.markdown(f"- **{k}:** {stats_dict.get(k, 'N/A')}")

        for category in selected_categories:
            category_info = INFO_CATEGORIES.get(category)
            if not category_info:
                continue
            keys = category_info["keys"]
            is_chart = category == "Structural"
            with colA:
                render_info_block(category_info["label"], orig_stats, keys, is_chart=is_chart)
            with colB:
                render_info_block(category_info["label"], mod_stats, keys, is_chart=is_chart)

    st.divider()
    
else:
    st.info("👈 Please upload one or more images from the sidebar to begin building your workflow.")


