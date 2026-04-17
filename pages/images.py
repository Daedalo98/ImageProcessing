import streamlit as st
import cv2
import numpy as np
import os
import io
from PIL import Image
import uuid
import img_functions as fn

# ==========================================
# STREAMLIT UI ARCHITECTURE
# ==========================================

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

def remove_transform(index):
    st.session_state.transform_pipeline.pop(index)

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

def move_transform_down(index):
    """Swaps a pipeline step with the one below it."""
    pipeline = st.session_state.transform_pipeline
    if index < len(pipeline) - 1:
        pipeline[index + 1], pipeline[index] = pipeline[index], pipeline[index + 1]

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

# --- SIDEBAR: ALL CONTROLS ---
with st.sidebar:
    st.header("1. Input & Output")
    uploaded_files = st.file_uploader("Upload Images (Select multiple for batch)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    output_folder = st.text_input("Output Path", value="./output_images")

    preview_width = preview_height = None
    if uploaded_files:
        preview_bytes = uploaded_files[0].getvalue()
        with Image.open(io.BytesIO(preview_bytes)) as sample_image:
            preview_width, preview_height = sample_image.size
    
    st.header("2. Build Pipelines")
    
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
            use_container_width=True, 
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
            max_translate_x = preview_width if preview_width else 500
            max_translate_y = preview_height if preview_height else 500
            max_crop_x = max(0, preview_width - 1) if preview_width else 10000
            max_crop_y = max(0, preview_height - 1) if preview_height else 10000
            max_crop_w = preview_width if preview_width else 10000
            max_crop_h = preview_height if preview_height else 10000
            max_ksize = min(preview_width, preview_height, 99) if preview_width and preview_height else 99

            p_w = preview_width if preview_width else 1000
            p_h = preview_height if preview_height else 1000

            if max_ksize % 2 == 0:
                max_ksize -= 1
            max_ksize = max(1, max_ksize)

            if op == "translate_image":
                params['tx'] = st.number_input("Tx", min_value=-max_translate_x, max_value=max_translate_x, value=0, step=10, key=f"tx_{step_id}",
                    help="Moves the image horizontally. Positive values shift right, negative values shift left. Empty space is filled with black.")
                params['ty'] = st.number_input("Ty", min_value=-max_translate_y, max_value=max_translate_y, value=0, step=10, key=f"ty_{step_id}",
                    help="Moves the image vertically. Positive values shift down, negative values shift up.")
            elif op == "rotate_image":
                params['angle'] = st.number_input("Angle (degrees)", min_value=-360, max_value=360, value=0, step=5, key=f"rot_{step_id}",
                    help="Rotates the image around its exact center. Positive values rotate counter-clockwise.")
            elif op == "scale_image":
                params['scale_factor'] = st.number_input("Scale Factor", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key=f"scale_{step_id}",
                    help="Scales the image by the specified factor.")
            elif op == "resize_image":
                params['width'] = st.number_input(
                    "New Width (Pixels)", 
                    min_value=1, 
                    max_value=20000, 
                    value=p_w, 
                    step=1, 
                    key=f"rw_{step_id}",
                    help="Absolute width of the output image in pixels."
                )
                params['height'] = st.number_input(
                    "New Height (Pixels)", 
                    min_value=1, 
                    max_value=20000, 
                    value=p_h, 
                    step=1, 
                    key=f"rh_{step_id}",
                    help="Absolute height of the output image in pixels."
                )
                params['interpolation'] = st.selectbox(
                    "Interpolation Method", 
                    ["INTER_LINEAR", "INTER_NEAREST", "INTER_CUBIC"], 
                    key=f"int_{step_id}",
                    help="LINEAR is standard. NEAREST prevents blurring (great for pixel art/masks). CUBIC is highest quality for making images larger."
                )
            elif op == "shear_image":
                params['shx'] = st.number_input("Shear X", min_value=-5.0, max_value=5.0, value=0.0, step=0.1, key=f"shx_{step_id}",
                    help="Shears the image along the X-axis.")
                params['shy'] = st.number_input("Shear Y", min_value=-5.0, max_value=5.0, value=0.0, step=0.1, key=f"shy_{step_id}",
                    help="Shears the image along the Y-axis.")
            elif op == "square_image":
                params['mode'] = st.selectbox(
                    "Squaring Strategy", 
                    ["Crop", "Pad"], 
                    key=f"sq_mode_{step_id}",
                    help="Choose how to force the 1:1 aspect ratio. 'Crop' finds the shortest side and chops off the excess from the center. 'Pad' finds the longest side and adds black bars to the shorter side to fill it out.")
            elif op == "flip_image":
                params['mode'] = st.selectbox("Flip Mode", ["Horizontal", "Vertical", "Both"], key=f"flip_{step_id}",
                    help="Horizontal flips left-to-right. Vertical flips top-to-bottom. Both flips in both directions.")
            elif op == "crop_image":
                params['x'] = st.number_input("X", min_value=0, max_value=max_crop_x, value=0, step=1, key=f"cropx_{step_id}",
                    help="X coordinate of the top-left corner of the crop region.")
                params['y'] = st.number_input("Y", min_value=0, max_value=max_crop_y, value=0, step=1, key=f"cropy_{step_id}",
                    help="Y coordinate of the top-left corner of the crop region.")
                params['w'] = st.number_input("Width", min_value=1, max_value=max_crop_w, value=min(100, max_crop_w), step=1, key=f"cropw_{step_id}",
                    help="Width of the crop region.")
                params['h'] = st.number_input("Height", min_value=1, max_value=max_crop_h, value=min(100, max_crop_h), step=1, key=f"croph_{step_id}",
                    help="Height of the crop region.")
            elif op == "adjust_brightness_contrast":
                params['alpha'] = st.number_input("Alpha (Contrast)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key=f"alpha_{step_id}",
                    help="Values > 1 increase contrast (brights brighter, darks darker). Values < 1 decrease it. 1.0 is original.")
                params['beta'] = st.number_input("Beta (Brightness)", min_value=-255, max_value=255, value=0, step=1, key=f"beta_{step_id}",
                    help="Adds a flat value to every pixel. > 0 lightens the image, < 0 darkens it. Values cap strictly at 0 and 255.")
            elif op == "apply_gamma":
                params['gamma'] = st.number_input("Gamma", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key=f"gamma_{step_id}",
                    help="Adjusts the gamma correction of the image.")
            elif op == "apply_histogram_equalization":
                params['method'] = st.selectbox("Method", ["Global", "CLAHE"], key=f"he_{step_id}",
                    help="Global applies histogram equalization to the entire image. CLAHE applies adaptive histogram equalization to local regions.")
            elif op == "apply_thresholding":
                params['thresh'] = st.number_input("Threshold Value", min_value=0, max_value=255, value=127, step=1, key=f"thresh_{step_id}",
                    help="The threshold value for binarization.")
                params['type'] = st.selectbox("Threshold Type", ["Binary", "Binary Inverted", "Truncate", "To Zero", "To Zero Inverted"], key=f"thtype_{step_id}",
                    help="The type of thresholding to apply.")
            elif op == "apply_posterization":
                params['bits'] = st.number_input("bits", min_value=1, max_value=8, value=4, step=1, key=f"post_{step_id}",
                    help="Number of bits to keep for each color channel. Reduces the number of colors in the image.")
            elif op == "apply_gaussian_blur":
                params['ksize'] = st.number_input("Kernel Size", min_value=1, max_value=max_ksize, value=min(5, max_ksize), step=2, key=f"gblur_{step_id}",
                    help="Size of the kernel. Must be an odd number.")
            elif op == "apply_median_blur":
                params['ksize'] = st.number_input("Kernel Size", min_value=1, max_value=max_ksize, value=min(5, max_ksize), step=2, key=f"mblur_{step_id}",
                    help="Size of the kernel. Must be an odd number.")
            elif op == "apply_canny":
                params['t1'] = st.number_input("Threshold 1", min_value=0, max_value=255, value=100, step=1, key=f"canny1_{step_id}",
                    help="The lower threshold for edge detection.")
                params['t2'] = st.number_input("Threshold 2", min_value=0, max_value=255, value=200, step=1, key=f"canny2_{step_id}",
                    help="The upper threshold for edge detection.")
            elif op == "apply_morphology":
                params['operation'] = st.selectbox("Operation", ["Erosion", "Dilation", "Opening", "Closing", "Gradient"], key=f"morph_op_{step_id}",
                    help="The morphological operation to apply.")
                params['ksize'] = st.number_input("Kernel Size", min_value=1, max_value=max_ksize, value=min(5, max_ksize), step=2, key=f"morph_ksize_{step_id}",
                    help="Size of the kernel. Must be an odd number.")
                params['shape'] = st.selectbox("Kernel Shape", ["Rect", "Ellipse", "Cross"], key=f"morph_shape_{step_id}",
                    help="The shape of the kernel.")
            elif op == "apply_dwt":
                params['wavelet'] = st.selectbox("Wavelet", ["haar", "db1", "db2", "sym2", "coif1"], key=f"dwt_{step_id}",
                    help="The wavelet to use for the discrete wavelet transform.")
            elif op == "add_gaussian_noise":
                params['mean'] = st.number_input("Mean", min_value=-100.0, max_value=100.0, value=0.0, step=0.1, key=f"noise_mean_{step_id}",
                    help="The mean of the Gaussian noise to add.")
                params['std'] = st.number_input("Standard Deviation", min_value=0.0, max_value=100.0, value=0.1, step=0.1, key=f"noise_std_{step_id}",
                    help="The standard deviation of the Gaussian noise to add.")
            elif op == "add_salt_pepper_noise":
                params['salt_prob'] = st.number_input("Salt Probability", min_value=0.0, max_value=1.0, value=0.01, step=0.01, key=f"sp_salt_{step_id}",
                    help="The probability of adding salt noise.")
                params['pepper_prob'] = st.number_input("Pepper Probability", min_value=0.0, max_value=1.0, value=0.01, step=0.01, key=f"sp_pepper_{step_id}",
                    help="The probability of adding pepper noise.")
            elif op == "add_jpeg_artifacts":
                params['quality'] = st.number_input("Quality", min_value=1, max_value=100, value=50, step=1, key=f"jpeg_{step_id}",
                    help="The quality of the JPEG image (1-100).")
            
            # --- FACE CONTROLS ---
            elif op == "align_face":
                params['id1'] = st.number_input("Landmark 1 ID", min_value=0, max_value=477, value=33, key=f"fa1_{step_id}",
                    help="Hover over the Main Output image on the right to find the Landmark ID (red dot) you want to target.")
                params['id2'] = st.number_input("Landmark 2 ID", min_value=0, max_value=477, value=263, key=f"fa2_{step_id}")
                params['t1_x'] = st.number_input("Target 1 X", value=int(p_w*0.3), key=f"fat1x_{step_id}", help="Where Landmark 1 should move horizontally.")
                params['t1_y'] = st.number_input("Target 1 Y", value=int(p_h*0.4), key=f"fat1y_{step_id}", help="Where Landmark 1 should move vertically.")
                params['t2_x'] = st.number_input("Target 2 X", value=int(p_w*0.7), key=f"fat2x_{step_id}", help="Where Landmark 2 should move horizontally.")
                params['t2_y'] = st.number_input("Target 2 Y", value=int(p_h*0.4), key=f"fat2y_{step_id}", help="Where Landmark 2 should move vertically.")
            elif op == "advanced_crop_face":
                params['bb_type'] = st.selectbox("Crop Shape", 
                    ["Minimum Rectangle", "Minimum Square", "Minimum Oval", "Polygonal"], 
                    key=f"ac_bb_{step_id}",
                    help="Defines the geometry of the mask used to isolate the face.")
                
                # Show string input ONLY if Polygonal is selected
                if params['bb_type'] == "Polygonal":
                    params['poly_string'] = st.text_input("Landmark IDs (Comma separated)", 
                        value="10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109", 
                        key=f"ac_poly_{step_id}",
                        help="Enter at least 3 Landmark IDs separated by commas. Hover over the main output mesh to find IDs. Example creates an octagon around the face.")
                else:
                    params['poly_string'] = "" # Pass empty if not used
                
                params['padding'] = st.number_input("Padding (Pixels)", min_value=0, max_value=500, value=0, step=5, key=f"ac_pad_{step_id}",
                    help="Puffs the shape outwards. 0 hugs the face exactly. Higher values include more of the background (hair, neck).")
                
                params['exterior_mode'] = st.selectbox("Exterior Action", ["Cut Out", "Set Exterior to 0"], key=f"ac_ext_{step_id}",
                    help="'Cut Out' physically slices the image file down to the masked size. 'Set to 0' leaves the image resolution unchanged but blacks out the background.")
                
            pipeline_config.append({'op': op, 'params': params})

    # Batch Processing Trigger
    st.divider()
    if st.button("🚀 Run Batch Process", type="primary", width='stretch'):
        if not uploaded_files:
            st.error("Please upload files first.")
        else:
            os.makedirs(output_folder, exist_ok=True)
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(uploaded_files):
                img = np.array(Image.open(file))
                
                # Apply pipeline
                for step in pipeline_config:
                    op, p = step['op'], step['params']
                    if not op:
                        continue
                    func = getattr(fn, op, None)
                    if func is not None:
                        img = func(img, **p)
                
                # Save
                save_path = os.path.join(output_folder, f"mod_{file.name}")
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape)==3 else img)
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            # Save Log TXT
            log_path = os.path.join(output_folder, "pipeline_log.txt")
            with open(log_path, "w") as f:
                f.write("APPLIED TRANSFORMATIONS:\n")
                for i, step in enumerate(pipeline_config):
                    f.write(f"{i+1}. {step['op']} - {step['params']}\n")
            
            st.success(f"Batch completed! Log saved to {log_path}")

# --- MAIN LAYOUT (Side-by-Side View) ---
if uploaded_files:
    # Use the first uploaded image for the live preview
    preview_file = uploaded_files[0]
    preview_bytes = preview_file.getvalue()
    pil_img = Image.open(io.BytesIO(preview_bytes))
    original_img = np.array(pil_img)
    
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
        st.image(original_img, width='stretch')
        
    with col2:
        st.subheader("Modified Preview (Interactive)")
        
        # Run landmark detection on the CURRENT state of the modified image
        current_landmarks = fn.get_face_landmarks(processed_img)
        
        # Render the interactive Plotly mesh
        fig = fn.create_main_preview(processed_img, current_landmarks, highlight_landmarks)
        st.plotly_chart(fig, width='stretch')

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


