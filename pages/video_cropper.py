import streamlit as st
import cv2
import tempfile
from pathlib import Path
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
from moviepy import VideoFileClip
import shutil
from streamlit_cropper import st_cropper
from PIL import Image
import json

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Static Face Cropper")

# --- UI: SIDEBAR ---
st.sidebar.header("Directories")

# 1. Ask for Input, Frames, and Output directories
input_videos_dir = st.sidebar.text_input("Input Videos Main Directory", help="Path containing all subfolders with videos.")
frames_base_dir = st.sidebar.text_input("Frames Base Directory Path", help="Path containing all subfolders with frames.")
output_base_dir = st.sidebar.text_input("Output Directory", help="Where to save the cropped videos.")

# --- NEW ADDITION ---
maintain_structure = st.sidebar.checkbox(
    "Replicate subfolder structure", 
    value=True, 
    help="If checked, preserves your subfolders. If unchecked, dumps all videos directly into the root Output Directory."
)

st.sidebar.header("Crop Parameters")
bb_type = st.sidebar.selectbox("Bounding Box Type", ["Rectangle", "Square"])

# 2. Change padding to a positive/negative pixel offset
padding_px = st.sidebar.number_input("Padding (Pixels)", min_value=-500, max_value=500, value=50, step=10,
                                     help="Positive values enlarge the box, negative values shrink it.")
target_size = st.sidebar.number_input("Output Resolution (px)", min_value=256, max_value=2160, value=512)

uploaded_video = st.sidebar.file_uploader("Upload Source Video", type=["mp4", "mov", "avi"])

# --- CORE LOGIC ---

def process_frames(frames_base_dir: str, real_video_basename: str, logger):
    """
    Hunts down the frames folder and extracts all bounding boxes.
    Returns: (actual_detections_dict, path_to_sample_image) or (None, None) if failed.
    """
    logger.write(f"🔎 Searching for subfolder containing: `{real_video_basename}`...")
    
    target_frames_dir = None
    if frames_base_dir and os.path.exists(frames_base_dir):
        base_path = Path(frames_base_dir)
        for path in base_path.rglob(f"*{real_video_basename}*"):
            if path.is_dir():
                target_frames_dir = path
                break

    if not target_frames_dir:
        logger.error(f"❌ Could not find a frames folder matching '{real_video_basename}' in '{frames_base_dir}'.")
        return None, None
        
    logger.write(f"✅ Found frames folder: `{target_frames_dir}`")

    # --- MEDIAPIPE INITIALIZATION ---
    model_path = 'blaze_face_short_range.tflite'
    if not os.path.exists(model_path):
        logger.write("📥 Downloading MediaPipe face detection model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        urllib.request.urlretrieve(url, model_path)
        logger.write("✅ Model downloaded.")

    logger.write("⚙️ Initializing MediaPipe Vision AI...")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
    detector = vision.FaceDetector.create_from_options(options)

    actual_detections = {}
    img_index = 0

    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = sorted([f for f in target_frames_dir.iterdir() if f.suffix.lower() in valid_extensions])

    if not image_files:
        logger.error("❌ The frames folder was found, but it contains no images.")
        return None, None

    logger.write(f"🖼️ Found {len(image_files)} images. Extracting faces...")
    
    # Process the images
    faces_found = 0
    for img_path in image_files:
        mp_image = mp.Image.create_from_file(str(img_path))
        detection_result = detector.detect(mp_image)
        
        if detection_result.detections:
            faces_found += 1
            bbox = detection_result.detections[0].bounding_box
            
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            actual_detections[img_index] = [x, y, w, h]
            img_index += 1

    detector.close()
    
    if faces_found == 0:
        logger.error("❌ MediaPipe processed the images but could not detect any faces.")
        return None, None
        
    logger.write(f"✅ Face tracking complete. Found faces in {faces_found}/{len(image_files)} frames.")

    # Return the dictionary and the first image path so we can draw a preview later
    return actual_detections, image_files[0]

def calculate_global_bb(detections, width, height, bb_type, padding_px):
    """
    Calculates the final fixed bounding box using additive padding.
    """
    all_x1 = [box[0] for box in detections.values()]
    all_y1 = [box[1] for box in detections.values()]
    all_x2 = [box[0] + box[2] for box in detections.values()]
    all_y2 = [box[1] + box[3] for box in detections.values()]

    min_x, min_y = min(all_x1), min(all_y1)
    max_x, max_y = max(all_x2), max(all_y2)

    w = max_x - min_x
    h = max_y - min_y
    center_x = min_x + (w / 2)
    center_y = min_y + (h / 2)

    # Apply Additive Padding (adds/subtracts pixels to all 4 sides)
    new_w = w + (padding_px * 2)
    new_h = h + (padding_px * 2)

    if bb_type == "Square":
        side = max(new_w, new_h)
        new_w, new_h = side, side

    # Clamp to frame boundaries
    final_x1 = max(0, int(center_x - new_w / 2))
    final_y1 = max(0, int(center_y - new_h / 2))
    final_x2 = min(width, int(center_x + new_w / 2))
    final_y2 = min(height, int(center_y + new_h / 2))

    return final_x1, final_y1, final_x2, final_y2

def process_video(input_path, output_path, real_video_basename, frames_base_dir, bb_type, padding_px, target_size, logger, manual_coords=None):
    """
    Handles the calculation of the global bounding box (or uses manual override) 
    and the actual video rendering.
    """
    
    # --- EXTRACT VIDEO DIMENSIONS (Needed for all paths) ---
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- PATH A: MANUAL OVERRIDE ---
    if manual_coords is not None:
        logger.write("### Phase 1, 2 & 3 Skipped: Using Manual Override")
        
        # Unpack the coordinates provided by the streamlit-cropper
        raw_x1, raw_y1, raw_x2, raw_y2 = manual_coords
        
        # Failsafe: Clamp coordinates to frame limits just in case the crop 
        # widget returned floating point numbers or slightly exceeded bounds.
        final_x1 = max(0, int(raw_x1))
        final_y1 = max(0, int(raw_y1))
        final_x2 = min(width, int(raw_x2))
        final_y2 = min(height, int(raw_y2))
        
        logger.write(f"📐 Using Custom Coordinates: `({final_x1}, {final_y1})` to `({final_x2}, {final_y2})`")

    # --- PATH B: AI DETECTION ---
    else:
        # --- STEP 1: PROCESS FRAMES ---
        logger.write("### Phase 1: Analyzing Frames")
        detections, sample_img_path = process_frames(frames_base_dir, real_video_basename, logger)
        
        if not detections:
            logger.error("🛑 Pipeline aborted due to frame processing failure.")
            cap.release()
            return False, "Phase 1: No faces detected in frames."

        # --- STEP 2: CALCULATE GLOBAL BOX ---
        logger.write("### Phase 2: Calculating Global Bounding Box")
        final_x1, final_y1, final_x2, final_y2 = calculate_global_bb(detections, width, height, bb_type, padding_px)
        logger.write(f"📐 Final Global Coordinates clamped to frame limits: `({final_x1}, {final_y1})` to `({final_x2}, {final_y2})`")

        # --- STEP 3: SHOW PREVIEW ---
        logger.write("📸 Generating bounding box preview on sample frame...")
        sample_img = cv2.imread(str(sample_img_path))
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        cv2.rectangle(sample_img, (final_x1, final_y1), (final_x2, final_y2), (0, 255, 0), 4)
        logger.image(sample_img, caption=f"Global {bb_type} applied to sample frame.", use_container_width=True)


    # --- STEP 4: CROPPING ---
    logger.write("### Phase 4: Video Cropping")
    logger.write("⏳ Slicing video and applying resizing...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_size, target_size))

    progress_bar = st.progress(0)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[final_y1:final_y2, final_x1:final_x2]
        
        if cropped_frame.size > 0:
            final_frame = cv2.resize(cropped_frame, (target_size, target_size))
            out.write(final_frame)

        frame_idx += 1
        if frame_idx % 15 == 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    out.release()
    progress_bar.empty()
    
    # --- STEP 5: BROWSER CODEC CONVERSION ---
    logger.write("🔄 Converting video codec for browser compatibility (H.264)...")
    try:
        from moviepy import VideoFileClip
        import shutil
        clip = VideoFileClip(output_path)
        h264_path = output_path.replace(".mp4", "_h264.mp4")
        clip.write_videofile(h264_path, codec="libx264", audio=False, logger=None)
        clip.close()
        shutil.move(h264_path, output_path)
        logger.write("✅ Codec conversion successful!")
    except Exception as e:
        logger.error(f"❌ Codec conversion failed: {e}")
        return False, f"Codec conversion failed: {e}"

    logger.write("✅ Video rendering complete!")
    return True, "Success"

# --- UI: MAIN PAGE ---
st.title("Batch Face Cropper")

if input_videos_dir and os.path.exists(input_videos_dir):
    # Find all videos in the main folder and subfolders
    valid_vid_exts = ('.mp4', '.mov', '.avi')
    all_videos = [p for p in Path(input_videos_dir).rglob("*") if p.suffix.lower() in valid_vid_exts]
    
    st.write(f"📁 Found **{len(all_videos)}** videos in input directory.")

    if len(all_videos) > 0:
        # --- FEATURE 1: PREVIEW TOOL ---
        st.subheader("1. Tune Parameters & Interactive Bounding Box")
        test_video = st.selectbox("Select a video to test or manually crop:", all_videos, format_func=lambda x: x.name)
        
        # We use session state to remember if the preview has been generated
        if st.button("🖼️ Analyze Video & Open Drawing Tool"):
            st.session_state['preview_active'] = True
            st.session_state['test_video'] = test_video

        if st.session_state.get('preview_active'):
            with st.spinner("Extracting frames and calculating AI Bounding Box..."):
                import sys
                class DummyLogger:
                    def write(self, msg): pass
                    def error(self, msg): st.error(msg)
                
                vid_basename = st.session_state['test_video'].stem
                
                # 1. AI Processing
                detections, sample_img_path = process_frames(frames_base_dir, vid_basename, DummyLogger())
                
                if detections:
                    cap = cv2.VideoCapture(str(st.session_state['test_video']))
                    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    
                    fx1, fy1, fx2, fy2 = calculate_global_bb(detections, vw, vh, bb_type, padding_px)
                    
                    # 2. Prepare the Image
                    sample_img = cv2.imread(str(sample_img_path))
                    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
                    
                    # Draw the AI's suggested box in BLUE
                    cv2.rectangle(sample_img, (fx1, fy1), (fx2, fy2), (0, 0, 255), 4) 
                    
                    st.info("🟦 **Blue Box:** AI's suggestion based on padding. \n\n🟩 **Green Box:** Your interactive selection. Drag the corners to draw your own bounding box.")
                    
                    # Convert to PIL Image for the interactive cropper
                    pil_img = Image.fromarray(sample_img)
                    
                    # 3. THE INTERACTIVE CROPPER
                    # If bb_type is Square, this strictly forces the user to draw a 1:1 box!
                    aspect_ratio = (1, 1) if bb_type == "Square" else None
                    
                    box = st_cropper(
                        pil_img, 
                        realtime_update=True, 
                        box_color='#00FF00', 
                        aspect_ratio=aspect_ratio, 
                        return_type='box'
                    )
                    
                    # 4. Extract User's Manual Coordinates
                    manual_x1 = box['left']
                    manual_y1 = box['top']
                    manual_x2 = box['left'] + box['width']
                    manual_y2 = box['top'] + box['height']
                    
                    st.write(f"**Your Custom Coordinates:** `({manual_x1}, {manual_y1})` to `({manual_x2}, {manual_y2})`")
                    
                    # 5. Save manual coordinates to session state so the Batch Processor can use them
                    st.session_state['manual_bb'] = (manual_x1, manual_y1, manual_x2, manual_y2)

        st.divider()

        # --- FEATURE 2: BATCH PROCESSING ---
        st.subheader("2. Run Pipeline")
        
        use_manual_bb = st.checkbox("Override AI: Use the Green Box I drew above for all videos", value=False)
        
        if st.button("🚀 Batch Process All Videos"):
            if not output_base_dir:
                st.error("Please provide a valid Output Directory in the sidebar.")
            elif use_manual_bb and 'manual_bb' not in st.session_state:
                st.error("You must draw a preview box first before you can use the manual override!")
            else:
                os.makedirs(output_base_dir, exist_ok=True)
                manual_coords = st.session_state['manual_bb'] if use_manual_bb else None
                
                # Dictionary to track failed videos
                problematic_videos = {}

                for idx, vid_path in enumerate(all_videos):
                    vid_basename = vid_path.stem
                    
                    if maintain_structure:
                        relative_path = vid_path.parent.relative_to(Path(input_videos_dir))
                        target_dir = Path(output_base_dir) / relative_path
                        target_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        target_dir = Path(output_base_dir)
                        
                    output_file = target_dir / f"cropped_{vid_path.name}"
                    
                    st.write(f"### Processing {idx+1}/{len(all_videos)}: `{vid_basename}`")
                    with st.status(f"Running pipeline for {vid_path.name}...", expanded=True) as status_logger:
                        
                        # Catch the success boolean AND the message
                        success, msg = process_video(
                            str(vid_path), str(output_file), vid_basename, 
                            frames_base_dir, bb_type, padding_px, target_size, 
                            logger=status_logger, manual_coords=manual_coords
                        )
                        
                        if success:
                            status_logger.update(label=f"✅ {vid_path.name} Complete", state="complete", expanded=False)
                        else:
                            status_logger.update(label=f"❌ {vid_path.name} Failed", state="error", expanded=False)
                            # Log it to our dictionary
                            problematic_videos[str(vid_path)] = msg
                
                # --- SAVE THE ERROR REPORT ---
                if problematic_videos:
                    report_path = Path(output_base_dir) / "failed_videos_report.json"
                    with open(report_path, "w") as f:
                        json.dump(problematic_videos, f, indent=4)
                    st.warning(f"⚠️ {len(problematic_videos)} videos failed (AI couldn't find faces). See the Recovery Tool below.")
                else:
                    st.success(f"🎉 All videos processed successfully!")

        # --- FEATURE 3: PROBLEMATIC VIDEOS RECOVERY ---
        if output_base_dir and os.path.exists(output_base_dir):
            report_path = Path(output_base_dir) / "failed_videos_report.json"
            
            if report_path.exists():
                with open(report_path, "r") as f:
                    problematic_videos = json.load(f)

                if problematic_videos:
                    st.divider()
                    st.subheader("3. Recovery: Handle Problematic Videos")
                    st.error(f"Found {len(problematic_videos)} failed videos.")

                    with st.expander("View Error Logs"):
                        st.json(problematic_videos)

                    recovery_mode = st.radio("How would you like to process these?", [
                        "Ad-hoc: Draw a specific box for each video",
                        "Global: Draw ONE box to apply to ALL failed videos"
                    ])

                    failed_paths = list(problematic_videos.keys())
                    ref_vid = st.selectbox("Select a video to view and draw the box:", failed_paths, format_func=lambda x: Path(x).name)

                    # Forcibly extract frame 0 from the video to draw on
                    cap = cv2.VideoCapture(ref_vid)
                    ret, frame = cap.read()
                    cap.release()

                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        aspect_ratio = (1, 1) if bb_type == "Square" else None

                        box = st_cropper(pil_img, realtime_update=True, box_color='#00FF00', aspect_ratio=aspect_ratio)
                        m_x1, m_y1 = box['left'], box['top']
                        m_x2, m_y2 = box['left'] + box['width'], box['top'] + box['height']
                        manual_coords = (m_x1, m_y1, m_x2, m_y2)

                        st.write(f"**Box Coordinates:** `({m_x1}, {m_y1})` to `({m_x2}, {m_y2})`")

                        # AD-HOC PROCESSING
                        if recovery_mode == "Ad-hoc: Draw a specific box for each video":
                            if st.button(f"🔧 Process ONLY `{Path(ref_vid).name}`"):
                                vid_path = Path(ref_vid)
                                
                                if maintain_structure:
                                    relative_path = vid_path.parent.relative_to(Path(input_videos_dir))
                                    target_dir = Path(output_base_dir) / relative_path
                                    target_dir.mkdir(parents=True, exist_ok=True)
                                else:
                                    target_dir = Path(output_base_dir)
                                output_file = target_dir / f"cropped_{vid_path.name}"

                                class SilentLogger:
                                    def write(self, msg): pass
                                    def error(self, msg): st.error(msg)

                                with st.spinner(f"Cropping {vid_path.name}..."):
                                    success, msg = process_video(
                                        str(vid_path), str(output_file), vid_path.stem,
                                        frames_base_dir, bb_type, padding_px, target_size,
                                        logger=SilentLogger(), manual_coords=manual_coords
                                    )

                                if success:
                                    st.success("Fixed!")
                                    del problematic_videos[ref_vid]
                                    with open(report_path, "w") as f:
                                        json.dump(problematic_videos, f, indent=4)
                                    st.rerun() # Refreshes the UI instantly

                        # GLOBAL PROCESSING
                        else:
                            if st.button(f"🚀 Process ALL {len(problematic_videos)} videos with this exact box"):
                                for v_path_str in list(problematic_videos.keys()):
                                    vid_path = Path(v_path_str)
                                    
                                    if maintain_structure:
                                        relative_path = vid_path.parent.relative_to(Path(input_videos_dir))
                                        target_dir = Path(output_base_dir) / relative_path
                                        target_dir.mkdir(parents=True, exist_ok=True)
                                    else:
                                        target_dir = Path(output_base_dir)
                                    output_file = target_dir / f"cropped_{vid_path.name}"

                                    class SilentLogger:
                                        def write(self, msg): pass
                                        def error(self, msg): pass

                                    success, msg = process_video(
                                        str(vid_path), str(output_file), vid_path.stem,
                                        frames_base_dir, bb_type, padding_px, target_size,
                                        logger=SilentLogger(), manual_coords=manual_coords
                                    )

                                    if success:
                                        del problematic_videos[v_path_str]

                                with open(report_path, "w") as f:
                                    json.dump(problematic_videos, f, indent=4)
                                st.rerun()
                    else:
                        st.error("Could not read frame from video. File might be corrupted.")

    st.info("Please enter a valid Input Videos Directory in the sidebar.")