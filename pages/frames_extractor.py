import streamlit as st
import os
import time
import datetime
import subprocess
import imageio_ffmpeg
from pathlib import Path
import cv2

# ==========================================
# SYSTEM SETUP
# ==========================================
# We use imageio_ffmpeg to ensure we have a reliable, local FFmpeg binary 
# without relying on the user's system environment variables.
FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()

# Define the types of video files our script will look for.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# --- UI Setup ---
st.set_page_config(page_title="Max-Speed Frame Extractor", layout="wide")
st.title("⚡ Max-Speed Sequential Frame Extractor")
st.write("Extracts every frame using automated hardware acceleration and maximum internal threading.")

# ==========================================
# 1. DIRECTORY CONFIGURATION
# ==========================================
st.header("1. Output Mode & Folders")

out_mode = st.radio(
    "How should the extracted frames be saved?",
    options=[
        "A. Save in the original folder (Creates a 'VideoName_frames' subfolder next to each video)",
        "B. Save to a separate Output Folder (Replicates original subfolder structure)"
    ],
    index=0
)

col1, col2 = st.columns(2)
with col1:
    source_folder = st.text_input("Source Video Folder Path", placeholder="./source_videos")

with col2:
    if out_mode.startswith("A"):
        output_folder = None
        st.info("ℹ️ Output path not required for Mode A.")
    else:
        output_folder = st.text_input("Output Folder Path (Required)", placeholder="./extracted_frames")

# ==========================================
# 1.5 EXTRACTION SETTINGS
# ==========================================
st.markdown("---") # Visual divider
st.header("1.5 Extraction Settings")

# Add a number input to control the exact frames per second extracted.
# We default to 3 as requested. This prevents the SSD bottleneck from happening
# as quickly, because you are generating vastly less data per video.
target_fps = st.number_input(
    "Frames Per Second (FPS) to extract", 
    min_value=1, 
    max_value=30, 
    value=3, 
    help="Determines how many frames are saved per second of video. Lower numbers process much faster and save disk space."
)

# ==========================================
# 2. PROCESSING PIPELINE
# ==========================================
st.header("2. Run Extraction")

if st.button("🚀 Start Extraction Process", type="primary"):
    
    # --- Step A: Validate Folders ---
    if not source_folder or not os.path.isdir(source_folder):
        st.error("❌ Source folder does not exist or is invalid.")
        st.stop()
        
    if out_mode.startswith("B") and not output_folder:
        st.error("❌ Mode B requires a valid Output Folder path.")
        st.stop()

    source_path = Path(source_folder)
    
    # --- Step B: Scan for Videos ---
    videos_to_process = []
    for root, dirs, files in os.walk(source_path, onerror=lambda e: print(f"Skipping: {e}")):
        for filename in files:
            file_path = Path(root) / filename
            if file_path.suffix.lower() in VIDEO_EXTENSIONS:
                videos_to_process.append(file_path)

    total_videos = len(videos_to_process)
    
    if total_videos == 0:
        st.warning(f"⚠️ No supported video files found in {source_folder}.")
        st.stop()
        
    st.success(f"✅ Found {total_videos} videos. Starting high-speed sequential extraction...")

    # --- Step C: UI Elements for Tracking Progress ---
    progress_bar = st.progress(0, text="Initializing Pipeline...")
    log_area = st.empty()
    logs = []

    # Helper function to print logs to the UI without crashing Streamlit
    def add_log(msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        logs.append(f"[{timestamp}] {msg}")
        # Keep only the last 20 logs so the browser doesn't run out of memory on massive jobs
        log_area.code("\n".join(logs[-20:]), language="text")

    # --- Step D: The Extraction Loop ---
    start_time = time.time()
    videos_completed = 0
    results = []

    add_log("Starting auto-accelerated processing loop...")

    # We iterate sequentially. No threading limits SSD bottlenecking, 
    # while FFmpeg handles internal threading for max speed.
    for vid_path in videos_to_process:
        base_name = vid_path.stem
        
        # 1. Determine exactly where to save the files based on the user's choice
        if out_mode.startswith("A"):
            # Put the frames folder right next to the original video file
            out_dir = vid_path.parent / f"{base_name}_frames"
        else:
            # Replicate the subfolder math in the new destination directory
            relative_path = vid_path.relative_to(source_path)
            out_dir = Path(output_folder) / relative_path.parent / f"{base_name}_frames"
            
        # Create the directory if it doesn't exist yet
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Output template configuration
        # %06d ensures files are named _000001.jpg, _000002.jpg. 
        # This prevents numbering glitches on videos with more than 9,999 frames.
        out_template = out_dir / f"{base_name}_%06d.jpg"
        
        # 3. Crash Recovery / Resume Logic
        # If the first frame already exists, we assume the job finished previously.
        check_path = out_dir / f"{base_name}_000001.jpg"
        if check_path.exists():
            add_log(f"⏭️ '{base_name}' already extracted. Skipping.")
            results.append({"vid": base_name, "status": "Skipped"})
        else:
            add_log(f"🎬 Processing '{base_name}'...")
            
            # --- NEW ADDITION: Calculate the Frame Step Size ---
            # 1. Use OpenCV to instantly read the video's original FPS
            cap = cv2.VideoCapture(str(vid_path))
            vid_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Fallback just in case OpenCV fails to read the metadata
            if vid_fps <= 0:
                st.warning(f"⚠️ Could not read FPS for '{base_name}'. Defaulting to 30 FPS for step calculation.")
                vid_fps = 30.0 
                
            # 2. Calculate the step. 
            # Example: 30 original FPS / 3 target FPS = 10. (Keep 1 every 10 frames)
            # max(1, ...) ensures we never get a step size of 0.
            step_size = max(1, round(vid_fps / target_fps))
            
            # ---------------------------------------------------------
            # 🚀 THE OPTIMIZED FFMPEG COMMAND (WITH ORIGINAL FRAME NAMING)
            # ---------------------------------------------------------
            cmd = [
                FFMPEG_BINARY, 
                "-y",               
                "-hwaccel", "auto", 
                "-i", str(vid_path),
                "-threads", "0",    
                
                # --- NEW ADDITION: The Math & Filtering ---
                # 1. 'setpts=N': Normally, timestamps are measured in milliseconds. 
                #    This command rewrites the timestamp of every frame to be its literal integer frame number (0, 1, 2, 3...).
                # 2. 'select=not(mod(n\,step_size))': This checks the input frame number (n).
                #    If n is a multiple of our step size (e.g., 0, 10, 20, 30), it keeps the frame. Otherwise, it drops it.
                "-vf", f"setpts=N,select='not(mod(n\\,{step_size}))'", 
                
                # '-vsync 0' (or passthrough) ensures FFmpeg doesn't try to duplicate frames to fix framerates after we just dropped them.
                "-vsync", "0",      
                
                # '-frame_pts true' tells the JPEG generator: "Do not count 1, 2, 3. Instead, look at the 
                # timestamp of the frame (which we just forced to be the original frame index) and use that for %06d."
                "-frame_pts", "true", 
                
                "-q:v", "2",        
                str(out_template)   
            ]

            try:
                # Run the command and wait for it to finish. capture_output hides the messy terminal text.
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                # If FFmpeg crashes, it returns a non-zero code. We catch it and log the error.
                if process.returncode != 0:
                    # We only grab the last 500 characters of the error so we don't flood the UI.
                    raise Exception(f"FFmpeg error: {process.stderr[-500:]}") 
                    
                add_log(f"✅ Successfully extracted '{base_name}'.")
                results.append({"vid": base_name, "status": "Success"})
                
            except Exception as e:
                add_log(f"❌ Error on '{base_name}': {str(e)}")
                results.append({"vid": base_name, "status": "Error"})
        
        # --- Step E: Progress Update Math ---
        videos_completed += 1
        elapsed = time.time() - start_time
        time_per_vid = elapsed / videos_completed
        eta_secs = (total_videos - videos_completed) * time_per_vid
        
        # Format the ETA to a readable HH:MM:SS format
        eta_str = str(datetime.timedelta(seconds=int(eta_secs)))
        prog_text = f"Processed {videos_completed} of {total_videos} videos | ETA: {eta_str}"
        
        # Update the UI progress bar
        progress_bar.progress(videos_completed / total_videos, text=prog_text)

    # --- Step F: Completion Summary ---
    add_log("\n🎉 All Jobs Complete!")
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    progress_bar.progress(1.0, text=f"Job Complete! Total time: {total_time_str}")
    
    # Calculate the final results for the UI metrics
    success_count = sum(1 for r in results if r["status"] == "Success")
    skip_count = sum(1 for r in results if r["status"] == "Skipped")
    error_count = sum(1 for r in results if r["status"] == "Error")
    
    st.subheader("Process Summary")
    colA, colB, colC = st.columns(3)
    colA.metric("Successful Extractions", success_count)
    colB.metric("Skipped (Already Existed)", skip_count)
    colC.metric("Errors", error_count)