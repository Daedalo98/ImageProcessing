import streamlit as st
import pandas as pd
import os
import cv2  # Added for ultra-fast frame extraction
import video_functions as vf
from moviepy import VideoFileClip

# --- UI Setup ---
st.set_page_config(page_title="Bulk Video Pipeline", layout="wide")
st.title("🎬 Bulk Video Pipeline: Clip & Extract")
st.write("Automatically clip videos based on a spreadsheet, then optionally extract all frames.")

# ==========================================
# 1. Input Handling
# ==========================================
st.header("1. Folders & Data")
col1, col2 = st.columns(2)
with col1:
    source_folder = st.text_input("Source Video Folder Path", placeholder="/path/to/source/videos")
with col2:
    output_folder = st.text_input("Output Folder Path", placeholder="/path/to/save/clips")

uploaded_file = st.file_uploader("Upload Spreadsheet (CSV, XLSX, XLS)", type=["csv", "xlsx", "xls"])

# Initialize df as None to handle UI state gracefully
df = None 

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")

# ==========================================
# 2. Dynamic Column Mapping
# ==========================================
st.header("2. Column Mapping")

if df is None:
    st.warning("⚠️ Please upload a spreadsheet in Section 1 to map columns.")
else:
    columns = df.columns.tolist()
    
    col_v, col_s, col_e = st.columns(3)
    with col_v:
        video_col = st.selectbox("Video Filename Column", options=columns, index=0)
    with col_s:
        start_col = st.selectbox("Start Time Column", options=columns, index=min(1, len(columns)-1))
    with col_e:
        end_col = st.selectbox("End Time / Duration Column", options=columns, index=min(2, len(columns)-1))
        
    col_l, col_t = st.columns(2)
    with col_l:
        label_cols = st.multiselect("Label Columns (Added to Output Filename)", options=columns)
    with col_t:
        time_mode = st.radio("Second Time Column is:", ["End Time", "Duration"], horizontal=True)

    with st.expander("Preview Uploaded Data"):
        st.dataframe(df.head(5))

# ==========================================
# 3. Processing Logic (Clipping)
# ==========================================
st.header("3. Process Video Clips")

if df is None:
    st.warning("⚠️ Waiting for spreadsheet upload to enable processing.")
else:
    if st.button("Clip Videos", type="primary"):
        # Folder validations
        if not source_folder or not os.path.isdir(source_folder):
            st.error("❌ Source folder does not exist or is invalid.")
            st.stop()
            
        if not output_folder:
            st.error("❌ Please provide an Output Folder path.")
            st.stop()

        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
                st.info(f"Created output folder: {output_folder}")
            except Exception as e:
                st.error(f"❌ Failed to create output folder: {e}")
                st.stop()

        # UI Elements for Tracking
        st.subheader("Processing Log")
        progress_bar = st.progress(0, text="Initializing...")
        log_area = st.empty()
        logs = []

        def add_log(msg):
            """Helper to append messages to the live log."""
            logs.append(msg)
            # Display the last 15 lines so it doesn't get overwhelmingly tall
            log_area.code("\n".join(logs[-15:]), language="text")

        results = []
        total_rows = len(df)

        add_log(f"Starting job: {total_rows} rows to process.")

        for index, row in df.iterrows():
            target_filename = str(row[video_col])
            add_log(f"--- Row {index+1}: Target -> {target_filename} ---")
            
            # Times
            start_val, end_val = row[start_col], row[end_col]
            start_sec = vf.parse_time(start_val)
            end_or_dur_sec = vf.parse_time(end_val)

            if start_sec is None or end_or_dur_sec is None:
                add_log(f"[ERROR] Invalid time format for {target_filename}")
                results.append({"Row": index+1, "File": target_filename, "Status": "Error", "Details": "Invalid time format."})
                continue

            end_sec = (start_sec + end_or_dur_sec) if time_mode == "Duration" else end_or_dur_sec

            if start_sec >= end_sec:
                add_log(f"[ERROR] Start time ({start_sec}s) >= End time ({end_sec}s)")
                results.append({"Row": index+1, "File": target_filename, "Status": "Error", "Details": "Start time >= End time."})
                continue

            # File matching
            actual_file = vf.find_video_file(source_folder, target_filename)
            if not actual_file:
                add_log(f"[ERROR] Could not find file matching '{target_filename}' in source folder.")
                results.append({"Row": index+1, "File": target_filename, "Status": "Error", "Details": "File not found."})
                continue

            source_path = os.path.join(source_folder, actual_file)
            base_name = os.path.splitext(actual_file)[0]
            labels = "_".join([vf.sanitize_filename(row[c]) for c in label_cols if pd.notna(row[c])]) or "clip"
            out_name = f"{base_name}_{labels}_{start_sec:.1f}_{end_sec:.1f}.mp4"
            out_path = os.path.join(output_folder, out_name)

            # Clipping
            add_log(f"[INFO] Clipping '{actual_file}' from {start_sec}s to {end_sec}s...")
            try:
                with VideoFileClip(source_path) as video:
                    if start_sec > video.duration:
                        msg = f"Start time ({start_sec}s) exceeds duration ({video.duration}s)."
                        add_log(f"[ERROR] {msg}")
                        results.append({"Row": index+1, "File": actual_file, "Status": "Error", "Details": msg})
                        continue
                    
                    actual_end = min(end_sec, video.duration)
                    clip = video.subclip(start_sec, actual_end)
                    clip.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
                    
                add_log(f"[SUCCESS] Saved to {out_name}")
                results.append({"Row": index+1, "File": actual_file, "Status": "Success", "Details": f"Clipped to {out_name}"})
            except Exception as e:
                add_log(f"[ERROR] Exception during clipping: {str(e)}")
                results.append({"Row": index+1, "File": actual_file, "Status": "Error", "Details": str(e)})
            
            # Update overall progress
            progress_bar.progress((index + 1) / total_rows, text=f"Processed {index+1} of {total_rows}")

        add_log("Job Complete!")
        progress_bar.progress(1.0, text="Job Complete!")
        
        # Summary Table
        st.subheader("Process Summary")
        results_df = pd.DataFrame(results)
        st.dataframe(
            results_df.style.map(lambda v: 'color: #ff4b4b; font-weight: bold;' if v == 'Error' else 'color: #00cc66; font-weight: bold;', subset=['Status']), 
            use_container_width=True
        )

st.markdown("---")

# ==========================================
# 4. Frame Extraction (OpenCV)
# ==========================================
st.header("4. Bulk Frame Extraction")
st.write("Extract every frame from all videos in a specified directory. Folders will be created for each video automatically.")

# Default to the output folder of the clipping stage if it was provided
frames_source = st.text_input("Folder containing videos to extract frames from:", value=output_folder if output_folder else "")

if not frames_source:
    st.info("ℹ️ Provide a folder path above to enable frame extraction.")
else:
    if st.button("Extract Frames", type="secondary"):
        if not os.path.isdir(frames_source):
            st.error("❌ The provided folder does not exist.")
        else:
            # Gather valid video files
            valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
            videos_to_extract = [f for f in os.listdir(frames_source) if f.lower().endswith(valid_extensions)]
            
            if not videos_to_extract:
                st.warning(f"No videos found in `{frames_source}` ending in {valid_extensions}")
            else:
                total_vids = len(videos_to_extract)
                overall_vid_prog = st.progress(0, text=f"Extracting videos: 0/{total_vids}")
                
                # Dedicated area for the active video's frame extraction progress
                active_frame_prog = st.empty() 
                frame_log = st.empty()
                f_logs = []

                for i, vid_file in enumerate(videos_to_extract):
                    vid_path = os.path.join(frames_source, vid_file)
                    vid_name = os.path.splitext(vid_file)[0]
                    
                    # Create subfolder for this video's frames
                    out_dir = os.path.join(frames_source, f"{vid_name}_frames")
                    os.makedirs(out_dir, exist_ok=True)
                    
                    f_logs.append(f"Opening {vid_file}...")
                    frame_log.code("\n".join(f_logs[-5:]), language="text")

                    # OpenCV Frame Extraction
                    cap = cv2.VideoCapture(vid_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_count = 0
                    
                    # Update active progress bar
                    prog_bar = active_frame_prog.progress(0.0, text=f"Extracting frames for {vid_file}... (0/{total_frames})")
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break # End of video
                        
                        # Save frame (e.g., frame_000001.jpg)
                        frame_filename = os.path.join(out_dir, f"frame_{frame_count:06d}.jpg")
                        cv2.imwrite(frame_filename, frame)
                        frame_count += 1
                        
                        # Update progress bar every 30 frames to save Streamlit rendering overhead
                        if frame_count % 30 == 0 and total_frames > 0:
                            prog_bar.progress(min(frame_count / total_frames, 1.0), text=f"Extracting frames for {vid_file}... ({frame_count}/{total_frames})")

                    cap.release()
                    f_logs.append(f"✅ Extracted {frame_count} frames to {out_dir}")
                    frame_log.code("\n".join(f_logs[-5:]), language="text")
                    
                    # Clear the individual frame progress bar and update the overall video one
                    active_frame_prog.empty()
                    overall_vid_prog.progress((i + 1) / total_vids, text=f"Extracting videos: {i+1}/{total_vids}")

                st.success("🎉 All frame extractions completed successfully!")