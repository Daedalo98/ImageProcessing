import streamlit as st
import pandas as pd
import os
import cv2
from pathlib import Path
import video_functions as fn
import datetime
import time
import imageio_ffmpeg
import subprocess

# Get the exact path to the hidden ffmpeg executable
FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()

# --- UI Setup ---
st.set_page_config(page_title="Bulk Video Pipeline", layout="wide")
st.title("🎬 Bulk Video Pipeline: Clip & Extract")
st.write("Automatically clip videos based on a spreadsheet, grouped by video for maximum speed.")

# ==========================================
# 1. Input Handling & Output Modes
# ==========================================
st.header("1. Folders, Modes & Data")

out_mode = st.radio(
    "How should the cut videos be saved?",
    options=[
        "A. Auto-create folder next to original video (No output path needed)",
        "B. Save to specific Output Folder (Grouped by video name)",
        "C. Save to specific Output Folder (Replicate original subfolder tree)"
    ],
    index=0
)

col1, col2 = st.columns(2)
with col1:
    source_folder = st.text_input("Source Video Folder Path", placeholder="/path/to/source/videos")

with col2:
    if out_mode.startswith("A"):
        output_folder = None
        st.info("ℹ️ Output path not required for Mode A.")
    else:
        output_folder = st.text_input("Output Folder Path (Required)", placeholder="/path/to/save/clips")

uploaded_file = st.file_uploader("Upload Spreadsheet (CSV, XLSX, XLS)", type=["csv", "xlsx", "xls"])

# Ask the user where the header is (defaults to 1)
header_row = st.number_input("Which row contains the Column Headers?", min_value=1, value=1, help="Usually 1. Increase this if your Excel file has titles or blank rows at the top.")

sheets = None
columns = []

if uploaded_file:
    try:
        # Pass the header_row into our load_spreadsheet function
        sheets, columns = fn.load_spreadsheet(uploaded_file.getvalue(), uploaded_file.name, header_row)
        if not columns:
            st.error("❌ The sheets in this Excel file do not share any common columns!")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {e}")

# ==========================================
# 2. Dynamic Column Mapping
# ==========================================
st.header("2. Column Mapping & Data Compilation")

if not columns:
    st.warning("⚠️ Please upload a spreadsheet in Section 1 to map columns.")
else:
    # 1. Restore the UI widgets so the user can map the columns
    col_v, col_s, col_e = st.columns(3)
    with col_v:
        video_col = st.selectbox("Video Filename Column", options=columns, index=0)
    with col_s:
        start_col = st.selectbox("Start Time Column", options=columns, index=min(1, len(columns)-1))
    with col_e:
        end_col = st.selectbox("End Time / Duration Column", options=columns, index=min(2, len(columns)-1))
        
    col_ml, col_sl, col_t = st.columns(3)
    with col_ml:
        main_label_col = st.selectbox("Main Label Column", options=["None"] + columns)
    with col_sl:
        sec_label_cols = st.multiselect("Secondary Labels/Notes", options=columns)
    with col_t:
        # We need to save time_mode to session_state so fn.parse_time can access it later if needed, 
        # or pass it explicitly. For simplicity, we'll store it as a normal variable.
        time_mode = st.radio("Second Time Column is:", ["End Time", "Duration"], horizontal=True)

    # 2. The Dictionary compilation step
    if st.button("Compile Data Plan", type="secondary"):
        with st.spinner("Compiling dictionary across all sheets..."):
            st.session_state.video_dict = fn.build_video_dict(
                sheets, video_col, start_col, end_col, main_label_col, sec_label_cols, header_row
            )
            # Save time_mode so we can use it in Section 3
            st.session_state.time_mode = time_mode 
        
    # 3. Show metrics if compiled
    if "video_dict" in st.session_state:
        v_dict = st.session_state.video_dict
        total_videos = len(v_dict)
        total_cuts = sum(len(cuts) for cuts in v_dict.values())
        
        st.success("✅ Data compiled successfully! You can now proceed to Step 3.")
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Unique Videos Found", total_videos)
        col_m2.metric("Total Cuts Scheduled", total_cuts)
        
        with st.expander("Preview Dictionary Structure (First 3 Videos)"):
            preview_dict = {k: v_dict[k] for k in list(v_dict.keys())[:3]}
            st.json(preview_dict)

# ==========================================
# 3. Processing Logic (Grouped by Video)
# ==========================================
st.header("3. Process Video Clips")

# Check for the dictionary, not df
if "video_dict" not in st.session_state:
    st.warning("⚠️ Waiting for data compilation in Section 2 to enable processing.")
else:
    skip_errors = st.checkbox("⚠️ Ignore errored rows and process valid rows anyway if pre-flight check fails.")
    
if st.button("Start Pre-flight Check and Video Cutting Process", type="primary"):
    # Basic folder validations
    if not source_folder or not os.path.isdir(source_folder):
        st.error("❌ Source folder does not exist or is invalid.")
        st.stop()
        
    if not out_mode.startswith("A") and not output_folder:
        st.error("❌ Modes B and C require a valid Output Folder path.")
        st.stop()
        
    st.subheader("🔍 Pre-flight Validation")
    val_status = st.empty()
    val_status.info("Running pre-flight checks... Please wait.")
    
    validation_errors = []
    video_cache = {} 
    
    # Get unique videos directly from the dictionary keys instead of df
    unique_videos = st.session_state.video_dict.keys()
    
    # Step A: Validate files exist and get durations instantly via OpenCV
    for vid_name_str in unique_videos:
        vid_path = fn.find_video_recursive(source_folder, vid_name_str)

        if not vid_path:
            validation_errors.append(f"Missing Video: '{vid_name_str}' could not be found in {source_folder}")
        else:
            try:
                # Ultra-fast duration check using OpenCV instead of loading MoviePy
                cap = cv2.VideoCapture(str(vid_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frames / fps if fps > 0 else 0
                cap.release()
                
                if duration <= 0:
                    validation_errors.append(f"Invalid Video: '{vid_name_str}' appears to have 0 duration or is corrupted.")
                else:
                    video_cache[vid_name_str] = {"path": vid_path, "duration": duration}
            except Exception as e:
                validation_errors.append(f"Error reading video '{vid_name_str}': {e}")

    # Step B: Validate all dictionary entries
    valid_dict = {} 
    validation_warnings = [] # NEW: We will store non-fatal auto-fixes here
    
    for vid_name_str, cuts in st.session_state.video_dict.items():
        valid_dict[vid_name_str] = [] 
        
        for cut in cuts:
            # Change global_row to our new sheet_row variable
            row_idx = cut["sheet_row"]
            sheet = cut["sheet"]
            row_has_error = False
            
            err_context = f"📄 Sheet '{sheet}' | 🎥 '{vid_name_str}' | Row {row_idx}"
            
            start_sec, err1 = fn.parse_time(cut["start_val"], err_context)
            end_or_dur_sec, err2 = fn.parse_time(cut["end_val"], err_context)
            
            if err1: 
                validation_errors.append(err1)
                row_has_error = True
            if err2: 
                validation_errors.append(err2)
                row_has_error = True

            # If time parsing was successful, do the rigorous compatibility checks
            if not err1 and not err2:
                end_sec = (start_sec + end_or_dur_sec) if st.session_state.time_mode == "Duration" else end_or_dur_sec
                
                if start_sec < 0 or end_sec < 0:
                    validation_errors.append(f"{err_context} ❌ Times cannot be negative (Start: {start_sec}s, End: {end_sec}s).")
                    row_has_error = True

                elif start_sec >= end_sec:
                    validation_errors.append(f"{err_context} ❌ Start time ({start_sec}s) must be before End time ({end_sec}s).")
                    row_has_error = True
                
                elif vid_name_str in video_cache:
                    vid_dur = video_cache[vid_name_str]["duration"]
                    
                    if start_sec >= vid_dur:
                        validation_errors.append(f"{err_context} ❌ Start time ({start_sec}s) is beyond actual video duration ({vid_dur:.1f}s).")
                        row_has_error = True
                        
                    elif end_sec > vid_dur:
                        # 🌟 THE AUTO-FIX: We intercept the error, cap the time, and append a warning (NOT an error!)
                        validation_warnings.append(
                            f"{err_context} ⚠️ End time ({end_sec}s) exceeded duration ({vid_dur:.1f}s). Auto-capped to {vid_dur:.1f}s."
                        )
                        end_sec = vid_dur # Actually apply the fix to the variable
                        # Note: We do NOT set row_has_error = True here, so it proceeds as valid!
                        
                elif not vid_name_str in video_cache:
                    row_has_error = True

            # If no critical errors, add it to the valid_dict 
            if not row_has_error:
                cut["start_sec"] = start_sec
                cut["end_sec"] = end_sec
                valid_dict[vid_name_str].append(cut)

        if not valid_dict[vid_name_str]:
            del valid_dict[vid_name_str]

    total_valid_cuts = sum(len(cuts) for cuts in valid_dict.values())

    # Handle Pre-flight Results
    if validation_errors:
        val_status.warning(f"⚠️ Pre-flight check found {len(validation_errors)} errors:")
        for err in validation_errors:
            st.write(f"- {err}")
            
        if not skip_errors:
            st.info("💡 To skip these errors and process the valid rows, check the 'Ignore errored rows' box above and click the button again.")
            st.stop() 
        else:
            # Use total_valid_cuts instead of the old valid_row_indices
            st.success(f"✅ Proceeding with the {total_valid_cuts} valid cuts.")
    else:
        val_status.success("✅ Pre-flight check passed! All files found and rows are valid.")
        
    if total_valid_cuts == 0:
        st.error("❌ No valid cuts left to process. Halting execution.")
        st.stop()

    st.subheader("⚙️ Processing Log")
    PADDING_SEC = 0.2

    progress_bar = st.progress(0, text="Initializing Pipeline...")
    log_area = st.empty()
    logs = []

    def add_log(msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        logs.append(f"[{timestamp}] {msg}")
        log_area.code("\n".join(logs[-20:]), language="text")

    results = []
    total_cuts = sum(len(cuts) for cuts in valid_dict.values())
    cuts_completed = 0
    
    # Define start_time_global BEFORE the loop starts!
    start_time_global = time.time()
    
    for base_name, cuts in valid_dict.items():
        vid_path = video_cache[base_name]["path"]
        vid_duration = video_cache[base_name]["duration"]

        # Define out_dir based on out_mode
        if out_mode.startswith("A"):
            out_dir = vid_path.parent / f"{vid_path.stem}_clips"
        else:
            output_path = Path(output_folder)
            if out_mode.startswith("B"):
                out_dir = output_path / base_name
            else:  # Mode C
                relative = vid_path.relative_to(Path(source_folder))
                out_dir = output_path / relative.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        add_log(f"--- Group: {base_name} ---")

        try:
            for cut in cuts:
                row_display_idx = cut["sheet_row"]
                
                # We already calculated the seconds in the pre-flight phase!
                start_sec = cut["start_sec"]
                end_sec = cut["end_sec"]
                
                # Apply Padding Logic for FFmpeg
                padded_start = max(0.0, start_sec - PADDING_SEC)
                padded_end = min(vid_duration, end_sec + PADDING_SEC)

                # Filename Generation using dictionary keys
                parts = [base_name]
                if cut["main_label"]:
                    parts.append(fn.sanitize_filename(cut["main_label"]))
                
                sec_labels_str = "_".join([fn.sanitize_filename(l) for l in cut["sec_labels"]])
                if sec_labels_str:
                    parts.append(sec_labels_str)
                    
                start_str = fn.seconds_to_hhmmss(start_sec)
                end_str = fn.seconds_to_hhmmss(end_sec)
                out_name = f"{'_'.join(parts)}_{start_str}_{end_str}.mp4"
                out_path = out_dir / out_name

                # Check if the file already exists before doing any work
                if out_path.exists():
                    add_log(f"Row {row_display_idx}: ⏭️ '{out_name}' already exists. Skipping.")
                    results.append({"Row": row_display_idx, "File": base_name, "Status": "Skipped", "Details": "File already exists."})
                    
                    # We still need to update the progress bar so the ETA stays accurate!
                    cuts_completed += 1
                    elapsed_time = time.time() - start_time_global
                    time_per_cut = elapsed_time / cuts_completed
                    remaining_cuts = total_cuts - cuts_completed
                    eta_seconds = remaining_cuts * time_per_cut
                    
                    prog_text = f"Processed {cuts_completed} of {total_cuts} cuts | ETA: {fn.format_time_str(eta_seconds)}"
                    progress_bar.progress(cuts_completed / total_cuts, text=prog_text)
                    
                    # 'continue' forces the loop to skip the FFmpeg step and move to the next cut
                    continue

                # Cut Execution
                add_log(f"Row {row_display_idx}: Slicing from {padded_start:.1f}s to {padded_end:.1f}s...")
            
                cmd = [
                    FFMPEG_BINARY,
                    "-y",                   
                    "-i", str(vid_path),    
                    "-ss", str(padded_start), 
                    "-to", str(padded_end),   
                    "-c:v", "libx264",      
                    "-c:a", "aac",          
                    str(out_path)           
                ]
                
                # Execute the FFmpeg command
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                # Check if FFmpeg failed internally
                if process.returncode != 0:
                    raise Exception(f"FFmpeg error: {process.stderr}")

                add_log(f"Row {row_display_idx}: ✅ Successfully wrote '{out_name}'.")
                results.append({"Row": row_display_idx, "File": base_name, "Status": "Success", "Details": f"Clipped: {out_name}"})
                
                # Update Progress & ETA
                cuts_completed += 1
                elapsed_time = time.time() - start_time_global
                time_per_cut = elapsed_time / cuts_completed
                remaining_cuts = total_cuts - cuts_completed
                eta_seconds = remaining_cuts * time_per_cut
                
                prog_text = f"Processed {cuts_completed} of {total_cuts} cuts | ETA: {fn.format_time_str(eta_seconds)}"
                progress_bar.progress(cuts_completed / total_cuts, text=prog_text)


        except Exception as e:
                add_log(f"❌ [CRITICAL] Failed processing video '{base_name}': {str(e)}")
                # CHANGE 6: Iterate through our dictionary 'cuts' instead of a Pandas 'group'
                for cut in cuts:
                    row_idx = cut["global_row"]
                    if not any(r.get("Row") == row_idx for r in results):
                        results.append({"Row": row_idx, "File": base_name, "Status": "Error", "Details": str(e)})
                        cuts_completed += 1
                progress_bar.progress(cuts_completed / total_cuts, text=f"Processed {cuts_completed} of {total_cuts} cuts.")
            
    add_log("\n🎉 All Jobs Complete!")
    progress_bar.progress(1.0, text=f"Job Complete! Total time: {fn.format_time_str(time.time() - start_time_global)}")
    
    # Summary Table
    st.subheader("Process Summary")
    results_df = pd.DataFrame(results).sort_values(by="Row")
    
    # Added color mapping for the 'Skipped' status (Orange/Yellow)
    def style_status(val):
        if val == 'Error':
            return 'color: #ff4b4b; font-weight: bold;' # Red
        elif val == 'Skipped':
            return 'color: #ffa500; font-weight: bold;' # Orange
        else:
            return 'color: #00cc66; font-weight: bold;' # Green

    st.dataframe(results_df.style.map(style_status, subset=['Status']), width='stretch')