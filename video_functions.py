import pandas as pd
import re
from pathlib import Path
import datetime
import streamlit as st
import io

# --- HELPER FUNCTIONS ---

def format_time_str(seconds):
    """Formats raw seconds into MM:SS or HH:MM:SS for the ETA display."""
    return str(datetime.timedelta(seconds=int(seconds)))

def parse_time(time_val, row_idx=None):
    """
    Ultra-resilient time parser. 
    Extracts numbers intelligently, handling messy delimiters and AM/PM modifiers.
    Returns: (total_seconds (float), error_message (str or None))
    """
    # 1. Null/Empty Check
    if pd.isna(time_val) or str(time_val).strip() == "":
        return None, f"Row {row_idx}: Empty time value."
    
    time_str = str(time_val).strip().lower()
    
    # 2. Pure Float/Int Bypass 
    # (If it's already a clean number like "12.5", treat it directly as seconds)
    try:
        return float(time_str), None
    except ValueError:
        pass

    # 3. Detect AM/PM (This forces Left-to-Right "Time of Day" processing)
    is_am = 'am' in time_str or 'a.m.' in time_str
    is_pm = 'pm' in time_str or 'p.m.' in time_str
    is_time_of_day = is_am or is_pm

    # 4. Clean up the string
    # Remove all alphabetical characters ('am', 'pm', 'hrs', 'sec', etc.)
    # This leaves behind only numbers and whatever messy symbols they typed.
    clean_str = re.sub(r'[a-z]', '', time_str)

    # 5. Extract all sequential numeric chunks
    # This completely bypasses the delimiter problem. It just grabs the numbers.
    chunks = re.findall(r'\d+', clean_str)
    
    if not chunks:
        return None, f"Row {row_idx}: No valid numbers found in '{time_val}'."
    
    if len(chunks) > 4:
        return None, f"Row {row_idx}: Too many time segments in '{time_val}'."

    # Convert strings to floats
    parts = [float(x) for x in chunks]

    # 6. Handle Fractional Seconds gracefully
    # If there are exactly 4 parts, the 4th is ALWAYS a fraction of a second (HH, MM, SS, ms)
    if len(parts) == 4:
        ms_str = chunks[3]
        # Dynamically calculate the fraction (e.g., '123' -> 0.123, '5' -> 0.5)
        fraction = float(ms_str) / (10 ** len(ms_str))
        parts = [parts[0], parts[1], parts[2] + fraction]

    secs = 0.0

    # 7. Apply Parsing Logic
    if is_time_of_day:
        # Time of day is read Left-to-Right (Hour -> Minute -> Second)
        hr = parts[0]
        if is_pm and hr < 12:
            hr += 12
        if is_am and hr == 12:
            hr = 0
        
        secs += hr * 3600
        if len(parts) > 1:
            secs += parts[1] * 60
        if len(parts) > 2:
            secs += parts[2]
            
    else:
        # Duration is read Right-to-Left (Second -> Minute -> Hour)
        # This preserves your original script's logic for things like "12:30"
        parts.reverse()
        secs += parts[0] # Seconds
        if len(parts) > 1:
            secs += parts[1] * 60 # Minutes
        if len(parts) > 2:
            secs += parts[2] * 3600 # Hours

    return secs, None

def find_video_recursive(source_dir, target_name):
    """
    Recursively searches all subfolders for a video matching the target_name.
    """
    source_path = Path(source_dir)
    target_stem = Path(target_name).stem.lower()
    valid_exts = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    
    for file in source_path.rglob('*'):
        if file.is_file() and file.stem.lower() == target_stem and file.suffix.lower() in valid_exts:
            return file # Returns a pathlib.Path object
    return None

def sanitize_filename(name):
    """Removes invalid filename characters."""
    return re.sub(r'[\\/*?:"<>|]', "", str(name)).replace(" ", "_")

def seconds_to_hhmmss(total_seconds):
    """
    Converts raw seconds (float or int) into an hh.mm.ss string format.
    Rounds down to the nearest whole second for clean filenames.
    """
    # Calculate hours, minutes, and seconds
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    
    # Return formatted string with leading zeros (e.g., 01.05.09)
    return f"{h:02d}.{m:02d}.{s:02d}"

@st.cache_data
def load_spreadsheet(file_bytes, file_name, header_row=1):
    file_like_object = io.BytesIO(file_bytes)
    
    # Pandas uses 0-based indexing for headers. So if user says row 1, we pass 0.
    pd_header = header_row - 1 
    
    if file_name.lower().endswith('.csv'):
        df = pd.read_csv(file_like_object, header=pd_header)
        return {"Sheet1": df}, df.columns.tolist()
    else:
        sheets = pd.read_excel(file_like_object, sheet_name=None, header=pd_header)
        
        common_cols = None
        for _, df in sheets.items():
            cols = set(df.columns)
            if common_cols is None:
                common_cols = cols
            else:
                common_cols = common_cols.intersection(cols)
                
        return sheets, list(common_cols)
    
# Add header_row as a parameter
def build_video_dict(sheets, vid_col, start_col, end_col, main_lbl_col, sec_lbl_cols, header_row):
    video_dict = {}
    
    for sheet_name, df in sheets.items():
        for idx, row in enumerate(df.to_dict('records')):
            
            # 🌟 DYNAMIC MATH: 
            # If header is row 3, the first data row is row 4. 
            # enumerate 'idx' starts at 0. So: 0 + 3 + 1 = 4. Perfect!
            excel_row = idx + header_row + 1 
            
            vid_name = str(row[vid_col])
            
            # Skip empty rows
            if pd.isna(row[vid_col]) or vid_name.strip() == "" or vid_name == "nan":
                continue
                
            if vid_name not in video_dict:
                video_dict[vid_name] = []
                
            sec_labels = [str(row[c]) for c in sec_lbl_cols if pd.notna(row[c])]
            main_label = row[main_lbl_col] if main_lbl_col != "None" and pd.notna(row[main_lbl_col]) else None
            
            video_dict[vid_name].append({
                "sheet_row": excel_row, 
                "sheet": sheet_name,
                "start_val": row[start_col],
                "end_val": row[end_col],
                "main_label": main_label,
                "sec_labels": sec_labels
            })
            
    return video_dict