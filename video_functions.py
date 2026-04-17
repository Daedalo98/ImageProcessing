import streamlit as st
import pandas as pd
import os
import re
import difflib

# --- HELPER FUNCTIONS ---

def parse_time(time_val):
    """
    Parses flexible time formats (hh:mm:ss, mm:ss.ms, h;mm,ss, etc.) into total seconds.
    """
    if pd.isna(time_val): return None
    if isinstance(time_val, (int, float)): return float(time_val)
    
    time_str = str(time_val).strip()
    
    # Replace common typo delimiters (semicolons, commas) with colons
    # Note: We leave '.' intact as it usually denotes fractional seconds.
    time_str = re.sub(r'[;,]', ':', time_str)
    
    parts = time_str.split(':')
    secs = 0.0
    try:
        if len(parts) == 3: # hh:mm:ss
            secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2: # mm:ss
            secs = int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1: # ss
            secs = float(parts[0])
        else:
            return None
        return secs
    except ValueError:
        return None

def find_video_file(folder, target_name):
    """
    Finds the actual video file in the directory using exact match, 
    extension-less match, or fuzzy matching.
    """
    if not os.path.isdir(folder): return None
    
    # List all files in the directory
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not files: return None

    # 1. Exact match
    if target_name in files: return target_name
    
    target_stem = os.path.splitext(target_name)[0].lower()
    file_map = {os.path.splitext(f)[0].lower(): f for f in files}
    
    # 2. Exact match ignoring extension
    if target_stem in file_map: return file_map[target_stem]
    
    # 3. Fuzzy matching
    matches = difflib.get_close_matches(target_stem, file_map.keys(), n=1, cutoff=0.7)
    if matches:
        return file_map[matches[0]]
        
    return None

def sanitize_filename(name):
    """Removes invalid characters from filenames."""
    return re.sub(r'(?u)[^-\w.]', '_', str(name).strip())