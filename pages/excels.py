import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import folium # Maps core library
from streamlit_folium import st_folium # Streamlit wrapper for Folium

# --- THE FIX: Hex to RGBA Converter ---
def hex_to_rgba(hex_code, opacity):
    """Converts a hex color code to an rgba string for precise Plotly transparency."""
    hex_code = hex_code.lstrip('#')
    if len(hex_code) == 6:
        r = int(hex_code[0:2], 16)
        g = int(hex_code[2:4], 16)
        b = int(hex_code[4:6], 16)
        return f"rgba({r},{g},{b},{opacity})"
    return hex_code # Fallback

# --- Page Config ---
st.set_page_config(page_title="Ultra-Explorer Pro", layout="wide")

# --- Initialize Session State ---
# This allows us to keep track of configurations for each file independently
if 'file_configs' not in st.session_state:
    st.session_state.file_configs = {}

st.title("📊 Multi-File Unified Visualizer")
st.markdown("Upload files, configure each individually, and see them on a single interactive timeline.")

# --- Sidebar: Global UI Settings ---
with st.sidebar:
    st.header("🎨 Global Layout")

    bg_color = st.color_picker("Plot Background Color", "#FFFFFF")
    paper_color = st.color_picker("Paper Background Color", "#F0F2F6")
    global_font_color = st.color_picker("Global Font Color", "#262730")

    chart_title = st.text_input("Chart Title", "My Unified Visualization")
    chart_subtitle = st.text_input("Chart Subtitle", "Comparing multiple datasets")

    plot_height = st.slider("Plot Height", 400, 2000, 600)
    font_size = st.number_input("Global Font Size", 5, 50, 15)
    legend_pos = st.selectbox("Legend Position",  ['v', 'h'], index=0)
    text_pos = st.selectbox("Data Label Position", ['top left', 'top center', 'top right', 'middle left', 'middle center', 'middle right', 'bottom left', 'bottom center', 'bottom right'], index=0)
    bar_text_pos = st.selectbox("Bar Label Position", ['inside', 'outside', 'auto', 'none'], index=0, help="For Bar charts, 'auto' will place labels inside bars if space allows, otherwise outside.")
    st.divider()

# --- File Ingestion ---
uploaded_files = st.file_uploader(
    "Upload CSV or Excel files", 
    type=["csv", "xlsx", "xls"], 
    accept_multiple_files=True
)

if uploaded_files:
    # 1. Store and Sniff Data
    for file in uploaded_files:
        if file.name not in st.session_state.file_configs:
            try:
                # Robust CSV detection
                if file.name.endswith('.csv'):
                    temp_df = pd.read_csv(file, sep=None, engine='python', on_bad_lines='warn')
                else:
                    temp_df = pd.read_excel(file)

                # Capture the first column name as a safe default
                default_col = temp_df.columns[0]
                
                # Initialize ALL keys that the rest of the script expects
                st.session_state.file_configs[file.name] = {
                    "df": temp_df,
                    "active": True,
                    "x_col": default_col,      
                    "break_col": "All",        
                    "is_date": False,
                    "date_format": "%d.%m.%Y",
                    "styles": {},              
                    "cumulative": False,
                    
                    # --- NEW: Map Defaults ---
                    "map_active": False,
                    "lat_col": default_col,
                    "lon_col": default_col,
                    "max_samples": 1000,
                    "start_date": None,
                    "end_date": None
                }
                
                # Auto-date conversion
                for col in temp_df.columns:
                    if temp_df[col].dtype == 'object':
                        try:
                            temp_df[col] = pd.to_datetime(temp_df[col])
                        except: pass
                
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")

    # 2. Configuration Interface
    st.header("⚙️ Configure Data Series")
    
    # We use tabs to keep the UI clean if many files are uploaded
    if st.session_state.file_configs:
        tabs = st.tabs(list(st.session_state.file_configs.keys()))
        
        for i, (fname, config) in enumerate(st.session_state.file_configs.items()):
            with tabs[i]:
                c1, c2, c3 = st.columns([1, 1, 2])
                local_df = config["df"]
                
                # --- Inside the tab/loop for each file ---
                with c1:
                    st.subheader("Selection")
                    config["active"] = st.checkbox("Include in Plot", value=config.get("active", True), key=f"act_{fname}")
                    # Example for the X-Axis selectbox
                    x_col = st.selectbox(
                        f"X-Axis (Time/Group)", 
                        local_df.columns, 
                        index=list(local_df.columns).index(config.get("x_col", local_df.columns[0])), # Sync index
                        key=f"x_{fname}"
                    )

                    # ADDITION: Prepend "All" to the list of columns
                    breakdown_options = ["All"] + list(local_df.columns)
                    break_col = st.selectbox(f"Breakdown Column", breakdown_options, key=f"brk_{fname}")

                    config["x_col"] = x_col
                    config["break_col"] = break_col

                    st.subheader("Time Configuration")
                    # 1. SHOW THE RAW PREVIEW
                    sample_raw = str(local_df[x_col].iloc[0])
                    st.info(f"Raw Sample: `{sample_raw}`")

                    # 2. ASK FOR THE FORMAT
                    is_date = st.checkbox("Process as Date", value=True, key=f"isdate_{fname}")
                    
                    if is_date:
                        st.markdown("[Date Format Guide](https://strftime.org/)")
                        # Default to a common format, but let user override
                        date_format = st.text_input(
                            "Enter Date Format", 
                            value="%d.%m.%Y", 
                            key=f"fmt_{fname}",
                            help="Example: %d.%m.%Y for 14.08.2023 or %d/%m/%Y for 14/08/2023"
                        )
                        config["date_format"] = date_format
                        config["is_date"] = True

                with c2:
                    st.subheader("Filters")
                    # LOGIC CHANGE: Only show multiselect if a specific column is chosen
                    if break_col != "All":
                        unique_vals = local_df[break_col].unique().tolist()
                        chosen_vals = st.multiselect(f"Keep values in {break_col}", unique_vals, default=unique_vals, key=f"filt_{fname}")
                    else:
                        st.info("Showing total count for this file.")
                        chosen_vals = [] # Not used in "All" mode

                    config["chosen_vals"] = chosen_vals

                with c3:
                    st.subheader("Trace Customization")
                    
                    # Determine which categories exist for this file
                    if break_col == "All":
                        cats_to_style = ["Total Count"]
                    else:
                        cats_to_style = local_df[break_col].unique().tolist()

                    # Create a nested dictionary in session state to store individual trace styles
                    if "styles" not in config: config["styles"] = {}

                    for cat in cats_to_style:
                        with st.expander(f"🎨 Style: {cat}", expanded=False):
                            cc1, cc2 = st.columns(2)
                            with cc1:
                                config["styles"][cat] = {
                                    "type": st.selectbox("Type", ["Bar", "Line", "Scatter", "Area"], index=0, key=f"t_{fname}_{cat}"),
                                    "color": st.color_picker("Color", value="#636EFA", key=f"c_{fname}_{cat}"),
                                    "size": st.slider("Size/Width", 1, 20, 5, key=f"s_{fname}_{cat}"),
                                    "font_color": st.color_picker("Label/Text Color", value="#000000", key=f"fc_{fname}_{cat}"),
                                    "show_labels": st.checkbox("Show Data Labels", value=False, key=f"lbl_{fname}_{cat}")
                                }
                            with cc2:
                                config["styles"][cat].update({
                                    "symbol": st.selectbox("Dot Shape", ["circle", "square", "diamond", "cross"], key=f"sym_{fname}_{cat}"),
                                    "opacity": st.slider("Opacity", 0.0, 1.0, 0.7, key=f"op_{fname}_{cat}"),
                                    "cumulative": st.checkbox("Cumulative", value=False, key=f"cum_{fname}_{cat}")
                                })

                
                st.divider()
                st.subheader("🗺️ Geographic Mapping")
                
                # Main toggle to activate the map for this file
                config["map_active"] = st.checkbox("Plot Data on Map", value=config.get("map_active", False), key=f"map_act_{fname}")

                if config["map_active"]:
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        # 1. Coordinate Selection
                        config["lat_col"] = st.selectbox("Latitude Column", local_df.columns, index=list(local_df.columns).index(config.get("lat_col", local_df.columns[0])), key=f"lat_{fname}")
                        config["lon_col"] = st.selectbox("Longitude Column", local_df.columns, index=list(local_df.columns).index(config.get("lon_col", local_df.columns[0])), key=f"lon_{fname}")
                        
                        # 2. Sampling Filter
                        config["max_samples"] = st.number_input("Max Points to Render", min_value=10, max_value=20000, value=config.get("max_samples", 1000), step=100, help="High numbers may crash the browser.", key=f"samp_{fname}")

                    with mc2:
                        # 3. Time Interval Filter
                        st.write("⏱️ **Time Interval Filter**")
                        if config.get("is_date"):
                            # We use None as default to allow 'open-ended' filtering if left blank
                            config["start_date"] = st.date_input("Start Date", value=config.get("start_date", None), key=f"sd_{fname}")
                            config["end_date"] = st.date_input("End Date", value=config.get("end_date", None), key=f"ed_{fname}")
                        else:
                            st.info("Enable 'Process as Date' in the Selection tab to use time filtering.")

    # 3. Dynamic Plotting
    st.divider()
    fig = go.Figure()

    # --- Inside the Plotting Section ---
    for fname, config in st.session_state.file_configs.items():

        if not config.get("active") or "x_col" not in config:
            st.warning(f"⚠️ File '{fname}' is missing required configuration. Please check the settings.")
            continue

        f_df = config["df"].copy()
        x_col = config["x_col"]

        # --- 1. FORCE UNIFORM DATETIME (Manual Format Mode) ---
        if config.get("is_date"):
            # Convert raw data to string first to ensure consistency
            raw_strings = config["df"][config["x_col"]].astype(str).str.strip()
            
            # Use the format provided by the user in the UI
            f_df[x_col] = pd.to_datetime(
                raw_strings,
                format=config.get("date_format"),
                errors='coerce'
            )
            
            # Provide immediate feedback if the user's format failed
            if f_df[x_col].isna().all():
                st.warning(f"⚠️ File '{fname}': Format '{config.get('date_format')}' does not match '{raw_strings.iloc[0]}'")

        # 2. FILTERING & AGGREGATION (as previously discussed)
        if config["break_col"] == "All":
            plot_data = f_df.groupby(x_col).size().reset_index(name='Freq')
            plot_data['Category'] = "Total Count"
        else:
            filtered = f_df[f_df[config["break_col"]].isin(config["chosen_vals"])]
            plot_data = filtered.groupby([x_col, config["break_col"]]).size().reset_index(name='Freq')
            plot_data.rename(columns={config["break_col"]: 'Category'}, inplace=True)

        # 3. CHRONOLOGICAL SORTING
        # This is the most important line for your request.
        # It ensures the line connects Jan -> Feb -> March, not alphabetically.
        plot_data = plot_data.sort_values(by=x_col).dropna(subset=[x_col])

        # --- 4: PLOTTING THE TRACES ---
        for cat in plot_data['Category'].unique():
            cat_df = plot_data[plot_data['Category'] == cat].sort_values(config["x_col"])
            
            style = config["styles"].get(cat, {"type": "Line", "color": "#636EFA", "size": 5, "symbol": "circle", "opacity": 0.7, "cumulative": False, "show_labels": False})
            
            y_vals = cat_df['Freq']
            if style.get("cumulative"):
                y_vals = y_vals.cumsum()

            show_text = style.get("show_labels", False)
            scatter_mode = 'lines+markers+text' if show_text else 'lines+markers'
            area_mode = 'lines+text' if show_text else 'lines'
            pure_scatter_mode = 'markers+text' if show_text else 'markers'

            # --- THE FIX: Generate the transparent color ---
            rgba_color = hex_to_rgba(style["color"], style["opacity"])

            # Using specific parameters for each type
            if style["type"] == "Bar":
                fig.add_trace(go.Bar(
                    x=cat_df[config["x_col"]], y=y_vals, 
                    name=f"{fname}: {cat}",
                    # FIX: Apply RGBA to the fill, but keep a solid 1px line border
                    marker=dict(color=rgba_color, line=dict(color=style["color"], width=1.5)),
                    text=y_vals if show_text else None,
                    textposition='auto', 
                    textfont=dict(color=style.get("font_color", "#000000")),
                    width=[(style["size"]/20) * 86400000] * len(cat_df) if config.get("is_date") else None 
                ))
                
            elif style["type"] == "Line":
                fig.add_trace(go.Scatter(
                    x=cat_df[config["x_col"]], y=y_vals, 
                    mode=scatter_mode, 
                    name=f"{fname}: {cat}",
                    # FIX: Solid line, but transparent markers with solid borders
                    line=dict(color=style["color"], width=style["size"]),
                    marker=dict(symbol=style["symbol"], size=style["size"]+2, color=rgba_color, line=dict(color=style["color"], width=1.5)),
                    text=y_vals if show_text else None,
                    textposition=text_pos, 
                    textfont=dict(color=style.get("font_color", "#000000")) 
                ))
                
            elif style["type"] == "Area":
                fig.add_trace(go.Scatter(
                    x=cat_df[config["x_col"]], y=y_vals, 
                    mode=area_mode, 
                    fill='tozeroy', 
                    name=f"{fname}: {cat}",
                    line=dict(color=style["color"], width=style["size"]),
                    # FIX: Apply explicit RGBA to the fill area
                    fillcolor=rgba_color,
                    text=y_vals if show_text else None,
                    textposition=text_pos,
                    textfont=dict(color=style.get("font_color", "#000000")) 
                ))
                
            elif style["type"] == "Scatter":
                fig.add_trace(go.Scatter(
                    x=cat_df[config["x_col"]], y=y_vals, 
                    mode=pure_scatter_mode, 
                    name=f"{fname}: {cat}",
                    # FIX: Transparent dots with crisp solid borders
                    marker=dict(color=rgba_color, size=style["size"]*2, symbol=style["symbol"], line=dict(color=style["color"], width=1.5)),
                    text=y_vals if show_text else None,
                    textposition=text_pos,
                    textfont=dict(color=style.get("font_color", "#000000")) 
                ))
                
    # Create a generic or dynamic label
    # If you want to show the specific column name of the first active file:
    active_files = [c["x_col"] for c in st.session_state.file_configs.values() if c.get("active")]
    x_axis_label = active_files[0] if active_files else "X-Axis"

    # --- Legend Position Engine ---
    legend_layout = dict(
        orientation=legend_pos,
        font=dict(color=global_font_color)
    )
    
    # If horizontal, push it to the top right to avoid chart overlap
    if legend_pos == 'h':
        legend_layout.update(yanchor="bottom", y=1.02, xanchor="right", x=1)

    # Global Layout Enforcement
    # --- Inside the Dynamic Plotting Section (at the end) ---
    
    # 1. Construct the HTML Title String
    # If there is a subtitle, we add a line break (<br>) and make the subtitle slightly smaller using <i> or <span>
    if chart_subtitle:
        full_title_text = f"<b>{chart_title}</b><br><span style='font-size: {font_size * 0.8}px; color: {global_font_color};'>{chart_subtitle}</span>"
    else:
        full_title_text = f"<b>{chart_title}</b>"

    # 2. Global Layout Enforcement
    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=paper_color,
        height=plot_height,
        template="plotly_white",
        barmode='group',
        
        # --- NEW: Inject the Title ---
        title=dict(
            text=full_title_text,
            x=0.5, # 0.5 centers the title perfectly. 0 is left, 1 is right.
            xanchor='center',
            yanchor='top',
            font=dict(size=font_size * 1.5, color=global_font_color) # Make main title 1.5x bigger than global font
        ),

        # This forces the global font color on all text elements
        font=dict(
            size=font_size, 
            color=global_font_color 
        ),
        
        legend=dict(
            orientation=legend_pos,
            font=dict(color=global_font_color) 
        ),
        
        xaxis=dict(
            title=dict(text=x_axis_label, font=dict(color=global_font_color)),
            tickfont=dict(color=global_font_color),
            type="date", 
            tickformat="%d-%m-%Y",  
            tickangle=-45
        ),
        
        yaxis=dict(
            title=dict(text="Counts", font=dict(color=global_font_color)),
            tickfont=dict(color=global_font_color)
        )
    )

    st.plotly_chart(fig, width='stretch')



    # ==========================================
    # --- GLOBAL MAP RENDERER ---
    # ==========================================
    
    if any(c.get("map_active") for c in st.session_state.file_configs.values()):
        st.divider()
        st.header("🌍 Global Map Visualization")
        
        # Start without a fixed location
        m = folium.Map(zoom_start=2, tiles="CartoDB positron")
        
        # We will collect all valid coordinates to auto-center the map later
        all_lats = []
        all_lons = []

        for fname, config in st.session_state.file_configs.items():
            if not config.get("map_active"):
                continue
                
            df_map = config["df"].copy()
            x_col = config["x_col"]
            
            # --- FILTER 1: Clean Coordinates (ROBUST VERSION) ---
            # Drop empty rows first
            df_map = df_map.dropna(subset=[config["lat_col"], config["lon_col"]])
            
            # Force everything to string, replace commas with dots, remove hidden spaces
            df_map[config["lat_col"]] = df_map[config["lat_col"]].astype(str).str.replace(',', '.').str.strip()
            df_map[config["lon_col"]] = df_map[config["lon_col"]].astype(str).str.replace(',', '.').str.strip()
            
            # Now convert to numeric. Anything that STILL fails becomes NaN
            df_map[config["lat_col"]] = pd.to_numeric(df_map[config["lat_col"]], errors='coerce')
            df_map[config["lon_col"]] = pd.to_numeric(df_map[config["lon_col"]], errors='coerce')
            
            # Drop the ones that failed conversion
            df_map = df_map.dropna(subset=[config["lat_col"], config["lon_col"]])

            # --- DIAGNOSTIC CHECK ---
            # This tells you if your filters deleted everything
            if df_map.empty:
                st.warning(f"⚠️ No valid coordinates found for '{fname}' after applying filters. Check your column selection and date ranges.")
                continue # Skip drawing pins for this file
            else:
                st.success(f"📍 Plotting {len(df_map)} points for '{fname}'...")
            
            # --- FILTER 2: Time Interval ---
            if config.get("is_date"):
                # Parse the dates exactly as we did for the Plotly graph
                df_map[x_col] = pd.to_datetime(
                    df_map[x_col].astype(str).str.strip(),
                    format=config.get("date_format"),
                    errors='coerce'
                )
                
                # Apply the boundaries if the user selected them
                if config.get("start_date"):
                    df_map = df_map[df_map[x_col].dt.date >= config["start_date"]]
                if config.get("end_date"):
                    df_map = df_map[df_map[x_col].dt.date <= config["end_date"]]

            # --- FILTER 3: Max Sampling ---
            if len(df_map) > config["max_samples"]:
                # .sample() picks random rows, ensuring we don't overload Folium
                df_map = df_map.sample(n=config["max_samples"], random_state=42)

            # --- RENDER MARKERS ---
            for _, row in df_map.iterrows():
                # Extract lat/lon safely
                lat = float(row[config["lat_col"]])
                lon = float(row[config["lon_col"]])
                
                # Keep track of coordinates for bounding box
                all_lats.append(lat)
                all_lons.append(lon)

                # --- Dynamic Color Mapping ---
                # 1. Determine which category this row belongs to
                if config["break_col"] == "All":
                    cat = "Total Count"
                else:
                    cat = str(row[config["break_col"]]) # Force string to match the dictionary keys
                
                # 2. Retrieve the custom color assigned in the styling tab. 
                # If it doesn't exist (e.g., user hasn't opened the styling tab yet), fallback to default blue.
                cat_style = config["styles"].get(cat, {})
                point_color = cat_style.get("color", "#636EFA")
                
                # 3. Add the marker with the matched color
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,
                    color=point_color,        # This colors the outer border of the dot
                    fill_color=point_color,   # This colors the inside of the dot
                    fill=True,
                    fill_opacity=0.7,
                    # We also add the Category to the hover tooltip for even better interpretability!
                    tooltip=f"<b>File:</b> {fname}<br><b>Category:</b> {cat}<br><b>Lat:</b> {lat}<br><b>Lon:</b> {lon}" 
                ).add_to(m)

        # --- AUTO-CENTER THE MAP ---
        if all_lats and all_lons:
            # Tell Folium to zoom exactly to where the points are
            m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

        # Display the map in Streamlit
        st_folium(m, width=1200, height=600, returned_objects=[])

else:
    st.info("👋 Welcome! Please upload one or more CSV/Excel files to start exploring.")