import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import folium # Maps core library
from streamlit_folium import st_folium # Streamlit wrapper for Folium
from folium.plugins import MarkerCluster, HeatMap

import pyproj
from pyproj import Transformer

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

def fix_coords(row):
    lat_s = str(row[config["lat_col"]]).strip()
    lon_s = str(row[config["lon_col"]]).strip()
    
    # 1. Clean obvious thousands separators
    if lat_s.count(',') > 1: lat_s = lat_s.replace(',', '')
    if lon_s.count(',') > 1: lon_s = lon_s.replace(',', '')
    if lat_s.count('.') > 1: lat_s = lat_s.replace('.', '')
    if lon_s.count('.') > 1: lon_s = lon_s.replace('.', '')

    try:
        # 2. Convert standard European comma to dot, then to float
        lat_f = float(lat_s.replace(',', '.'))
        lon_f = float(lon_s.replace(',', '.'))
    except ValueError:
        return pd.Series([None, None])

    # 3. Fix Mangled Eastings (e.g., 794.901 read as decimal)
    if 180 < lon_f < 1000:
        lon_f = lon_f * 1000

    # ==========================================
    # --- NEW: THE INTEGERIZED DEGREE FIX ---
    # ==========================================
    # If the user DID NOT check the "Projected Coordinates" box, 
    # but the number is massive, it is multiplied by 100,000!
    if lat_f > 90.0 and not config.get("is_projected"):
        # Check if dividing by 100,000 puts it in valid GPS range
        if -90.0 <= (lat_f / 100000.0) <= 90.0:
            lat_f = lat_f / 100000.0
            lon_f = lon_f / 100000.0

    # 4. Auto-Transformation (Meters to Degrees)
    # ONLY happens if the user explicitly checks the EPSG box in the UI
    if config.get("is_projected") and config.get("epsg_code"):
        try:
            lon_f, lat_f = transformer.transform(lon_f, lat_f)
        except:
            pass

    return pd.Series([lat_f, lon_f])

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
                    temp_df = pd.read_csv(file, sep=';', engine='python', on_bad_lines='warn')
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
                        config["max_samples"] = st.number_input("Max Points to Render", min_value=1, max_value=20000, value=config.get("max_samples", 1000), step=10, help="High numbers may crash the browser.", key=f"samp_{fname}")

                        config["map_render_mode"] = st.selectbox(
                            "Render Mode", 
                            ["Scatter (Individual)", "Marker Clusters", "Heatmap"], 
                            index=1, # Default to Clusters for best balance
                            key=f"rmode_{fname}",
                            help="Clusters: Groups markers for performance. Heatmap: Shows density hotspots."
                        )

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
    # --- HYBRID GLOBAL MAP RENDERER ---
    # ==========================================

    if any(c.get("map_active") for c in st.session_state.file_configs.values()):
        st.divider()
        st.header("🌍 Global Map Visualization")

        # --- INITIALIZE MAP ---
        m = folium.Map(zoom_start=2, tiles="CartoDB positron")
        all_lats, all_lons = [], []
        legend_items = {} # We will populate this in the loop!

        # ==========================================
        # --- FEATURE 1: MAP TITLE & SUBTITLE ---
        # ==========================================
        # We reuse the chart_title and chart_subtitle from your sidebar!
        title_html = f'''
             <div style="position: absolute; top: 10px; left: 50px; width: auto; max-width: 400px;
                         z-index: 9999; background-color: rgba(255, 255, 255, 0.8); 
                         border-radius: 8px; padding: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
                 <h3 style="margin-top:0; margin-bottom:5px; color:{global_font_color};">{chart_title}</h3>
                 <h5 style="margin:0; color: gray;">{chart_subtitle}</h5>
             </div>
             '''
        m.get_root().html.add_child(folium.Element(title_html))


        all_lats, all_lons = [], []
        for fname, config in st.session_state.file_configs.items():
            if not config.get("map_active"): continue
                
            df_map = config["df"].copy()

            # --- PREPARE DATA FOR RENDERING ---

            # ==========================================
            # --- FILTER 1: SMART COORDINATE ENGINE ---
            # ==========================================
            
            # Capture the raw evidence BEFORE filtering
            raw_sample = config["df"][[config["lat_col"], config["lon_col"]]].head(5)
            df_map = df_map.dropna(subset=[config["lat_col"], config["lon_col"]])

            # Initialize cartographic transformer 
            # EPSG:32632 is standard UTM Zone 32N (covers most of Germany)
            # EPSG:4326 is the WGS84 system that Folium requires
            transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)

            # Apply the engine to clean and standardize all coordinates simultaneously
            df_map[[config["lat_col"], config["lon_col"]]] = df_map.apply(fix_coords, axis=1)
            
            # Drop any rows that completely failed parsing
            df_map = df_map.dropna(subset=[config["lat_col"], config["lon_col"]])
            
            # 5. GEOGRAPHIC REALITY CHECK 
            # We enforce this again just to ensure nothing slipped past
            df_map = df_map[
                (df_map[config["lat_col"]] >= -90.0) & (df_map[config["lat_col"]] <= 90.0) &
                (df_map[config["lon_col"]] >= -180.0) & (df_map[config["lon_col"]] <= 180.0)
            ]

            # --- DIAGNOSTIC WARNING WITH SAMPLES ---
            if df_map.empty:
                st.warning(f"⚠️ No valid GPS coordinates found for '{fname}' using columns: '{config['lat_col']}' & '{config['lon_col']}'.")
                st.info("🔍 **Debug Inspector: Raw Data Sample**")
                st.dataframe(raw_sample, width='stretch')
                continue

            # ==========================================
            # --- FILTER 2: Time Interval ---
            # ==========================================
        
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

            # ==========================================
            # --- FILTER 3: Max Sampling ---
            # ==========================================

            if len(df_map) > config["max_samples"]:
                # .sample() picks random rows, ensuring we don't overload Folium
                df_map = df_map.sample(n=config["max_samples"], random_state=42)

            # --- PREPARE DATA FOR RENDERING ---
            # We need a list of [lat, lon] for Heatmaps or individual markers

            points = []
            for _, row in df_map.iterrows():
                lat = row[config["lat_col"]]
                lon = row[config["lon_col"]]
                
                lat, lon = float(lat), float(lon)
                all_lats.append(lat); all_lons.append(lon)
                
                # Identify category and color
                cat = "Total Count" if config["break_col"] == "All" else str(row[config["break_col"]])
                color = config["styles"].get(cat, {}).get("color", "#636EFA")

                # Identify category and color
                cat = "Total Count" if config["break_col"] == "All" else str(row[config["break_col"]])
                cat_style = config["styles"].get(cat, {})
                color = cat_style.get("color", "#636EFA")
                opacity = cat_style.get("opacity", 0.7) # FEATURE: Grab the opacity!
                
                # Build the legend dictionary dynamically
                if cat not in legend_items:
                    legend_items[cat] = color

                points.append({'loc': [lat, lon], 'cat': cat, 'color': color, 'opacity': opacity})
                

            # --- SWITCH RENDER MODE ---

            render_mode = config.get("map_render_mode", "Marker Clusters")

            # ==========================================
            # --- NEW: GROUP POINTS BY CATEGORY ---
            # ==========================================
            # To apply specific colors to Heatmaps and Clusters, we must create 
            # separate layers for each category within the file.
            cat_groups = {}
            for p in points:
                cat = p['cat']
                if cat not in cat_groups:
                    cat_groups[cat] = {'locs': [], 'color': p['color']}
                cat_groups[cat]['locs'].append(p['loc'])


            # ==========================================
            # --- RENDER ENGINE ---
            # ==========================================
            if render_mode == "Heatmap":
                for cat, group in cat_groups.items():
                    c_color = group['color']
                    # Use the first point's opacity for this category group
                    c_opacity = group['locs_and_ops'][0]['opacity'] if 'locs_and_ops' in group else 0.7 
                    
                    # FEATURE 2: Heatmap Opacity
                    # The maximum intensity (1.0) now caps out at the user's chosen opacity!
                    custom_gradient = {
                        0.0: hex_to_rgba(c_color, 0.0), 
                        0.5: hex_to_rgba(c_color, c_opacity * 0.5), 
                        1.0: hex_to_rgba(c_color, c_opacity) 
                    }
                    
                    HeatMap(
                        group['locs'], 
                        name=f"{fname} - {cat}", # Cleaned up name for the Layer Control
                        radius=15, 
                        blur=10, 
                        gradient=custom_gradient
                    ).add_to(m)

            elif render_mode == "Marker Clusters":
                # 2. CLUSTER MODE: Custom CSS colored cluster bubbles per category
                for cat, group in cat_groups.items():
                    c_color = group['color']
                    
                    # Inject the Python color variable {c_color} into the JavaScript
                    custom_icon_callback = f"""
                        function (cluster) {{
                            return L.divIcon({{
                                html: '<div style="background-color: white; border: 3px solid {c_color}; color: {c_color}; font-weight: bold; border-radius: 50%; text-align: center; line-height: 30px;">' + cluster.getChildCount() + '</div>',
                                className: 'marker-cluster',
                                iconSize: L.point(30, 30)
                            }});
                        }}
                    """
                    
                    # Assign the custom javascript icon to this specific cluster group
                    mc = MarkerCluster(name=f"Cluster: {fname} ({cat})", icon_create_function=custom_icon_callback).add_to(m)
                    
                    for loc in group['locs']:
                        folium.CircleMarker(
                            location=loc,
                            radius=5,
                            color=c_color,
                            fill=True,
                            fill_color=c_color,
                            fill_opacity=0.7,
                            tooltip=f"File: {fname}<br>Category: {cat}"
                        ).add_to(mc)

            else:
                # 3. SCATTER MODE: Classic individual dots
                for p in points:
                    folium.CircleMarker(
                        location=p['loc'], 
                        radius=4, 
                        color=p['color'],
                        fill=True,
                        fill_color=p['color'],
                        fill_opacity=0.7,
                        tooltip=f"File: {fname}<br>Category: {p['cat']}"
                    ).add_to(m)


        # ==========================================
        # --- FEATURE 3: THE DYNAMIC LEGEND ---
        # ==========================================
        # Construct the HTML for the legend based on the colors we discovered
        legend_html = '''
        <div style="position: absolute; bottom: 50px; right: 50px; width: auto; 
                    z-index: 9999; background-color: rgba(255, 255, 255, 0.9); 
                    border-radius: 8px; padding: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
            <h4 style="margin-top:0; margin-bottom:10px;">Legend</h4>
        '''
        for cat, color in legend_items.items():
            legend_html += f'<div><span style="background-color: {color}; width: 15px; height: 15px; display: inline-block; border-radius: 50%; margin-right: 8px;"></span>{cat}</div>'
        legend_html += '</div>'
        
        m.get_root().html.add_child(folium.Element(legend_html))

        # ==========================================
        # --- FEATURE 4: LAYER CONTROL (Z-Index / Toggle) ---
        # ==========================================
        # This adds the toggle menu to the top right corner.
        # It automatically detects every HeatMap, MarkerCluster, or FeatureGroup added with a `name=` attribute!
        folium.LayerControl(collapsed=False).add_to(m)

        # Auto-center logic
        if all_lats:
            m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

        # Render the map
        st_folium(m, width=1200, height=600, returned_objects=[])

        # ==========================================
        # --- FEATURE 5: DOWNLOAD OPTIMAL INTERACTIVE MAP ---
        # ==========================================
        # Extract the raw HTML string of the fully built map
        map_html = m.get_root().render()
        
        # Provide a Streamlit download button
        st.download_button(
            label="📥 Download Interactive Map (HTML)",
            data=map_html,
            file_name="ultra_explorer_map.html",
            mime="text/html",
            help="Downloads the map as an interactive webpage. You can open it in any browser!"
        )

else:
    st.info("👋 Welcome! Please upload one or more CSV/Excel files to start exploring.")