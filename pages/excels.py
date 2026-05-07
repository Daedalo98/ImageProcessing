import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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

    plot_height = st.slider("Plot Height", 400, 2000, 600)
    font_size = st.number_input("Global Font Size", 5, 50, 15)
    legend_pos = st.radio("Legend Position", ["v", "h"], index=0)
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
                
                # Auto-date conversion
                for col in temp_df.columns:
                    if temp_df[col].dtype == 'object':
                        try:
                            temp_df[col] = pd.to_datetime(temp_df[col])
                        except: pass
                
                # Initialize default config for this file
                st.session_state.file_configs[file.name] = {
                    "df": temp_df,
                    "active": True
                }
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
                    config["active"] = st.checkbox("Include in Plot", value=True, key=f"act_{fname}")
                    x_col = st.selectbox(f"X-Axis (Time/Group)", local_df.columns, key=f"x_{fname}")
                    
                    # ADDITION: Prepend "All" to the list of columns
                    breakdown_options = ["All"] + list(local_df.columns)
                    break_col = st.selectbox(f"Breakdown Column", breakdown_options, key=f"brk_{fname}")

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
                                    "size": st.slider("Size/Width", 1, 20, 5, key=f"s_{fname}_{cat}")
                                }
                            with cc2:
                                config["styles"][cat].update({
                                    "symbol": st.selectbox("Dot Shape", ["circle", "square", "diamond", "cross"], key=f"sym_{fname}_{cat}"),
                                    "opacity": st.slider("Opacity", 0.0, 1.0, 0.7, key=f"op_{fname}_{cat}"),
                                    "cumulative": st.checkbox("Cumulative", value=False, key=f"cum_{fname}_{cat}")
                                })

    # 3. Dynamic Plotting
    st.divider()
    fig = go.Figure()

    # --- Inside the Plotting Section ---
    for fname, config in st.session_state.file_configs.items():
        if not config["active"]: continue
        
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
            
            # Retrieve the specific style for this category
            # Fallback to defaults if style isn't found
            style = config["styles"].get(cat, {"type": "Line", "color": "#636EFA", "size": 5, "symbol": "circle", "opacity": 0.7, "cumulative": False})
            
            y_vals = cat_df['Freq']
            if style["cumulative"]:
                y_vals = y_vals.cumsum()

            # Using specific parameters for each type
            if style["type"] == "Bar":
                fig.add_trace(go.Bar(
                    x=cat_df[config["x_col"]], y=y_vals, 
                    name=f"{fname}: {cat}",
                    marker=dict(color=style["color"], opacity=style["opacity"]),
                    # Use size for bar width (0.1 to 1.0 logic)
                    width=[(style["size"]/20) * 86400000] * len(cat_df) if config.get("is_date") else None 
                ))
            elif style["type"] == "Line":
                fig.add_trace(go.Scatter(
                    x=cat_df[config["x_col"]], y=y_vals, 
                    mode='lines+markers', 
                    name=f"{fname}: {cat}",
                    line=dict(color=style["color"], width=style["size"]),
                    marker=dict(symbol=style["symbol"], size=style["size"]+2, opacity=style["opacity"])
                ))
            elif style["type"] == "Area":
                fig.add_trace(go.Scatter(
                    x=cat_df[config["x_col"]], y=y_vals, 
                    mode='lines', fill='tozeroy', 
                    name=f"{fname}: {cat}",
                    line=dict(color=style["color"], width=style["size"]),
                    fillcolor=style["color"],
                    opacity=style["opacity"]
                ))
            elif style["type"] == "Scatter":
                fig.add_trace(go.Scatter(
                    x=cat_df[config["x_col"]], y=y_vals, 
                    mode='markers', 
                    name=f"{fname}: {cat}",
                    marker=dict(color=style["color"], size=style["size"]*2, symbol=style["symbol"], opacity=style["opacity"])
                ))
    # Create a generic or dynamic label
    # If you want to show the specific column name of the first active file:
    active_files = [c["x_col"] for c in st.session_state.file_configs.values() if c.get("active")]
    x_axis_label = active_files[0] if active_files else "X-Axis"

    # Global Layout Enforcement
    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=paper_color,
        height=plot_height,
        template="plotly_white",
        font=dict(size=font_size),
        legend=dict(orientation=legend_pos),
        barmode='group',
        xaxis=dict(
            type="date", # This forces the axis to be a linear timeline
            title=dict(text=x_axis_label, font=dict(size=font_size)), # Use the dynamic label here
            tickformat="%d-%m-%Y",  
            tickangle=-45,
            tickfont=dict(size=font_size)
        ),
        yaxis=dict(tickfont=dict(size=font_size), title="Counts / Cumulative Sum")
    )

    st.plotly_chart(fig, width='stretch')

else:
    st.info("👋 Welcome! Please upload one or more CSV/Excel files to start exploring.")