import streamlit as st
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="My Streamlit App", page_icon="🚀", layout="wide")

# --- HELPER FUNCTION FOR TUTORIALS ---
def render_tutorial(page_name):
    """
    Placeholder function for rendering markdown tutorials.
    I've put it inside an expander to keep the UI clean, but you can change this!
    """
    with st.expander(f"📖 How to use the {page_name} page", expanded=False):
        # TODO: Later, you can replace the dummy text with code to read an .md file
        # Example:
        # file_path = f"tutorials/{page_name.lower()}.md"
        # if os.path.exists(file_path):
        #     with open(file_path, "r") as file:
        #         st.markdown(file.read())
        # else:
        #     st.warning("Tutorial not found.")
        
        st.markdown(f"**Placeholder:** Markdown tutorial for the `{page_name}` page will go here.")

# --- PAGE FUNCTIONS ---
def home_page():
    st.title("🏠 Home")
    render_tutorial("Home")
    
    st.write("Welcome to the main dashboard! Build out your main app features here.")

def analytics_page():
    st.title("📊 Analytics")
    render_tutorial("Analytics")
    
    st.write("Data visualizations, charts, and metrics will go here.")

def settings_page():
    st.title("⚙️ Settings")
    render_tutorial("Settings")
    
    st.write("App configuration options and user preferences.")

# --- MAIN NAVIGATION ---
def main():
    st.sidebar.title("Navigation")
    
    # Dictionary routing to the page functions
    pages = {
        "Home": home_page,
        "Analytics": analytics_page,
        "Settings": settings_page
    }
    
    # Sidebar selection
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Execute the selected page function
    pages[selection]()

if __name__ == "__main__":
    main()