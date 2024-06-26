# styles.py
import streamlit as st
import base64
from pathlib import Path

def load_css():
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    css_file = current_dir / "static" / "main.css"
    with open(css_file) as f:
        return f'<style>{f.read()}</style>'

def set_page_container_style():
    bg_path = Path(__file__).parent / "images" / "evolv_ai_background.png"
    with open(bg_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(31, 31, 31, 0.8), rgba(31, 31, 31, 0.8)), url(data:image/png;base64,{encoded_string});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def display_header():
    logo_path = Path(__file__).parent / "images" / "evolv_ai_logo.png"
    
    header_html = f"""
    <div class="header">
        <div class="logo-container">
            <img src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}" alt="EvolvAI Logo" class="logo-image">
            <div class="logo-text">EvolvAI</div>
        </div>
        <div class="tagline">Evolving Intelligence, Optimizing Solutions</div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)