import streamlit as st

def main():
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        st.warning("Please login to access this page.")
        st.stop()
        
    st.title("admin console")

    # Add your dashboard components here
    st.write("This is the dashboard page.")
