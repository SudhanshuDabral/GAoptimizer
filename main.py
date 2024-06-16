import streamlit as st
import streamlit_authenticator as stauth
import yaml
from utils.db import fetch_all_users, fetch_user_access
from jinja2 import Template

# Set the page configuration
st.set_page_config(page_title="My Streamlit App", layout="wide")

# Function to load and render HTML template
def load_template(template_name):
    with open(template_name, 'r') as file:
        template = Template(file.read())
    return template

def initialize_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'name' not in st.session_state:
        st.session_state.name = ""
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False
    if 'is_active' not in st.session_state:
        st.session_state.is_active = False
    if 'page' not in st.session_state:
        st.session_state.page = "GA Optimizer"

def load_config():
    users = fetch_all_users()
    if users is None:
        st.error("Failed to fetch user credentials from the database.")
        return None

    credentials = {'usernames': {}}
    for user in users:
        credentials['usernames'][user['username']] = {
            'id': str(user['user_id']),
            'name': user['name'],
            'password': user['password'],
            'email': user['email'],
            'is_admin': user['is_admin'],
            'is_active': user['is_active']
        }

    config = {
        'credentials': credentials,
        'cookie': {
            'expiry_days': 30,
            'key': 'some_signature_key',  # Replace with a secure key
            'name': 'streamlit_auth'
        },
        'preauthorized': {
            'emails': []
        }
    }

    # Save the config to a file in the root directory
    with open('config.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    return config

# Build config
config = load_config()
if config is None:
    st.stop()  # Stop if config could not be built

# Create the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

def login_page():
    template = load_template('static/template.html')
    st.markdown(template.render(), unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>Analysis made easy</h1>", unsafe_allow_html=True)

    # Normalize username input to lowercase
    name, authentication_status, username = authenticator.login(
        location="main",
        fields={
            "form_name": "Login",
            "username": "Username",
            "password": "Password",
            "login_button": "Login"
        }
    )

    if authentication_status:
        user_data = config['credentials']['usernames'][username]
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.user_id = user_data['id']
        st.session_state.name = user_data['name']
        st.session_state.is_admin = user_data['is_admin']
        st.session_state.is_active = user_data['is_active']
        st.session_state.access = fetch_user_access(username)
        st.rerun()

    elif authentication_status is False:
        st.error('Username/password is incorrect')
    elif authentication_status is None:
        st.warning('Please enter your username and password')

def navigation():
    nav_options = ["GA Optimizer", "Data Preparation", "Dashboard"]
    if st.session_state.is_admin:
        nav_options.append("Admin Console")

    nav_choice = st.sidebar.radio("Navigation", nav_options)

    if nav_choice == "GA Optimizer":
        if "GA Optimizer" in st.session_state.access:
            import modules.ga_main as ga_main
            ga_main.main()
        else:
            st.warning("You do not have permission to access this page.")
    elif nav_choice == "Data Preparation":
        if "Data Preparation" in st.session_state.access:
            import modules.data_prep as data_prep
            data_prep.main()
        else:
            st.warning("You do not have permission to access this page.")
    elif nav_choice == "Dashboard":
        if "Dashboard" in st.session_state.access:
            import modules.dashboard as dashboard
            dashboard.main()
        else:
            st.warning("You do not have permission to access this page.")
    elif nav_choice == "Admin Console":
        if st.session_state.is_admin:
            # Assuming an admin_console module exists
            import modules.admin_console as admin_console
            admin_console.main()
        else:
            st.warning("You do not have permission to access this page.")

def logout():
    stauth.Authenticate.logout("Logout", "sidebar")
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.name = ""
    st.session_state.is_admin = False
    st.session_state.is_active = False
    st.session_state.user_id = None
    st.rerun()

def main():
    initialize_state()

    if st.session_state.authenticated:
        navigation()
        st.sidebar.button("Logout", on_click=logout)
    else:
        login_page()

if __name__ == "__main__":
    main()
