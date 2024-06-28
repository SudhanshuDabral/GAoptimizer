import streamlit as st
import streamlit_authenticator as stauth
from utils.db import fetch_all_users, fetch_user_access
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

# Import all page modules at the top
import modules.ga_main as ga_main
import modules.data_prep as data_prep
import modules.dashboard as dashboard
import modules.admin_console as admin_console
from static.styles import load_css, set_page_container_style, display_header

# Set up logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, 'app_logs.log')

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5),
                    ])

logger = logging.getLogger(__name__)

# Get the absolute path to your favicon file
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
favicon_path = current_dir / "static" / "images" / "evolv_ai_favicon.png"

# Set page config with favicon
st.set_page_config(
    page_title="EvolvAI",
    page_icon=str(favicon_path),
    layout="wide"
)

# Load custom CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Set background image
set_page_container_style()

# Display header
display_header()

def initialize_state():
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'name' not in st.session_state:
        st.session_state['name'] = None
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
    if 'is_admin' not in st.session_state:
        st.session_state['is_admin'] = False
    if 'is_active' not in st.session_state:
        st.session_state['is_active'] = False
    if 'page' not in st.session_state:
        st.session_state['page'] = "GA Optimizer"
    if 'access' not in st.session_state:
        st.session_state['access'] = []
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = None

@st.cache_resource
def load_config():
    try:
        users = fetch_all_users()
        if users is None:
            logger.error("Failed to fetch user credentials from the database.")
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
                'key': st.secrets.get("cookie_key", "default_secret_key"),
                'name': 'streamlit_auth'
            },
            'preauthorized': {
                'emails': []
            }
        }

        logger.info("Config loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        return None

# Build config
config = load_config()
if config is None:
    st.stop()

# Create the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

def login():
    st.markdown("<h1 style='text-align: center;'>Analysis made easy</h1>", unsafe_allow_html=True)

    name, authentication_status, username = authenticator.login(
        fields={
            "form_name": "Login",
            "username": "Username",
            "password": "Password",
            "login_button": "Login"
        },
        location='main'
    )

    if authentication_status:
        logger.info(f"User logged in successfully: {username}")
        user_data = config['credentials']['usernames'][username]
        st.session_state['authentication_status'] = authentication_status
        st.session_state['username'] = username
        st.session_state['name'] = name
        st.session_state['user_id'] = user_data['id']
        st.session_state['is_admin'] = user_data['is_admin']
        st.session_state['is_active'] = user_data['is_active']
        st.session_state['access'] = fetch_user_access(username)
        st.rerun()
    elif authentication_status == False:
        logger.warning("Failed login attempt")
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

def navigation():
    with st.sidebar:
        st.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
        st.markdown('<h1>Navigation</h1>', unsafe_allow_html=True)

        nav_options = [
            "GA Optimizer",
            "Data Preparation",
            "Dashboard"
        ]
        if st.session_state['is_admin']:
            nav_options.append("Admin Console")

        nav_choice = st.radio("Select a page", nav_options, key="nav_radio")

        if nav_choice != st.session_state.get('current_page'):
            logger.info(f"Navigation changed from {st.session_state.get('current_page')} to {nav_choice}")
            if st.session_state.get('current_page') == "GA Optimizer":
                clear_ga_optimizer_state()
            elif st.session_state.get('current_page') == "Data Preparation":
                clear_data_prep_state()
            
            st.session_state['current_page'] = nav_choice

        st.markdown("<br>" * 5, unsafe_allow_html=True)
        st.markdown('<hr style="margin: 20px 0;">', unsafe_allow_html=True)

        if authenticator.logout('Logout', 'main'):
            logger.info("User logged out")
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    try:
        with st.spinner("Loading page..."):
            if nav_choice == "GA Optimizer":
                if "GA Optimizer" in st.session_state['access']:
                    ga_main.main(st.session_state["authentication_status"])
                else:
                    logger.warning("Unauthorized access attempt to GA Optimizer")
                    st.warning("You do not have permission to access this page.")
            elif nav_choice == "Data Preparation":
                if "Data Preparation" in st.session_state['access']:
                    data_prep.main(st.session_state["authentication_status"])
                else:
                    logger.warning("Unauthorized access attempt to Data Preparation")
                    st.warning("You do not have permission to access this page.")
            elif nav_choice == "Dashboard":
                if "Dashboard" in st.session_state['access']:
                    dashboard.main(st.session_state["authentication_status"])
                else:
                    logger.warning("Unauthorized access attempt to Dashboard")
                    st.warning("You do not have permission to access this page.")
            elif nav_choice == "Admin Console":
                if st.session_state['is_admin']:
                    admin_console.main(st.session_state["authentication_status"])
                else:
                    logger.warning("Unauthorized access attempt to Admin Console")
                    st.warning("You do not have permission to access this page.")
    except Exception as e:
        logger.error(f"Error in navigation: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please try refreshing the page or contact support if the issue persists.")

def clear_ga_optimizer_state():
    if 'ga_optimizer' in st.session_state:
        logger.info("Clearing GA Optimizer state")
        st.session_state.ga_optimizer = {
            'running': False,
            'results': [],
            'edited_df': None,
            'zscored_df': None,
            'excluded_rows': [],
            'show_zscore_tab': False,
            'regression_type': 'FPR',
            'monotonicity_results': {},
            'df_statistics': None,
            'show_monotonicity': False,
            'selected_wells': [],
            'r2_threshold': 0.55,
            'prob_crossover': 0.8,
            'prob_mutation': 0.2,
            'num_generations': 40,
            'population_size': 50
        }

def clear_data_prep_state():
    if 'data_prep' in st.session_state:
        logger.info("Clearing Data Preparation state")
        st.session_state.data_prep = {
            'well_details': None,
            'processing_files': False,
            'consolidated_output': None
        }

def main():
    try:
        initialize_state()
        logger.debug("Session state initialized")

        if st.session_state["authentication_status"]:
            navigation()
        else:
            login()

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()