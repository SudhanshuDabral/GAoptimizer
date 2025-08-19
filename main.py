import streamlit as st
import streamlit_authenticator as stauth
from utils.db import fetch_all_users, fetch_user_access
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from streamlit_option_menu import option_menu

# Disable Pillow debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

# Set up Streamlit page config
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
favicon_path = current_dir / "static" / "images" / "evolv_ai_favicon.png"

st.set_page_config(
    page_title="EvolvAI",
    page_icon=str(favicon_path),
    layout="wide"
)

# Import all page modules
import modules.ga_main as ga_main
import modules.data_prep as data_prep
import modules.dashboard as dashboard
import modules.admin_console as admin_console
import modules.model_explorer as model_explorer
import modules.realtime_analytics as realtime_analytics  # Import the new module
from static.styles import load_css, set_page_container_style, display_header

# Set up logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, 'app_logs.log')

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5),
                        ])

logger = logging.getLogger(__name__)

def log_message(level, message, exc_info=False):
    logger.log(level, f"[Evolv_main] {message}", exc_info=exc_info)

@st.cache_resource
def load_static_resources():
    # Load custom CSS
    css = load_css()
    # Set background image
    bg_style = set_page_container_style()
    return css, bg_style

def initialize_state():
    if 'state_initialized' not in st.session_state:
        default_states = {
            'authentication_status': None,
            'username': None,
            'name': None,
            'user_id': None,
            'is_admin': False,
            'is_active': False,
            'page': "GA Optimizer",
            'access': [],
            'current_page': None
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        st.session_state['state_initialized'] = True
        log_message(logging.DEBUG, "Session state initialized")

@st.cache_resource
def load_config():
    try:
        users = fetch_all_users()
        if users is None:
            log_message(logging.ERROR, "Failed to fetch user credentials from the database.")
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

        log_message(logging.INFO, "Config loaded successfully")
        return config
    except Exception as e:
        log_message(logging.ERROR, f"Failed to load config: {str(e)}")
        return None

def login():
    st.markdown("<h1 style='text-align: center;'>Analysis made easy</h1>", unsafe_allow_html=True)

    name, authentication_status, username = authenticator.login(
        fields={'Form name': 'Login'},
        location='main'
    )

    if authentication_status:
        log_message(logging.INFO, f"User logged in successfully: {username}")
        user_data = config['credentials']['usernames'][username]
        st.session_state.update({
            'authentication_status': authentication_status,
            'username': username,
            'name': name,
            'user_id': user_data['id'],
            'is_admin': user_data['is_admin'],
            'is_active': user_data['is_active'],
            'access': fetch_user_access(username)
        })
        st.rerun()
    elif authentication_status == False:
        log_message(logging.WARNING, "Failed login attempt")
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

def check_user_access(required_access):
    if required_access in st.session_state.get('access', []):
        return True
    log_message(logging.WARNING, f"Unauthorized access attempt to {required_access}")
    st.warning("You do not have permission to access this page.")
    return False

def navigation():
    with st.sidebar:
        st.markdown('<h1 style="text-align: center;">Navigation</h1>', unsafe_allow_html=True)

        nav_options = ["GA Optimizer", "Data Preparation", "Dashboard", "Model Explorer", "Real-time Analytics"]  # Added new option
        icons = ["diagram-3", "database-fill-check", "file-bar-graph", "search", "graph-up"]  # Added new icon
        
        if st.session_state['is_admin']:
            nav_options.append("Admin Console")
            icons.append("people-fill")

        nav_choice = option_menu(
            menu_title=None,
            options=nav_options,
            icons=icons,
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#2D2D2D"},
                "icon": {"color": "#00CED1", "font-size": "25px"},  # Distinct cyan color for icons
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "left", 
                    "margin": "0px", 
                    "--hover-color": "#3D3D3D",
                    "color": "#F5F5F5"
                },
                "nav-link-selected": {
                    "background-color": "#CC5500",
                    "color": "#FFFFFF",
                },
            }
        )

        if nav_choice != st.session_state.get('current_page'):
            log_message(logging.INFO, f"Navigation changed from {st.session_state.get('current_page')} to {nav_choice}")
            if st.session_state.get('current_page') == "GA Optimizer":
                clear_ga_optimizer_state()
            elif st.session_state.get('current_page') == "Data Preparation":
                clear_data_prep_state()
            
            st.session_state['current_page'] = nav_choice

        st.markdown("<br>" * 3, unsafe_allow_html=True)
        st.markdown('<hr style="margin: 20px 0;">', unsafe_allow_html=True)

        if authenticator.logout(button_name='Logout', location='main'):
            log_message(logging.INFO, "User logged out")
            st.rerun()

    try:
        with st.spinner("Loading page..."):
            if nav_choice == "GA Optimizer" and check_user_access("GA Optimizer"):
                ga_main.main(st.session_state["authentication_status"])
            elif nav_choice == "Data Preparation" and check_user_access("Data Preparation"):
                data_prep.main(st.session_state["authentication_status"])
            elif nav_choice == "Dashboard" and check_user_access("Dashboard"):
                dashboard.main(st.session_state["authentication_status"])
            elif nav_choice == "Model Explorer" and check_user_access("Model Explorer"):
                model_explorer.main(st.session_state["authentication_status"])
            elif nav_choice == "Real-time Analytics" and check_user_access("Real-time Analytics"):  # Add this condition
                realtime_analytics.main(st.session_state["authentication_status"])
            elif nav_choice == "Admin Console" and st.session_state['is_admin']:
                admin_console.main(st.session_state["authentication_status"])
    except Exception as e:
        log_message(logging.ERROR, f"Error in navigation: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please try refreshing the page or contact support if the issue persists.")

        

def clear_ga_optimizer_state():
    if 'ga_optimizer' in st.session_state:
        log_message(logging.INFO, "Clearing GA Optimizer state")
        st.session_state.ga_optimizer = {
            'running': False,
            'results': [],
            'edited_df': None,
            'zscored_df': None,
            'excluded_rows': [],
            'show_zscore': False,
            'regression_type': 'FPR',
            'monotonicity_results': {},
            'df_statistics': None,
            'zscored_statistics': None,
            'show_monotonicity': False,
            'r2_threshold': 0.55,
            'prob_crossover': 0.8,
            'prob_mutation': 0.2,
            'num_generations': 40,
            'population_size': 50,
        }

def clear_data_prep_state():
    if 'data_prep' in st.session_state:
        log_message(logging.INFO, "Clearing Data Preparation state")
        st.session_state.data_prep = {
            'well_details': None,
            'processing_files': False,
            'consolidated_output': None
        }

def main():
    try:
        css, bg_style = load_static_resources()
        st.markdown(css, unsafe_allow_html=True)
        st.markdown(bg_style, unsafe_allow_html=True)
        display_header()
        
        initialize_state()

        if st.session_state["authentication_status"]:
            navigation()
        else:
            login()

    except Exception as e:
        log_message(logging.ERROR, f"Error in main function: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please try refreshing the page or contact support if the issue persists.")

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

if __name__ == "__main__":
    main()