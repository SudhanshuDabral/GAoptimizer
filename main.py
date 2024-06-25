import streamlit as st
import streamlit_authenticator as stauth
from utils.db import fetch_all_users, fetch_user_access
from jinja2 import Template
import logging


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set the page configuration
st.set_page_config(page_title="My Streamlit App", layout="wide")

# Function to load and render HTML template
def load_template(template_name):
    try:
        with open(template_name, 'r') as file:
            template = Template(file.read())
        return template
    except Exception as e:
        logger.error(f"Failed to load template {template_name}: {str(e)}")
        return None

def initialize_state():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = ""
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
    if 'name' not in st.session_state:
        st.session_state['name'] = ""
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
                'key': st.secrets.get("cookie_key", "default_secret_key"),  # Use secret from config.toml
                'name': 'streamlit_auth'
            },
            'preauthorized': {
                'emails': []
            }
        }

        return config
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        return None
    
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
    if template:
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
        st.session_state['authenticated'] = True
        st.session_state['username'] = username
        st.session_state['user_id'] = user_data['id']
        st.session_state['name'] = user_data['name']
        st.session_state['is_admin'] = user_data['is_admin']
        st.session_state['is_active'] = user_data['is_active']
        st.session_state['access'] = fetch_user_access(username)
        st.rerun()

    elif authentication_status is False:
        st.error('Username/password is incorrect')
    elif authentication_status is None:
        st.warning('Please enter your username and password')

def add_vertical_space(num_lines=1):
    for _ in range(num_lines):
        st.sidebar.write("\n")

logger = logging.getLogger(__name__)

# Custom CSS for styling (simplified)
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .sidebar-nav {
        padding: 10px;
    }
    .sidebar-nav h1 {
        color: #262730;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .stRadio > div[role='radiogroup'] > label {
        color: #262730;
        font-size: 16px;
        padding: 10px 5px;
    }
</style>
""", unsafe_allow_html=True)

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

        # Clear page-specific session variables when navigating
        if nav_choice != st.session_state.get('current_page'):
            if st.session_state.get('current_page') == "GA Optimizer":
                clear_ga_optimizer_state()
            elif st.session_state.get('current_page') == "Data Preparation":
                clear_data_prep_state()
            # Add similar conditions for other pages if they have specific session variables
            
            # Update the current page
            st.session_state['current_page'] = nav_choice

        # Add vertical space
        st.markdown("<br>" * 5, unsafe_allow_html=True)

        # Add a horizontal line for visual separation
        st.markdown('<hr style="margin: 20px 0;">', unsafe_allow_html=True)

        # Add the logout button at the bottom
        if st.button("Logout", key="unique_logout_button", type="primary", use_container_width=True):
            logout()

        st.markdown('</div>', unsafe_allow_html=True)

    try:
        # Clear Streamlit cache
        st.cache_data.clear()
        st.cache_resource.clear()

        logger.debug(f"Navigation choice: {nav_choice}")
        
        if nav_choice == "GA Optimizer":
            logger.debug("Entering GA Optimizer section")
            if "GA Optimizer" in st.session_state['access']:
                import modules.ga_main as ga_main
                ga_main.main()
            else:
                st.warning("You do not have permission to access this page.")
        elif nav_choice == "Data Preparation":
            logger.debug("Entering Data Preparation section")
            if "Data Preparation" in st.session_state['access']:
                import modules.data_prep as data_prep
                data_prep.main()
            else:
                st.warning("You do not have permission to access this page.")
        elif nav_choice == "Dashboard":
            logger.debug("Entering Dashboard section")
            if "Dashboard" in st.session_state['access']:
                import modules.dashboard as dashboard
                dashboard.main()
            else:
                st.warning("You do not have permission to access this page.")
        elif nav_choice == "Admin Console":
            logger.debug("Entering Admin Console section")
            if st.session_state['is_admin']:
                import modules.admin_console as admin_console
                admin_console.main()
            else:
                st.warning("You do not have permission to access this page.")
    except Exception as e:
        logger.error(f"Error in navigation: {str(e)}", exc_info=True)
        st.error(f"An error occurred while loading the page: {str(e)}")
        
def logout():
    logger.debug("Logout function called")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_state()
    st.rerun()

#clear session state for GA optimizer
def clear_ga_optimizer_state():
    if 'ga_optimizer' in st.session_state:
        st.session_state.ga_optimizer = {
            'running': False,
            'results': [],
            'edited_df': None,
            'zscored_df': None,
            'excluded_rows': [],
            'show_zscore_tab': False,
            'regression_type': 'FPR',
            'monotonicity_results': {},
            'df_statistics': None
        }

#clear session state for data preparation
def clear_data_prep_state():
    if 'data_prep' in st.session_state:
        st.session_state.data_prep = {
            'well_details': None,
            'processing_files': False,
            'consolidated_output': None
        }

def main():
    try:
        initialize_state()
        logger.debug(f"Session state after initialization: {st.session_state}")

        if st.session_state['authenticated']:
            logger.debug("User is authenticated, entering navigation")
            navigation()
        else:
            logger.debug("User is not authenticated, showing login page")
            login_page()
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()