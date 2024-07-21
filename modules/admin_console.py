import streamlit as st
import pandas as pd
from utils.db import (
    fetch_all_users, fetch_user_access, reset_user_password, 
    toggle_user_status, update_user_info, update_user_access, 
    generate_random_password, create_user
)
import re


def get_user_data():
    users = fetch_all_users()
    user_data = []
    for user in users:
        access = fetch_user_access(user['username'])
        user_data.append({
            'Username': user['username'],
            'Email': user['email'],
            'Name': user['name'],
            'Is Admin': user['is_admin'],
            'Is Active': user['is_active'],
            'Page Access': ', '.join(access),
            'user_id': user['user_id']  # Hidden column for internal use
        })
    return pd.DataFrame(user_data)

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def is_valid_password(password):
    # At least 8 characters, 1 uppercase, 1 lowercase, 1 digit, 1 special character
    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    return re.match(pattern, password) is not None


def main(authentication_status):
    if not authentication_status:
        st.warning("Please login to access this page.")
        return
    elif not st.session_state['is_admin']:
        st.error("You do not have permission to access this page.")
        st.stop()

    st.title("Admin Console")

    # Display user table
    st.subheader("User Table")
    user_table = st.empty()
    df = get_user_data()
    user_table.dataframe(df.drop(columns=['user_id']), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.header("User Management")
    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for different actions
    tab1, tab2 = st.tabs(["Manage Existing Users", "Add New User"])

    with tab1:
        selected_user = st.selectbox("Select a user", df['Username'])
        action = st.radio("Choose an action", ["Reset Password", "Toggle Active Status", "Edit User Information"])

        if action in ["Reset Password", "Toggle Active Status"]:
            if st.button("Perform Action"):
                user_id = df[df['Username'] == selected_user]['user_id'].iloc[0]
                
                if action == "Reset Password":
                    new_password = generate_random_password()
                    if reset_user_password(user_id, new_password):
                        st.success(f"Password reset successful. New password: {new_password}")
                    else:
                        st.error("Failed to reset password.")

                elif action == "Toggle Active Status":
                    new_status = toggle_user_status(user_id)
                    if new_status is not None:
                        st.success(f"User status updated. New status: {'Active' if new_status else 'Inactive'}")
                    else:
                        st.error("Failed to update user status.")
                
                # Update user table
                df = get_user_data()
                user_table.dataframe(df.drop(columns=['user_id']), use_container_width=True, hide_index=True)

        elif action == "Edit User Information":
            user_id = df[df['Username'] == selected_user]['user_id'].iloc[0]
            user_info = df[df['Username'] == selected_user].iloc[0]

            with st.form("edit_user_form"):
                st.subheader("Edit User Information")
                new_username = st.text_input("Username", user_info['Username'])
                new_email = st.text_input("Email", user_info['Email'])
                new_name = st.text_input("Name", user_info['Name'])
                new_is_admin = st.checkbox("Is Admin", user_info['Is Admin'])
                
                all_pages = ["GA Optimizer", "Data Preparation", "Dashboard", "Model Explorer", "Real-time Analytics"]
                new_access = st.multiselect("Page Access", all_pages, default=user_info['Page Access'].split(', '))
                
                submit_button = st.form_submit_button("Update User")

            if submit_button:
                updated_info = {
                    'username': new_username,
                    'email': new_email,
                    'name': new_name,
                    'is_admin': new_is_admin
                }
                if update_user_info(user_id, updated_info) and update_user_access(user_id, new_access):
                    st.success("User information updated successfully.")
                    # Update user table
                    df = get_user_data()
                    user_table.dataframe(df.drop(columns=['user_id']), use_container_width=True, hide_index=True)
                    # Reset the form
                    st.rerun()
                else:
                    st.error("Failed to update user information.")

    with tab2:
        st.subheader("Add New User")
        with st.form("add_user_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_name = st.text_input("Name")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            new_is_admin = st.checkbox("Is Admin")
            
            all_pages = ["GA Optimizer", "Data Preparation", "Dashboard", "Model Explorer","Real-time Analytics"]
            new_access = st.multiselect("Page Access", all_pages)
            
            submit_button = st.form_submit_button("Add User")

        if submit_button:
            error_messages = []

            # Check if all fields are filled
            if not all([new_username, new_email, new_name, new_password, confirm_password]):
                error_messages.append("All fields are required.")

            # Check if username already exists
            existing_users = get_user_data()
            if new_username in existing_users['Username'].values:
                error_messages.append("Username already exists.")

            # Check if email already exists
            if new_email in existing_users['Email'].values:
                error_messages.append("Email already exists.")

            # Validate email format
            if not is_valid_email(new_email):
                error_messages.append("Invalid email format.")

            # Check if passwords match
            if new_password != confirm_password:
                error_messages.append("Passwords do not match.")

            # Validate password strength
            if not is_valid_password(new_password):
                error_messages.append("Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one digit, and one special character.")

            # Check if at least one page access is selected
            if not new_access:
                error_messages.append("Please select at least one page access.")

            if error_messages:
                for message in error_messages:
                    st.error(message)
            else:
                if create_user(new_username, new_email, new_name, new_password, new_is_admin, new_access):
                    st.success("User created successfully.")
                    # Update user table
                    df = get_user_data()
                    user_table.dataframe(df.drop(columns=['user_id']), use_container_width=True, hide_index=True)
                    # Reset the form
                    st.rerun()
                else:
                    st.error("Failed to create user.")

if __name__ == "__main__":
    main()