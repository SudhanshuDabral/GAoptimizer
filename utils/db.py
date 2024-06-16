import psycopg2
from psycopg2.extras import RealDictCursor
from streamlit_authenticator.utilities.hasher import Hasher
import uuid
import streamlit as st

def get_db_connection():
    return psycopg2.connect(
        host=st.secrets["database"]["host"],
        database=st.secrets["database"]["dbname"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        port=st.secrets["database"]["port"]
    )

def fetch_all_users():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM user_master")
    users = cur.fetchall()
    cur.close()
    conn.close()
    return users

def fetch_user_access(username):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT page_name 
        FROM access_control ac 
        JOIN user_master um ON ac.user_id = um.user_id 
        WHERE um.username = %s
    """, (username,))
    access = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return access

# Function to hash password using streamlit_authenticator.utilities.Hasher
def hash_password(password):
    hasher = Hasher([password])
    hashed_passwords = hasher.generate()
    return hashed_passwords[0] if hashed_passwords else None