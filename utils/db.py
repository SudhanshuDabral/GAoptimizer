import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from streamlit_authenticator.utilities.hasher import Hasher
import uuid
import streamlit as st
import json

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


# function to insert well information and data for modeling into the database
def call_insert_or_update_well_and_consolidated_output(well_details, data, user_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        data_json = json.dumps(data)


        query = """
        SELECT insert_or_update_well_and_consolidated_output(%s, %s, %s, %s, %s, %s, %s, %s) AS result
        """
        cur.execute(query, (
            well_details["Well Name"],
            well_details["Well API"],
            well_details["Latitude"],
            well_details["Longitude"],
            well_details["TVD (ft)"],
            well_details["Reservoir Pressure"],
            user_id,
            data_json
        ))

        # Fetch the raw result to see what is returned
        raw_result = cur.fetchone()
        # print(f"Raw result from stored procedure: {raw_result}")

        if raw_result and 'result' in raw_result:
            result_str = raw_result['result']
            # print(f"Result string: {result_str}")
            conn.commit()
            cur.close()
            conn.close()

            # Extract the JSON part from the result string
            json_start = result_str.find('[')
            json_end = result_str.rfind(']') + 1
            json_str = result_str[json_start:json_end]
           

            # Parse the JSON result string into a Python dictionary
            json_str = json_str.replace('""', '"')  # Handle extra double quotes
            result = json.loads(json_str)
            return result
        else:
            # print("No result or 'result' key not found in raw_result")
            conn.commit()
            cur.close()
            conn.close()
            return None

    except Exception as error:
        print(f"Error calling stored procedure: {error}")
        return None
    
# Function to bulk insert arrays per stage data into the database
def call_insert_arrays_data(data_id, arrays_data, user_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # # Delete existing records for the given data_id
        cur.execute("DELETE FROM arrays WHERE data_id = %s", (data_id,))

        # Convert arrays data to list of tuples for bulk insert
        arrays_data_tuples = []
        for row in arrays_data:
            arrays_data_tuples.append((
                data_id,
                row['SlrWin'],
                row['PmaxminWin'],
                row['DownholeWinProp'],
                user_id
            ))

        # Perform bulk insert
        psycopg2.extras.execute_batch(cur, 
            """
            INSERT INTO arrays (data_id, slr_win, pmaxmin_win, downhole_win_prop, updated_by)
            VALUES (%s, %s, %s, %s, %s)
            """, 
            arrays_data_tuples)

        conn.commit()
        cur.close()
        conn.close()
        return {'status': 'success', 'records_added': len(arrays_data_tuples)}
    except Exception as error:
        print(f"Error performing bulk insert: {error}")
        return {'status': 'error', 'message': str(error)}

#fuction to fetch well details
def get_well_details():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT well_id, well_name, well_api, latitude, longitude, tvd, reservoir_pressure FROM well_master")
        well_details = cur.fetchall()
        cur.close()
        conn.close()
        return well_details
    except Exception as error:
        print(f"Error fetching well details: {error}")
        return []

# Function to fetch data for modeling
def get_modeling_data(well_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT tee, median_dhpm, median_dp, downhole_ppm, total_dhppm, total_slurry_dp, median_slurry, stage
            FROM data_for_modeling
            WHERE well_id = %s
            ORDER BY stage ASC
        """, (well_id,))
        modeling_data = cur.fetchall()
        cur.close()
        conn.close()
        return modeling_data
    except Exception as error:
        print(f"Error fetching modeling data: {error}")
        return []

# function to get stages for a well    
def get_well_stages(well_id):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT DISTINCT stage
        FROM data_for_modeling
        WHERE well_id = %s
        ORDER BY stage
    """, (well_id,))
    stages = [row['stage'] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return stages

#function to get array data for given well and stage
def get_array_data(well_id, stage):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT a.slr_win, a.pmaxmin_win, a.downhole_win_prop
        FROM arrays a
        JOIN data_for_modeling d ON a.data_id = d.data_id
        WHERE d.well_id = %s AND d.stage = %s
    """, (well_id, stage))
    array_data = cur.fetchall()  # Fetch all rows instead of just one
    cur.close()
    conn.close()
    return array_data