import os
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from streamlit_authenticator.utilities.hasher import Hasher
import streamlit as st
import json
import string
import secrets
import logging
from logging.handlers import RotatingFileHandler
from psycopg2.extras import Json, register_uuid


# Set up logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, 'app_logs.log')

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5),
                        ])

logger = logging.getLogger(__name__)

def log_message(level, message):
    logger.log(level, f"[Database Ops] {message}")

def get_db_connection():
    log_message(logging.DEBUG, "Establishing database connection")
    return psycopg2.connect(
        host=st.secrets["database"]["host"],
        database=st.secrets["database"]["dbname"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        port=st.secrets["database"]["port"]
    )

def fetch_all_users():
    log_message(logging.INFO, "Fetching all users")
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM user_master")
    users = cur.fetchall()
    cur.close()
    conn.close()
    log_message(logging.INFO, f"Fetched {len(users)} users")
    return users

@st.cache_data(ttl=3600)
def fetch_user_access(username):
    log_message(logging.INFO, f"Fetching access for user: {username}")
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
    log_message(logging.INFO, f"Fetched {len(access)} access rights for user: {username}")
    return access

def hash_password(password):
    log_message(logging.DEBUG, "Hashing password")
    hasher = Hasher([password])
    hashed_passwords = hasher.generate()
    return hashed_passwords[0] if hashed_passwords else None

def call_insert_or_update_well_and_consolidated_output(well_details, data, user_id):
    log_message(logging.INFO, f"Inserting/updating well and consolidated output for well: {well_details['Well Name']}")
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

        raw_result = cur.fetchone()

        if raw_result and 'result' in raw_result:
            result_str = raw_result['result']
            conn.commit()
            cur.close()
            conn.close()

            json_start = result_str.find('[')
            json_end = result_str.rfind(']') + 1
            json_str = result_str[json_start:json_end]
           
            json_str = json_str.replace('""', '"')
            result = json.loads(json_str)
            log_message(logging.INFO, f"Successfully inserted/updated well: {well_details['Well Name']}")
            return result
        else:
            conn.commit()
            cur.close()
            conn.close()
            log_message(logging.WARNING, f"No result returned for well: {well_details['Well Name']}")
            return None

    except Exception as error:
        log_message(logging.ERROR, f"Error inserting/updating well {well_details['Well Name']}: {str(error)}")
        return None
    
def call_insert_arrays_data(data_id, arrays_data, user_id):
    log_message(logging.INFO, f"Inserting arrays data for data_id: {data_id}")
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("DELETE FROM arrays WHERE data_id = %s", (data_id,))

        arrays_data_tuples = []
        for row in arrays_data:
            arrays_data_tuples.append((
                data_id,
                row['SlrWin'],
                row['PmaxminWin'],
                row['DownholeWinProp'],
                user_id
            ))

        psycopg2.extras.execute_batch(cur, 
            """
            INSERT INTO arrays (data_id, slr_win, pmaxmin_win, downhole_win_prop, updated_by)
            VALUES (%s, %s, %s, %s, %s)
            """, 
            arrays_data_tuples)

        conn.commit()
        cur.close()
        conn.close()
        log_message(logging.INFO, f"Successfully inserted {len(arrays_data_tuples)} array records for data_id: {data_id}")
        return {'status': 'success', 'records_added': len(arrays_data_tuples)}
    except Exception as error:
        log_message(logging.ERROR, f"Error inserting arrays data for data_id {data_id}: {str(error)}")
        return {'status': 'error', 'message': str(error)}

@st.cache_data
def get_well_details():
    log_message(logging.INFO, "Fetching well details")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT well_id, well_name, well_api, latitude, longitude, tvd, reservoir_pressure FROM well_master")
        well_details = cur.fetchall()
        cur.close()
        conn.close()
        log_message(logging.INFO, f"Successfully fetched {len(well_details)} well details")
        return well_details
    except Exception as error:
        log_message(logging.ERROR, f"Error fetching well details: {str(error)}")
        return []
    
@st.cache_data
def get_modeling_data(well_id):
    log_message(logging.INFO, f"Fetching modeling data for well_id: {well_id}")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT data_id, well_id, tee, median_dhpm, median_dp, downhole_ppm, total_dhppm, total_slurry_dp, median_slurry, stage
            FROM data_for_modeling
            WHERE well_id = %s
            ORDER BY stage ASC
        """, (well_id,))
        modeling_data = cur.fetchall()
        cur.close()
        conn.close()
        log_message(logging.INFO, f"Fetched {len(modeling_data)} modeling data records for well_id: {well_id}")
        return modeling_data
    except Exception as error:
        log_message(logging.ERROR, f"Error fetching modeling data for well_id {well_id}: {str(error)}")
        return []

def get_well_stages(well_id):
    log_message(logging.INFO, f"Fetching stages for well_id: {well_id}")
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
    log_message(logging.INFO, f"Fetched {len(stages)} stages for well_id: {well_id}")
    return stages

@st.cache_data
def get_array_data(well_id, stage):
    log_message(logging.INFO, f"Fetching array data for well_id: {well_id}, stage: {stage}")
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT a.slr_win, a.pmaxmin_win, a.downhole_win_prop
        FROM arrays a
        JOIN data_for_modeling d ON a.data_id = d.data_id
        WHERE d.well_id = %s AND d.stage = %s
    """, (well_id, stage))
    array_data = cur.fetchall()
    cur.close()
    conn.close()
    log_message(logging.INFO, f"Fetched {len(array_data)} array data records for well_id: {well_id}, stage: {stage}")
    return array_data

def reset_user_password(user_id, new_password):
    log_message(logging.INFO, f"Resetting password for user_id: {user_id}")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        hashed_password = hash_password(new_password)
        cur.execute("""
            UPDATE user_master
            SET password = %s
            WHERE user_id = %s
        """, (hashed_password, user_id))
        conn.commit()
        cur.close()
        conn.close()
        log_message(logging.INFO, f"Successfully reset password for user_id: {user_id}")
        return True
    except Exception as error:
        log_message(logging.ERROR, f"Error resetting password for user_id {user_id}: {str(error)}")
        return False

def toggle_user_status(user_id):
    log_message(logging.INFO, f"Toggling status for user_id: {user_id}")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE user_master
            SET is_active = NOT is_active
            WHERE user_id = %s
            RETURNING is_active
        """, (user_id,))
        new_status = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        log_message(logging.INFO, f"Successfully toggled status for user_id: {user_id}. New status: {new_status}")
        return new_status
    except Exception as error:
        log_message(logging.ERROR, f"Error toggling status for user_id {user_id}: {str(error)}")
        return None

def update_user_info(user_id, updated_info):
    log_message(logging.INFO, f"Updating info for user_id: {user_id}")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE user_master
            SET username = %s, email = %s, name = %s, is_admin = %s
            WHERE user_id = %s
        """, (updated_info['username'], updated_info['email'], updated_info['name'], updated_info['is_admin'], user_id))
        conn.commit()
        cur.close()
        conn.close()
        log_message(logging.INFO, f"Successfully updated info for user_id: {user_id}")
        return True
    except Exception as error:
        log_message(logging.ERROR, f"Error updating info for user_id {user_id}: {str(error)}")
        return False

def update_user_access(user_id, access_list):
    log_message(logging.INFO, f"Updating access for user_id: {user_id}")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("DELETE FROM access_control WHERE user_id = %s", (user_id,))
        
        for page in access_list:
            cur.execute("""
                INSERT INTO access_control (user_id, page_name, updated_by)
                VALUES (%s, %s, %s)
            """, (user_id, page, user_id))
        
        conn.commit()
        cur.close()
        conn.close()
        log_message(logging.INFO, f"Successfully updated access for user_id: {user_id}")
        return True
    except Exception as error:
        log_message(logging.ERROR, f"Error updating access for user_id {user_id}: {str(error)}")
        return False

def generate_random_password(length=12):
    log_message(logging.DEBUG, f"Generating random password of length: {length}")
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for i in range(length))

def create_user(username, email, name, password, is_admin, access_list):
    log_message(logging.INFO, f"Creating new user: {username}")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        hashed_password = hash_password(password)
        
        cur.execute("""
            INSERT INTO user_master (username, email, name, password, is_admin)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING user_id
        """, (username, email, name, hashed_password, is_admin))
        
        user_id = cur.fetchone()[0]
        
        for page in access_list:
            cur.execute("""
                INSERT INTO access_control (user_id, page_name, updated_by)
                VALUES (%s, %s, %s)
            """, (user_id, page, user_id))
        
        conn.commit()
        cur.close()
        conn.close()
        log_message(logging.INFO, f"Successfully created user: {username}")
        return True
    except Exception as error:
        log_message(logging.ERROR, f"Error creating user {username}: {str(error)}")
        return False

#supplementary functions for the GA model saving
def convert_excluded_rows(excluded_rows_string):
    # Split the string into 36-character chunks (the length of a UUID)
    uuid_strings = [excluded_rows_string[i:i+36] for i in range(0, len(excluded_rows_string), 36)]
    
    # Convert each string to a UUID object
    uuid_list = [uuid.UUID(uuid_string) for uuid_string in uuid_strings]
    
    return uuid_list


# Function to insert the GA model into the database
def insert_ga_model(model_name, user_id, ga_params, ga_results, zscored_df, excluded_rows, sensitivity_df, zscored_statistics, baseline_productivity, predictors):
    log_message(logging.INFO, f"Inserting GA model: {model_name}")
    try:
        conn = get_db_connection()
        register_uuid()
        cur = conn.cursor()
        
        # Convert user_id to UUID if it's not already
        if not isinstance(user_id, uuid.UUID):
            user_id = uuid.UUID(str(user_id))
        
        # Convert excluded_rows to a list of UUIDs if it's not already
        if isinstance(excluded_rows, str):
            excluded_rows_list = convert_excluded_rows(excluded_rows)
        elif isinstance(excluded_rows, list):
            excluded_rows_list = [uuid.UUID(str(row)) for row in excluded_rows]
        else:
            raise ValueError("excluded_rows must be a string or a list")
        
        # Calculate additional metrics for sensitivity data
        sensitivity_df['Impact_Range'] = sensitivity_df['Max Productivity'] - sensitivity_df['Min Productivity']
        
        elasticity = []
        for _, row in sensitivity_df.iterrows():
            attr = row['Attribute']
            max_value = row['Max Productivity']
            min_value = row['Min Productivity']
            
            pct_change_attr = (zscored_statistics[attr]['max'] - zscored_statistics[attr]['min']) / zscored_statistics[attr]['median']
            pct_change_prod = (max_value - min_value) / baseline_productivity
            
            elasticity.append(pct_change_prod / pct_change_attr if pct_change_attr != 0 else 0)

        sensitivity_df['Elasticity'] = elasticity
        
        # Calculate relative impact
        total_impact = sensitivity_df['Impact_Range'].sum()
        sensitivity_df['Relative_Impact'] = sensitivity_df['Impact_Range'] / total_impact
        
        # Add Min Value and Max Value columns to sensitivity_df
        for attr in sensitivity_df['Attribute']:
            sensitivity_df.loc[sensitivity_df['Attribute'] == attr, 'Min Value'] = zscored_statistics[attr]['min']
            sensitivity_df.loc[sensitivity_df['Attribute'] == attr, 'Max Value'] = zscored_statistics[attr]['max']
        
        # Prepare the parameters for the function
        params = [
            model_name,
            user_id,
            float(ga_params['r2_threshold']),
            float(ga_params['prob_crossover']),
            float(ga_params['prob_mutation']),
            int(ga_params['num_generations']),
            int(ga_params['population_size']),
            ga_params['regression_type'],
            ga_results['response_equation'],
            float(ga_results['best_r2_score']),
            float(ga_results['full_dataset_r2']),
            ','.join(ga_results['selected_feature_names']),
            Json(zscored_df.to_dict('records')),
            excluded_rows_list,
            Json(sensitivity_df.to_dict('records')),
            ','.join(predictors)
        ]

        # Call the function
        cur.execute("""
            SELECT * FROM insert_ga_model(%s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s)
        """, params)
            
        # Fetch the result
        result = cur.fetchone()
        
        if result:
            status, model_id = result
            if status == 'SUCCESS':
                conn.commit()
                log_message(logging.INFO, f"Successfully inserted GA model: {model_name} with ID: {model_id}")
                return {'status': 'success', 'model_id': model_id}
            elif status == 'DUPLICATE':
                log_message(logging.WARNING, f"Duplicate model name: {model_name}")
                return {'status': 'duplicate', 'message': 'A model with this name already exists.'}
            else:
                conn.rollback()
                log_message(logging.ERROR, f"Error inserting GA model: {model_name}. Status: {status}")
                return {'status': 'error', 'message': 'An error occurred while saving the model.'}
        else:
            conn.rollback()
            log_message(logging.ERROR, f"No result returned when inserting GA model: {model_name}")
            return {'status': 'error', 'message': 'No result returned from the database.'}
    
    except Exception as error:
        conn.rollback()
        log_message(logging.ERROR, f"Error inserting GA model {model_name}: {str(error)}")
        return {'status': 'error', 'message': str(error)}
    
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()