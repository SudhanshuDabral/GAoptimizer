import os
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from streamlit_authenticator.utilities.hasher import Hasher
import streamlit as st
import json
import pandas as pd
import string
import secrets
import logging
from logging.handlers import RotatingFileHandler
from psycopg2.extras import Json, register_uuid
from psycopg2.extras import execute_values


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

def update_modeling_data(well_id, consolidated_output, user_id):
    log_message(logging.INFO, f"Updating modeling data for well_id: {well_id}")
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Get existing stages for this well
        cur.execute("SELECT stage FROM data_for_modeling WHERE well_id = %s", (well_id,))
        existing_stages = set(row[0] for row in cur.fetchall())

        result = []

        # Update existing records and insert new ones
        for row in consolidated_output:
            if row['Stages'] in existing_stages:
                # Update existing record
                update_query = """
                UPDATE data_for_modeling 
                SET tee = %s, median_dhpm = %s, median_dp = %s, downhole_ppm = %s, 
                    total_dhppm = %s, total_slurry_dp = %s, median_slurry = %s, 
                    total_dh_prop = %s, updated_by = %s, updated_on = CURRENT_TIMESTAMP
                WHERE well_id = %s AND stage = %s
                RETURNING data_id, stage
                """
                cur.execute(update_query, (
                    row['TEE'],
                    row['MedianDHPM'],
                    row['MedianDP'],
                    row['DownholePPM'],
                    row['TotalDHPPM'],
                    row['TotalSlurryDP'],
                    row['MedianSlurry'],
                    row['TotalDHProp'],
                    user_id,
                    well_id,
                    row['Stages']
                ))
            else:
                # Insert new record
                insert_query = """
                INSERT INTO data_for_modeling 
                (well_id, tee, median_dhpm, median_dp, downhole_ppm, total_dhppm, 
                 total_slurry_dp, median_slurry, total_dh_prop, stage, updated_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING data_id, stage
                """
                cur.execute(insert_query, (
                    well_id,
                    row['TEE'],
                    row['MedianDHPM'],
                    row['MedianDP'],
                    row['DownholePPM'],
                    row['TotalDHPPM'],
                    row['TotalSlurryDP'],
                    row['MedianSlurry'],
                    row['TotalDHProp'],
                    row['Stages'],
                    user_id
                ))

            # Fetch the result of each operation
            operation_result = cur.fetchone()
            if operation_result:
                result.append({"data_id": str(operation_result[0]), "stage": operation_result[1]})

        conn.commit()
        cur.close()
        conn.close()

        log_message(logging.INFO, f"Successfully updated/inserted {len(result)} records for well_id: {well_id}")
        return result

    except Exception as error:
        log_message(logging.ERROR, f"Error updating modeling data for well_id {well_id}: {str(error)}")
        return None
    
def bulk_insert_well_completion_records(well_id, stage, data, user_id):
    log_message(logging.INFO, f"Bulk inserting well completion records for well_id: {well_id}, stage: {stage}")
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Prepare the data for bulk insertion
        records = [
            (well_id, stage, row['treating_pressure'], row['slurry_rate'], 
             row['bottomhole_prop_mass'], row['time_seconds'], 
             row['epoch'], user_id)
            for _, row in data.iterrows()
        ]

        # Perform bulk insert
        insert_query = """
        INSERT INTO well_completion_records 
        (well_id, stage, treating_pressure, slurry_rate, bottomhole_prop_mass, time_seconds, epoch, updated_by)
        VALUES %s
        """
        execute_values(cur, insert_query, records)

        conn.commit()
        inserted_count = cur.rowcount
        cur.close()
        conn.close()

        log_message(logging.INFO, f"Successfully inserted {inserted_count} well completion records for well_id: {well_id}, stage: {stage}")
        return {"status": "success", "records_inserted": inserted_count}

    except Exception as error:
        log_message(logging.ERROR, f"Error inserting well completion records for well_id {well_id}, stage {stage}: {str(error)}")
        return {"status": "error", "message": str(error)}



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
                row['window_iteration'],
                row['SlrWin'],
                row['PmaxminWin'],
                row['DownholeWinProp'],
                user_id
            ))

        psycopg2.extras.execute_batch(cur, 
            """
            INSERT INTO arrays (data_id, window_iteration, slr_win, pmaxmin_win, downhole_win_prop, updated_by)
            VALUES (%s, %s, %s, %s, %s, %s)
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
    
def create_new_well(well_name, well_api, latitude, longitude, tvd, reservoir_pressure, user_id):
    log_message(logging.INFO, f"Creating new well: {well_name}")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO well_master (well_name, well_api, latitude, longitude, tvd, reservoir_pressure, updated_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING well_id
        """, (well_name, well_api, float(latitude), float(longitude), float(tvd), float(reservoir_pressure), user_id))
        
        well_id = cur.fetchone()[0]
        
        conn.commit()
        cur.close()
        conn.close()
        log_message(logging.INFO, f"Successfully created well: {well_name} with ID: {well_id}")
        return well_id
    except Exception as error:
        log_message(logging.ERROR, f"Error creating well {well_name}: {str(error)}")
        return None

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
            SELECT data_id, well_id, tee, median_dhpm, median_dp, downhole_ppm, total_dhppm, total_slurry_dp, median_slurry, total_dh_prop, stage
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


def get_array_data(well_id, stage):
    log_message(logging.INFO, f"Fetching array data for well_id: {well_id}, stage: {stage}")
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT a.window_iteration, a.slr_win, a.pmaxmin_win, a.downhole_win_prop
        FROM arrays a
        JOIN data_for_modeling d ON a.data_id = d.data_id
        WHERE d.well_id = %s AND d.stage = %s
        ORDER BY a.window_iteration ASC
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

#################################db functions for model_explorer.py############################################
def fetch_all_models_with_users():
    log_message(logging.INFO, "Fetching all models with user information")
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT gm.model_id, gm.model_name, gm.created_on, um.name as created_by
                    FROM ga_model gm
                    JOIN user_master um ON gm.updated_by = um.user_id
                    ORDER BY gm.created_on DESC
                """)
                models = cur.fetchall()
        log_message(logging.INFO, f"Successfully fetched {len(models)} models")
        return models
    except Exception as e:
        log_message(logging.ERROR, f"Error fetching models with user information: {str(e)}")
        return None

def fetch_model_details(model_id):
    log_message(logging.INFO, f"Fetching details for model_id: {model_id}")
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT gm.*, mr.equation, mr.weighted_r2_score, mr.full_dataset_r2_score, mr.selected_feature_names
                    FROM ga_model gm
                    JOIN model_results mr ON gm.model_id = mr.model_id
                    WHERE gm.model_id = %s
                """, (model_id,))
                model_details = cur.fetchone()
        
        if model_details:
            log_message(logging.INFO, f"Successfully fetched details for model_id: {model_id}")
            # Convert comma-separated strings to lists
            model_details['predictors'] = model_details['predictors'].split(',')
            model_details['selected_feature_names'] = model_details['selected_feature_names'].split(',')
        else:
            log_message(logging.WARNING, f"No details found for model_id: {model_id}")
        return model_details
    except Exception as e:
        log_message(logging.ERROR, f"Error fetching model details for model_id {model_id}: {str(e)}")
        return None

def fetch_training_data(model_id):
    log_message(logging.INFO, f"Fetching training data for model_id: {model_id}")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT mtd.*, wm.well_name
                    FROM model_training_data mtd
                    JOIN well_master wm ON mtd.well_id = wm.well_id
                    WHERE mtd.model_id = %s
                """, (model_id,))
                columns = [desc[0] for desc in cur.description]
                data = cur.fetchall()
                df = pd.DataFrame(data, columns=columns)
        log_message(logging.INFO, f"Successfully fetched {len(df)} training data records for model_id: {model_id}")
        return df
    except Exception as e:
        log_message(logging.ERROR, f"Error fetching training data for model_id {model_id}: {str(e)}")
        return None

def fetch_sensitivity_data(model_id):
    log_message(logging.INFO, f"Fetching sensitivity data for model_id: {model_id}")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM model_sensitivity
                    WHERE model_id = %s
                """, (model_id,))
                columns = [desc[0] for desc in cur.description]
                data = cur.fetchall()
                df = pd.DataFrame(data, columns=columns)
        log_message(logging.INFO, f"Successfully fetched {len(df)} sensitivity data records for model_id: {model_id}")
        return df
    except Exception as e:
        log_message(logging.ERROR, f"Error fetching sensitivity data for model_id {model_id}: {str(e)}")
        return None