import os
from psycopg2.extras import RealDictCursor
import logging
from logging.handlers import RotatingFileHandler
from utils.db import get_db_connection


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


def get_wells():
    log_message(logging.INFO, "Fetching all wells")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT DISTINCT wm.well_id, wm.well_name
            FROM well_master wm
            JOIN well_completion_records wcr ON wm.well_id = wcr.well_id
            ORDER BY wm.well_name
        """)
        wells = cur.fetchall()
        cur.close()
        conn.close()
        log_message(logging.INFO, f"Fetched {len(wells)} wells")
        return wells
    except Exception as error:
        log_message(logging.ERROR, f"Error fetching wells: {str(error)}")
        return []

def get_stages_for_well(well_id):
    log_message(logging.INFO, f"Fetching stages for well_id: {well_id}")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT DISTINCT stage
            FROM well_completion_records
            WHERE well_id = %s
            ORDER BY stage
        """, (well_id,))
        stages = [row['stage'] for row in cur.fetchall()]
        cur.close()
        conn.close()
        log_message(logging.INFO, f"Fetched {len(stages)} stages for well_id: {well_id}")
        return stages
    except Exception as error:
        log_message(logging.ERROR, f"Error fetching stages for well_id {well_id}: {str(error)}")
        return []

def get_well_completion_data(well_id, stage):
    log_message(logging.INFO, f"Fetching completion data for well_id: {well_id}, stage: {stage}")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT treating_pressure, slurry_rate, bottomhole_prop_mass, time_seconds, epoch
            FROM well_completion_records
            WHERE well_id = %s AND stage = %s
            ORDER BY time_seconds
        """, (well_id, stage))
        data = cur.fetchall()
        cur.close()
        conn.close()
        log_message(logging.INFO, f"Fetched {len(data)} records for well_id: {well_id}, stage: {stage}")
        return data
    except Exception as error:
        log_message(logging.ERROR, f"Error fetching completion data for well_id {well_id}, stage {stage}: {str(error)}")
        return []