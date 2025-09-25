# In db.py

import streamlit as st
import mysql.connector
from mysql.connector import Error as MySQLError
import sys

DB_CONFIG = {
    'host': st.secrets["DB_HOST"],
    'port': st.secrets.get("DB_PORT", 3306),
    'user': st.secrets["DB_HOST"],
    'password': st.secrets["DB_PASSWORD"],
    'database': st.secrets["DB_DATABASE"]
}

@st.cache_resource(ttl="30s")  # Set a short TTL to periodically refresh the connection
def get_db_connection():
    """
    Establishes and CACHES a single MySQL database connection.
    This function runs only once and reuses the connection across reruns.
    """
    st.write("Creating a new database connection...")
    try:
        # Retrieve credentials securely from st.secrets
        config = {
            'host': st.secrets["DB_HOST"],
            'user': st.secrets["DB_USER"],
            'password': st.secrets["DB_PASSWORD"],
            'database': st.secrets["DB_DATABASE"],
            'port': int(st.secrets.get("DB_PORT", 3306))

        }

        # Connect to MySQL
        conn = mysql.connector.connect(**config)
        return conn

    except MySQLError as err:
        st.error(f"Database Connection Failed! (Check Host/Security Group/Credentials)")
        st.code(f"Error Code: {err.errno}\nMessage: {err.msg}", language="text")
        # Exit the app gracefully to prevent further execution
        sys.exit(1)
    except KeyError as e:
        st.error(f"Missing Database Secret: {e}. Please check your .streamlit/secrets.toml.")
        sys.exit(1)

def execute_query(query, params=None, fetch_mode='none'):
    """
    A versatile helper function to execute any query. It manages
    connection retrieval, cursor creation, execution, and closing the cursor.

    Args:
        query (str): The SQL query string.
        params (tuple): Parameters for the query.
        fetch_mode (str): 'one', 'all', 'list', or 'dict' for fetching data.
    """
    conn = get_db_connection()
    cursor = None
    result = None

    try:
        is_dict_cursor = (fetch_mode in ['one_dict', 'all_dict'])
        cursor = conn.cursor(dictionary=is_dict_cursor)

        # Execute query with parameters
        cursor.execute(query, params)

        # Handle different query types
        if query.strip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            conn.commit()
            result = True  # Operation successful
        elif fetch_mode == 'one' or fetch_mode == 'one_dict':
            result = cursor.fetchone()
        elif fetch_mode == 'all' or fetch_mode == 'all_dict':
            result = cursor.fetchall()
        elif fetch_mode == 'list':
            result = [row[0] for row in cursor.fetchall()]

    except MySQLError as e:
        st.error(f"Database Operation Failed: {e}")
        # Clear the cache on failure to force a re-connection on the next rerun
        get_db_connection.clear()
        result = None

    finally:
        if cursor:
            cursor.close()

    return result


def create_user(username, email, password_hash):
    """Inserts a new user into the database without OTP fields."""
    query = "INSERT INTO users (username, email, password_hash, is_active) VALUES (%s, %s, %s, %s)"
    params = (username, email, password_hash, True)
    return execute_query(query, params)


def get_user(identifier):
    """Fetches a user by username or email, returning a dictionary."""
    query = "SELECT * FROM users WHERE username = %s OR email = %s"
    params = (identifier, identifier)
    return execute_query(query, params, fetch_mode='one_dict')


def update_user(user_id, **kwargs):
    """Updates a user's information in the database."""
    set_clause = ", ".join([f"{key} = %s" for key in kwargs.keys()])
    values = list(kwargs.values())
    params = tuple(values + [user_id])
    query = f"UPDATE users SET {set_clause} WHERE id = %s"
    return execute_query(query, params)


def save_user_session(username, session_name):
    """Saves a session name for a specific user."""
    user_query = "SELECT id FROM users WHERE username = %s"
    user_id_row = execute_query(user_query, (username,), fetch_mode='one')

    if user_id_row:
        user_id = user_id_row[0]
        session_query = "INSERT INTO user_sessions (user_id, session_name) VALUES (%s, %s)"
        return execute_query(session_query, (user_id, session_name))
    return False


def load_user_session(username, session_name):
    """Checks if a specific session exists for a user."""
    user_query = "SELECT id FROM users WHERE username = %s"
    user_id_row = execute_query(user_query, (username,), fetch_mode='one')

    if user_id_row:
        user_id = user_id_row[0]
        session_query = "SELECT 1 FROM user_sessions WHERE user_id = %s AND session_name = %s"
        session_found = execute_query(session_query, (user_id, session_name), fetch_mode='one')
        return session_found is not None
    return False


def delete_user_session(username, session_name):
    """Deletes a session entry for a specific user."""
    user_query = "SELECT id FROM users WHERE username = %s"
    user_id_row = execute_query(user_query, (username,), fetch_mode='one')

    if user_id_row:
        user_id = user_id_row[0]
        session_query = "DELETE FROM user_sessions WHERE user_id = %s AND session_name = %s"
        return execute_query(session_query, (user_id, session_name))
    return False


def get_user_saved_sessions(username):
    """Fetches all saved session names for a user."""
    user_query = "SELECT id FROM users WHERE username = %s"
    user_id_row = execute_query(user_query, (username,), fetch_mode='one')

    if user_id_row:
        user_id = user_id_row[0]
        sessions_query = "SELECT session_name FROM user_sessions WHERE user_id = %s"
        return execute_query(sessions_query, (user_id,), fetch_mode='list')
    return []

# def create_user(username, email, password_hash):
#     """Inserts a new user into the database without OTP fields."""
#     conn = mysql.connector.connect(**DB_CONFIG)
#     cursor = conn.cursor()
#     query = "INSERT INTO users (username, email, password_hash, is_active) VALUES (%s, %s, %s, %s)"
#     cursor.execute(query, (username, email, password_hash, True))
#     conn.commit()
#     cursor.close()
#     conn.close()
#
#
# def get_user(identifier):
#     """Fetches a user by username or email."""
#     conn = mysql.connector.connect(**DB_CONFIG)
#     cursor = conn.cursor(dictionary=True)
#     query = "SELECT * FROM users WHERE username = %s OR email = %s"
#     cursor.execute(query, (identifier, identifier))
#     user = cursor.fetchone()
#     cursor.close()
#     conn.close()
#     return user
#
#
# def update_user(user_id, **kwargs):
#     """Updates a user's information in the database."""
#     conn = mysql.connector.connect(**DB_CONFIG)
#     cursor = conn.cursor()
#
#     set_clause = ", ".join([f"{key} = %s" for key in kwargs.keys()])
#     values = list(kwargs.values())
#     values.append(user_id)
#
#     query = f"UPDATE users SET {set_clause} WHERE id = %s"
#     cursor.execute(query, tuple(values))
#     conn.commit()
#     cursor.close()
#     conn.close()
#
#
# def save_user_session(username, session_name):
#     """Saves a session name for a specific user."""
#     conn = mysql.connector.connect(**DB_CONFIG)
#     cursor = conn.cursor()
#
#     user_query = "SELECT id FROM users WHERE username = %s"
#     cursor.execute(user_query, (username,))
#     user_id = cursor.fetchone()[0]
#
#     session_query = "INSERT INTO user_sessions (user_id, session_name) VALUES (%s, %s)"
#     cursor.execute(session_query, (user_id, session_name))
#     conn.commit()
#     cursor.close()
#     conn.close()
#
#
# def load_user_session(username, session_name):
#     """Checks if a specific session exists for a user."""
#     conn = mysql.connector.connect(**DB_CONFIG)
#     cursor = conn.cursor()
#     user_query = "SELECT id FROM users WHERE username = %s"
#     cursor.execute(user_query, (username,))
#     user_id = cursor.fetchone()[0]
#
#     session_query = "SELECT * FROM user_sessions WHERE user_id = %s AND session_name = %s"
#     cursor.execute(session_query, (user_id, session_name))
#     session = cursor.fetchone()
#     cursor.close()
#     conn.close()
#     return session is not None
#
#
# def delete_user_session(username, session_name):
#     """Deletes a session entry for a specific user."""
#     conn = mysql.connector.connect(**DB_CONFIG)
#     cursor = conn.cursor()
#     user_query = "SELECT id FROM users WHERE username = %s"
#     cursor.execute(user_query, (username,))
#     user_id = cursor.fetchone()[0]
#
#     session_query = "DELETE FROM user_sessions WHERE user_id = %s AND session_name = %s"
#     cursor.execute(session_query, (user_id, session_name))
#     conn.commit()
#     cursor.close()
#     conn.close()
#
#
# def get_user_saved_sessions(username):
#     """Fetches all saved session names for a user."""
#     conn = mysql.connector.connect(**DB_CONFIG)
#     cursor = conn.cursor()
#     user_query = "SELECT id FROM users WHERE username = %s"
#     cursor.execute(user_query, (username,))
#     user_id = cursor.fetchone()[0]
#
#     sessions_query = "SELECT session_name FROM user_sessions WHERE user_id = %s"
#     cursor.execute(sessions_query, (user_id,))
#     sessions = [row[0] for row in cursor.fetchall()]
#     cursor.close()
#     conn.close()
#     return sessions