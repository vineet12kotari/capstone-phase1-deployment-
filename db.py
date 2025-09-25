# In db.py

import mysql.connector

# You'll need to configure this with your MySQL details
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'dinkar7898880500',
    'database': 'Streamlit_DB'
}


def create_user(username, email, password_hash):
    """Inserts a new user into the database without OTP fields."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    query = "INSERT INTO users (username, email, password_hash, is_active) VALUES (%s, %s, %s, %s)"
    cursor.execute(query, (username, email, password_hash, True))
    conn.commit()
    cursor.close()
    conn.close()


def get_user(identifier):
    """Fetches a user by username or email."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM users WHERE username = %s OR email = %s"
    cursor.execute(query, (identifier, identifier))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user


def update_user(user_id, **kwargs):
    """Updates a user's information in the database."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    set_clause = ", ".join([f"{key} = %s" for key in kwargs.keys()])
    values = list(kwargs.values())
    values.append(user_id)

    query = f"UPDATE users SET {set_clause} WHERE id = %s"
    cursor.execute(query, tuple(values))
    conn.commit()
    cursor.close()
    conn.close()


def save_user_session(username, session_name):
    """Saves a session name for a specific user."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    user_query = "SELECT id FROM users WHERE username = %s"
    cursor.execute(user_query, (username,))
    user_id = cursor.fetchone()[0]

    session_query = "INSERT INTO user_sessions (user_id, session_name) VALUES (%s, %s)"
    cursor.execute(session_query, (user_id, session_name))
    conn.commit()
    cursor.close()
    conn.close()


def load_user_session(username, session_name):
    """Checks if a specific session exists for a user."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    user_query = "SELECT id FROM users WHERE username = %s"
    cursor.execute(user_query, (username,))
    user_id = cursor.fetchone()[0]

    session_query = "SELECT * FROM user_sessions WHERE user_id = %s AND session_name = %s"
    cursor.execute(session_query, (user_id, session_name))
    session = cursor.fetchone()
    cursor.close()
    conn.close()
    return session is not None


def delete_user_session(username, session_name):
    """Deletes a session entry for a specific user."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    user_query = "SELECT id FROM users WHERE username = %s"
    cursor.execute(user_query, (username,))
    user_id = cursor.fetchone()[0]

    session_query = "DELETE FROM user_sessions WHERE user_id = %s AND session_name = %s"
    cursor.execute(session_query, (user_id, session_name))
    conn.commit()
    cursor.close()
    conn.close()


def get_user_saved_sessions(username):
    """Fetches all saved session names for a user."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    user_query = "SELECT id FROM users WHERE username = %s"
    cursor.execute(user_query, (username,))
    user_id = cursor.fetchone()[0]

    sessions_query = "SELECT session_name FROM user_sessions WHERE user_id = %s"
    cursor.execute(sessions_query, (user_id,))
    sessions = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return sessions