# In auth.py

import hashlib
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import your database functions
from db import create_user, get_user, update_user


def hash_password(password):
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def signup_user(username, email, password):
    """Handles user registration in a single step."""
    if get_user(username) or get_user(email):
        return False, "Username or email already exists."

    password_hash = hash_password(password)

    # Create the user directly with a True is_active status
    create_user(username, email, password_hash)

    return True, "Account created successfully! You can now log in."


def verify_and_login(username, password):
    """Verifies credentials for user login."""
    user = get_user(username)
    if user is None:
        return False, "Invalid username or password."

    if user['password_hash'] != hash_password(password):
        return False, "Invalid username or password."

    return True, "Login successful."
