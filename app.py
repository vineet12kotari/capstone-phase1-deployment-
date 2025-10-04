import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from datetime import datetime
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import speech_recognition as sr
import os
import xmltodict
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
import base64
from dotenv import load_dotenv
import json
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import snowflake.connector
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
import urllib.parse
import re
# --- New Imports for Authentication, Google Drive and MySQL ---
import gspread
from google.oauth2 import service_account
import mysql.connector

# Import your newly created modules
from auth import signup_user, verify_and_login
from db import create_user, get_user, update_user, save_user_session, load_user_session, delete_user_session, \
    get_user_saved_sessions

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API Key not found. Please set it in your .env file.")
else:
    llm = OpenAI(api_token=api_key)

# Create a directory for session state files
SESSION_STATE_DIR = "user_sessions"
if not os.path.exists(SESSION_STATE_DIR):
    os.makedirs(SESSION_STATE_DIR)

# --- Register fonts for ReportLab ---
pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
try:
    # Use os.path.join to create a platform-independent path to the fonts folder
    font_dir = os.path.join(os.path.dirname(__file__), 'fonts')

    # Check if the fonts exist before registering them
    if os.path.exists(os.path.join(font_dir, 'Arial.ttf')):
        pdfmetrics.registerFont(TTFont('Arial', os.path.join(font_dir, 'Arial.ttf')))

    if os.path.exists(os.path.join(font_dir, 'Arialbd.ttf')):
        pdfmetrics.registerFont(TTFont('Arial-Bold', os.path.join(font_dir, 'Arialbd.ttf')))

except Exception as e:
    st.warning(f"Could not load professional fonts: {e}. Falling back to default.")


def get_ai_chart_analysis(df_for_chart, chart_type, x_col, y_col=None, hue_col=None, llm_instance=None):
    """
    Uses an LLM to generate a factual insight and a strategic conclusion
    based ONLY on the data provided for the chart.
    """
    if llm_instance is None or df_for_chart.empty:
        return {
            "insight": "AI analysis is not available.",
            "conclusion": "AI analysis is not available."
        }

    # Use a limited sample of the data to keep the prompt concise and fast
    sample_df = df_for_chart.head(15)
    data_summary = sample_df.to_string()

    # Construct a detailed context for the AI
    chart_context = f"A '{chart_type}' was created. The X-axis represents '{x_col}'."
    if y_col and y_col not in ['None', 'count', 'frequency']:
        chart_context += f" The Y-axis represents '{y_col}'."
    if hue_col and hue_col != 'None':
        chart_context += f" The data is grouped or colored by '{hue_col}'."

    # --- HEATMAP SPECIFIC ENHANCEMENT ---
    heatmap_context = ""
    if chart_type == "Heatmap":
        heatmap_context = (
            "The data provided is a **correlation matrix**. Your analysis must focus on identifying the **strongest positive** "
            "(closest to 1) and **strongest negative** (closest to -1) correlations between variables."
        )

    try:
        sdf_local = SmartDataframe(df_for_chart, config={"llm": llm_instance})

        # (Incorporating Heatmap Context)
        insight_prompt = (
            f"You are a data analyst. Your task is to provide one factual, data-driven insight. "
            f"Base your analysis STRICTLY on the data summary below. {heatmap_context} "
            f"The data was used for this visualization: {chart_context}. "
            f"Describe the most significant pattern, comparison, or finding from these numbers. "
            f"For example: 'The count for Category A (X) is Y% higher than for Category B (Z)'. "
            f"Your response must be a single, concise sentence. Here is the data:\n{data_summary}"
        )
        insight = sdf_local.chat(insight_prompt)


        conclusion_prompt = (
            f"Based *only* on the following insight: '{insight}'. "
            f"Provide a brief, strategic conclusion for a business stakeholder. What is the business implication? "
            f"Your response MUST BE A TEXTUAL SUMMARY. Do not generate a chart or a file path."
        )
        conclusion = sdf_local.chat(conclusion_prompt)

        # SAFEGUARD: Check for and handle file path responses
        if isinstance(conclusion, str) and (
                '.png' in conclusion or '.jpg' in conclusion or '/' in conclusion or '\\' in conclusion):
            conclusion = "AI returned a chart instead of a textual conclusion. This can happen with ambiguous data. Please try again or use a different chart type for clearer analysis."
        elif not isinstance(conclusion, str):
            conclusion = "A non-textual conclusion was returned by the AI."

        return {"insight": str(insight).strip(), "conclusion": str(conclusion).strip()}

    except Exception as e:
        st.warning(f"Could not generate AI analysis for the chart: {e}")
        return {
            "insight": "AI insight could not be generated due to an error.",
            "conclusion": "AI conclusion could not be generated due to an error."
        }



# Helper Functions

def perform_action(action_func, action_log_text, action_data, fill_value=None):
    st.session_state.df_history[st.session_state.active_df_key].append(st.session_state.df.copy())
    st.session_state.redo_history = []

    st.session_state.df = action_func(st.session_state.df)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append(f"[{timestamp}] {action_log_text}")
    st.session_state.macro_actions.append(action_data)

    st.success(f"Action performed: {action_log_text}")
    if fill_value is not None:
        st.session_state.temp_fill_value = fill_value
    st.rerun()


def perform_viz_action(action_log_text, action_data, chart_data=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append(f"[{timestamp}] {action_log_text}")
    st.session_state.macro_actions.append(action_data)
    st.success(f"Visualization generated: {action_log_text}")
    if chart_data:
        st.session_state.chart_data.append(chart_data)


def run_macro(df, macro_actions, actions_to_run=None):
    """
    Executes a list of actions (a macro) on a DataFrame.
    Can be configured to run only a subset of actions.
    """
    current_df = df.copy()
    history_log = []

    if actions_to_run is None:
        actions_to_run = [action['action'] for action in macro_actions]

    for action in macro_actions:
        action_type = action.get("action")
        if action_type not in actions_to_run:
            continue

        params = action.get("params", {})
        log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Executing: {action_type}"
        try:
            if action_type == "remove_duplicates":
                before = len(current_df)
                current_df = current_df.drop_duplicates()
                log_entry += f" - Removed {before - len(current_df)} duplicate rows."
            elif action_type == "remove_columns":
                columns = params.get("columns", [])
                columns_to_drop = [col for col in columns if col in current_df.columns]
                if columns_to_drop:
                    current_df = current_df.drop(columns=columns_to_drop)
                    log_entry += f" - Removed columns: {', '.join(columns_to_drop)}."
                else:
                    log_entry += f" - Columns {', '.join(columns)} not found. Skipped."
            elif action_type == "split_column":
                col = params["col"]
                split_method = params["method"]
                new_names = params["new_names"]
                if col in current_df.columns:
                    if split_method == "By Delimiter":
                        delimiter = params["delimiter"]
                        split_data = current_df[col].astype(str).str.split(delimiter, expand=True)
                    elif split_method == "By Number of Characters":
                        num_chars = params["num_chars"]
                        if isinstance(num_chars, list):
                            def split_by_chars(text, lengths):
                                parts = []
                                current_pos = 0
                                for length in lengths:
                                    parts.append(text[current_pos:current_pos + length])
                                    current_pos += length
                                return parts

                            split_data = pd.DataFrame(
                                current_df[col].astype(str).apply(lambda x: split_by_chars(x, num_chars)).tolist(),
                                index=current_df.index)
                        else:
                            st.error("Number of characters must be a list for macro execution.")
                            continue

                    split_data.columns = new_names
                    current_df = pd.concat([current_df, split_data], axis=1)
                    log_entry += f" - Split column '{col}' into {len(new_names)} columns using '{split_method}'."
                else:
                    log_entry += f" - Column '{col}' not found. Skipped."
            elif action_type == "change_dtype":
                col = params["col"]
                new_type = params["new_type"]
                if col in current_df.columns:
                    if new_type == "datetime":
                        current_df[col] = pd.to_datetime(current_df[col], errors="coerce")
                    else:
                        current_df[col] = current_df[col].astype(new_type)
                    log_entry += f" - Changed '{col}' to type '{new_type}'."
                else:
                    log_entry += f" - Column '{col}' not found. Skipped."
            elif action_type == "fill_nulls":
                col = params["col"]
                strategy = params["strategy"]
                if col in current_df.columns:
                    if strategy in ["Mean", "Median", "Mode"]:
                        if "group_cols" in params:
                            group_cols = params["group_cols"]
                            if strategy == "Mean":
                                current_df[col] = current_df.groupby(group_cols)[col].transform(
                                    lambda x: x.fillna(x.mean()))
                            elif strategy == "Median":
                                current_df[col] = current_df.groupby(group_cols)[col].transform(
                                    lambda x: x.fillna(x.median()))
                            elif strategy == "Mode":
                                current_df[col] = current_df.groupby(group_cols)[col].transform(
                                    lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
                            log_entry += f" - Conditionally filled nulls in '{col}' with '{strategy}' based on {', '.join(group_cols)}."
                        else:
                            if strategy == "Mean":
                                current_df[col] = current_df[col].fillna(current_df[col].mean())
                            elif strategy == "Median":
                                current_df[col] = current_df[col].fillna(current_df[col].median())
                            elif strategy == "Mode":
                                current_df[col] = current_df[col].fillna(current_df[col].mode().iloc[0])
                            log_entry += f" - Filled nulls in '{col}' with '{strategy}'."
                    elif strategy == "Custom Value":
                        current_df[col] = current_df[col].fillna(params["value"])
                        log_entry += f" - Filled nulls in '{col}' with custom value '{params['value']}'."
                    elif strategy == "Fill with Previous":
                        current_df[col] = current_df[col].fillna(method='ffill')
                        log_entry += f" - Filled nulls in '{col}' with previous values."
                    elif strategy == "Fill with Next":
                        current_df[col] = current_df[col].fillna(method='bfill')
                        log_entry += f" - Filled nulls in '{col}' with next values."
                    elif strategy == "AI Recommendation":
                        log_entry += f" - Skipped AI Recommendation for '{col}' during macro run."
                else:
                    log_entry += f" - Column '{col}' not found. Skipped."

            elif action_type == "merge_columns":
                cols = params["cols"]
                new_name = params["new_name"]
                delimiter = params["delimiter"]
                if all(c in current_df.columns for c in cols):
                    current_df[new_name] = current_df[cols].astype(str).agg(delimiter.join, axis=1)
                    log_entry += f" - Merged columns {', '.join(cols)} into a new column '{new_name}'."
                else:
                    log_entry += f" - One or more columns {', '.join(cols)} not found. Skipped."
            elif action_type == "create_column":
                col_to_categorize = params["col"]
                new_category_col = params["new_col"]
                conditions = params["conditions"]
                labels = params["labels"]
                final_label = params["final_label"]
                if col_to_categorize in current_df.columns:
                    current_df[new_category_col] = final_label
                    for i in range(len(conditions)):
                        cond_text = convert_condition_to_python(conditions[i], col_to_categorize)
                        label = labels[i]
                        try:
                            mask = eval(cond_text, {'pd': pd, 'df': current_df})
                            current_df.loc[mask, new_category_col] = label
                        except Exception as e:
                            history_log.append(
                                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipping condition '{conditions[i]}' due to error: {e}")
                    log_entry += f" - Created a new column '{new_category_col}' from conditions."
                else:
                    log_entry += f" - Column '{col_to_categorize}' not found. Skipped."
            elif action_type == "sort":
                col = params["col"]
                ascending = params["ascending"]
                if col in current_df.columns:
                    current_df = current_df.sort_values(by=col, ascending=ascending)
                    log_entry += f" - Sorted dataset by '{col}'."
                else:
                    log_entry += f" - Column '{col}' not found. Skipped."
            elif action_type == "replace_values":
                col = params["col"]
                method = params["method"]
                if col in current_df.columns:
                    if method == "Manual":
                        old_values = params["old_values"]
                        new_values = params["new_values"]
                        replace_dict = dict(zip(old_values, new_values))
                        current_df[col] = current_df[col].astype(str).str.strip()
                        current_df[col] = current_df[col].apply(lambda x: replace_dict.get(x.lower(), x))
                        log_entry += f" - Replaced values manually in column '{col}'."
                    elif method == "By Condition":
                        condition = params["condition"]
                        replacement_value = params["replacement_value"]
                        try:
                            original_dtype = current_df[col].dtype
                            current_df[col] = current_df[col].astype(str)
                            mask = eval(f'current_df["{col}"].astype(float){condition}', {'current_df': current_df})
                            current_df.loc[mask, col] = str(replacement_value)
                            try:
                                current_df[col] = current_df[col].astype(original_dtype)
                            except:
                                st.warning(
                                    f"Could not convert '{col}' back to original type. It is now of type 'object'.")
                                pass

                            log_entry += f" - Replaced values by condition '{condition}' with '{replacement_value}' in column '{col}'."
                        except Exception as e:
                            log_entry += f" - Error applying conditional replace: {e}"
                else:
                    log_entry += f" - Column '{col}' not found. Skipped."
            elif action_type == "drop_null_rows":
                before = len(current_df)
                current_df = current_df.dropna()
                log_entry += f" - Dropped {before - len(current_df)} rows with NULL values."
            elif action_type == "remove_rows_by_condition":
                col = params["col"]
                operator = params["operator"]
                value = params["value"]
                if col in current_df.columns:
                    before = len(current_df)
                    if operator == "contains":
                        current_df = current_df[~current_df[col].astype(str).str.contains(value, case=False, na=False)]
                    elif pd.api.types.is_numeric_dtype(current_df[col]):
                        value = float(value)
                        if operator == "<":
                            current_df = current_df[current_df[col] >= value]
                        elif operator == ">":
                            current_df = current_df[current_df[col] <= value]
                        elif operator == "<=":
                            current_df = current_df[current_df[col] > value]
                        elif operator == ">=":
                            current_df = current_df[current_df[col] < value]
                        elif operator == "==":
                            current_df = current_df[current_df[col] != value]
                        elif operator == "!=":
                            current_df = current_df[current_df[col] == value]
                    log_entry += f" - Removed {before - len(current_df)} rows based on a condition."
                else:
                    log_entry += f" - Column '{col}' not found. Skipped."
            elif action_type == "remove_outliers":
                col = params["col"]
                if col in current_df.columns and pd.api.types.is_numeric_dtype(current_df[col]):
                    Q1 = current_df[col].quantile(0.25)
                    Q3 = current_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    before = len(current_df)
                    current_df = current_df[~((current_df[col] < lower_bound) | (current_df[col] > upper_bound))]
                    log_entry += f" - Removed {before - len(current_df)} outliers from '{col}'."
                else:
                    log_entry += f" - Column '{col}' not found or is not numeric. Skipped."
            elif action_type == "aggregate":
                log_entry += f" - Macro generated an aggregation, which is for display only. (Skipped)"
            elif action_type == "chart":
                log_entry += f" - Macro generated a chart, which is for display only. (Skipped)"
            elif action_type == "join":
                log_entry += f" - Join action is not supported for macro execution. (Skipped)"
            elif action_type == "create_column_arithmetic":
                new_col = params["new_col"]
                cols = params["cols"]
                operation = params["operation"]
                if all(c in current_df.columns for c in cols):
                    new_series = current_df[cols[0]].copy()
                    for col in cols[1:]:
                        if operation == '+':
                            new_series += current_df[col]
                        elif operation == '-':
                            new_series -= current_df[col]
                        elif operation == '*':
                            new_series *= current_df[col]
                        elif operation == '/':
                            new_series /= current_df[col]
                    current_df[new_col] = new_series
                    log_entry += f" - Created new column '{new_col}' by arithmetic operation on {', '.join(cols)}."
                else:
                    log_entry += f" - One or more columns {', '.join(cols)} not found. Skipped."
            elif action_type == "create_column_single_arithmetic":
                new_col = params["new_col"]
                col = params["col"]
                operation = params["operation"]
                factor = params["factor"]
                if col in current_df.columns:
                    new_series = current_df[col].copy()
                    if operation == "Add by":
                        new_series += factor
                    elif operation == "Subtract by":
                        new_series -= factor
                    elif operation == "Multiply by":
                        new_series *= factor
                    elif operation == "Divide by":
                        new_series /= factor
                    current_df[new_col] = new_series
                    log_entry += f" - Created new column '{new_col}' by '{operation}' on '{col}' with factor {factor}."
                else:
                    log_entry += f" - Column '{col}' not found. Skipped."
            elif action_type == "create_column_date_part":
                new_col = params["new_col"]
                col = params["col"]
                part = params["part"]
                if col in current_df.columns:
                    try:
                        temp_series = pd.to_datetime(current_df[col], errors='coerce')
                        if part == 'month_name':
                            new_series = temp_series.dt.month_name()
                        elif part == 'year':
                            new_series = temp_series.dt.year
                        elif part == 'day':
                            new_series = temp_series.dt.day
                        elif part == 'month':
                            new_series = temp_series.dt.month
                        current_df[new_col] = new_series
                        log_entry += f" - Created new column '{new_col}' by extracting '{part}' from '{col}'."
                    except Exception as e:
                        history_log.append(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR extracting date part from '{col}': {e}.")
                else:
                    log_entry += f" - Column '{col}' not found. Skipped."
            elif action_type == "create_column_date_add_subtract":
                new_col = params["new_col"]
                col = params["col"]
                value = params["value"]
                unit = params["unit"]
                operation = params["operation"]
                if col in current_df.columns:
                    try:
                        temp_series = pd.to_datetime(current_df[col], errors='coerce')
                        if operation == "Add":
                            if unit == "Days":
                                new_series = temp_series + pd.to_timedelta(value, unit='d')
                            elif unit == "Months":
                                new_series = temp_series + pd.DateOffset(months=value)
                            elif unit == "Years":
                                new_series = temp_series + pd.DateOffset(years=value)
                        elif operation == "Subtract":
                            if unit == "Days":
                                new_series = temp_series - pd.to_timedelta(value, unit='d')
                            elif unit == "Months":
                                new_series = temp_series - pd.DateOffset(months=value)
                            elif unit == "Years":
                                new_series = temp_series - pd.DateOffset(years=value)
                        current_df[new_col] = new_series
                        log_entry += f" - Created new column '{new_col}' by '{operation}ing' {value} {unit.lower()} from '{col}'."
                    except Exception as e:
                        history_log.append(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR performing date arithmetic on '{col}': {e}.")
                else:
                    log_entry += f" - Column '{col}' not found. Skipped."
            history_log.append(log_entry)
        except Exception as e:
            history_log.append(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR executing macro step '{action_type}': {e}.")
            return current_df, history_log, False
    history_log.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Macro finished successfully.")
    return current_df, history_log, True


def ai_fill_recommendation(df, column):
    if pd.api.types.is_numeric_dtype(df[column]):
        return df[column].mean()
    else:
        return df[column].mode().iloc[0] if not df[column].mode().empty else "N/A"


def get_chart_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def get_chart_download_link(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf


def recognize_speech(selected_language_code):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio, language=selected_language_code)
    except sr.UnknownValueError:
        st.error("Could not understand the audio")
        return None
    except sr.RequestError:
        st.error("Error with the speech recognition service")
        return None


def load_data(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".csv":
        return pd.read_csv(uploaded_file)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file, engine='openpyxl')
    elif ext == ".parquet":
        return pd.read_parquet(uploaded_file)
    elif ext == ".json":
        return pd.read_json(uploaded_file)
    elif ext == ".xml":
        xml_data = uploaded_file.read()
        data_dict = xmltodict.parse(xml_data)

        def find_list_in_dict(d):
            for value in d.values():
                if isinstance(value, list):
                    return value
                elif isinstance(value, dict):
                    result = find_list_in_dict(value)
                    if result is not None:
                        return result
            return None

        records = find_list_in_dict(data_dict)
        if records:
            return pd.DataFrame(records)
        else:
            st.error("Could not find a list of records in the XML file.")
            return None
    elif ext == ".orc":
        try:
            return pd.read_orc(uploaded_file)
        except ImportError:
            st.error("Please install pyarrow to read ORC files: pip install pyarrow")
            return None
    else:
        st.error("Unsupported file format.")
        return None


def load_data_from_s3(bucket, key, file_format, aws_access_key_id, aws_secret_access_key, aws_region):
    """Loads a dataset from an AWS S3 bucket into a pandas DataFrame."""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = io.BytesIO(obj['Body'].read())

        if file_format == 'csv':
            return pd.read_csv(data)
        elif file_format == 'xlsx':
            return pd.read_excel(data, engine='openpyxl')
        elif file_format == 'json':
            return pd.read_json(data)
        elif file_format == 'parquet':
            return pd.read_parquet(data)
        elif file_format == 'xml':
            xml_data = data.getvalue()
            data_dict = xmltodict.parse(xml_data)

            def find_list_in_dict(d):
                for value in d.values():
                    if isinstance(value, list):
                        return value
                    elif isinstance(value, dict):
                        result = find_list_in_dict(value)
                        if result is not None:
                            return result
                return None

            records = find_list_in_dict(data_dict)
            if records:
                return pd.DataFrame(records)
            else:
                st.error("Could not find a list of records in the XML file.")
                return pd.DataFrame()
        else:
            st.error("Unsupported file format for S3 loading.")
            return None
    except (NoCredentialsError, ClientError) as e:
        st.error(f"AWS S3 Error: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading file from S3: {e}")
        return None


def load_data_from_gdrive(credentials_file, sheet_id):
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = service_account.Credentials.from_service_account_info(
            credentials_file, scopes=scope
        )
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(sheet_id)
        worksheet = spreadsheet.get_worksheet(0)
        data = worksheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading data from Google Drive: {e}")
        return None


def load_data_from_mysql(host, user, password, database, query):
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except mysql.connector.Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return None


def load_and_merge_data(uploaded_files):
    if not uploaded_files:
        return None, "Please upload one or more files."
    all_dfs = []
    skipped_files = []
    first_df = None

    for uploaded_file in uploaded_files:
        try:
            first_df = load_data(uploaded_file)
            if first_df is not None:
                all_dfs.append(first_df)
                st.info(f"Loaded {uploaded_file.name} with {len(first_df)} rows.")
                break
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            skipped_files.append(uploaded_file.name)
            continue

    if first_df is None:
        return None, "No valid files were loaded."

    for uploaded_file in uploaded_files:
        if uploaded_file.name != uploaded_files[0].name:
            try:
                df = load_data(uploaded_file)
                if df is not None:
                    if not df.columns.equals(first_df.columns):
                        st.warning(f"Skipping {uploaded_file.name}: Schema mismatch.")
                        skipped_files.append(uploaded_file.name)
                        continue
                    all_dfs.append(df)
                    st.info(f"Loaded {uploaded_file.name} with {len(df)} rows.")
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")
                skipped_files.append(uploaded_file.name)

    if not all_dfs:
        return None, "No valid files were loaded."

    final_df = pd.concat(all_dfs, ignore_index=True)
    st.success(f"Successfully merged {len(all_dfs)} files into a single DataFrame with {len(final_df)} rows.")
    return final_df, skipped_files


def save_session_with_name(session_name, username):
    # This function is updated to use the username for data isolation
    if st.session_state.df is not None and session_name and username:
        user_dir = os.path.join(SESSION_STATE_DIR, username)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        filename = os.path.join(user_dir, f"{session_name}.json")
        try:
            # Get original dtypes for correct loading
            original_dtypes = {
                col: str(st.session_state.original_df[col].dtype)
                for col in st.session_state.original_df.columns
            }

            state_to_save = {
                "df": st.session_state.df.to_json(orient="split", date_format="iso"),
                "original_df": st.session_state.original_df.to_json(orient="split", date_format="iso"),
                "dataframes": {
                    key: df.to_json(orient="split", date_format="iso")
                    for key, df in st.session_state.dataframes.items()
                },
                "dtypes": original_dtypes,
                "active_df_key": st.session_state.active_df_key,
                "history": st.session_state.history,
                "macro_actions": st.session_state.macro_actions,
                "processed_files": st.session_state.processed_files,
                "chart_data": [
                    {
                        "fig_bytes": base64.b64encode(get_chart_bytes(data['fig'])).decode("utf-8"),
                        "insight": data['insight'],
                        "conclusion": data['conclusion'],
                    }
                    for data in st.session_state.chart_data
                ],
            }

            with open(filename, "w") as f:
                json.dump(state_to_save, f)

            # Save the session name to the database for the current user
            save_user_session(username, session_name)

            st.success(f"Session '{session_name}' saved successfully!")
        except Exception as e:
            st.error(f"Error saving session state: {e}")
    elif not session_name:
        st.warning("Please enter a name for the session.")
    else:
        st.warning("No dataset loaded to save.")


def load_session_by_name(session_name, username):
    # This function is updated to use the username for data isolation
    user_dir = os.path.join(SESSION_STATE_DIR, username)
    filename = os.path.join(user_dir, f"{session_name}.json")

    # First, verify the session exists for the user in the database
    if not load_user_session(username, session_name):
        st.warning(f"No saved session found with the name '{session_name}'.")
        return

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                state_loaded = json.load(f)

            # Load original df
            original_df_json = state_loaded.get("original_df", state_loaded["df"])
            st.session_state.original_df = pd.read_json(
                io.StringIO(original_df_json), orient="split", convert_dates=True
            )

            # Deserialize all DataFrames saved in the session
            st.session_state.dataframes = {
                key: pd.read_json(io.StringIO(df_json), orient="split", convert_dates=True)
                for key, df_json in state_loaded.get("dataframes", {}).items()
            }

            st.session_state.active_df_key = state_loaded.get("active_df_key", "Original Data")

            # Load the transformed DF as the current active DF
            current_df_json = state_loaded["df"]
            st.session_state.df = pd.read_json(
                io.StringIO(current_df_json), orient="split", convert_dates=True
            )

            # Ensure the active_df_key has the loaded DF
            if st.session_state.active_df_key not in st.session_state.dataframes:
                st.session_state.dataframes[st.session_state.active_df_key] = st.session_state.df.copy()
            else:
                # Update the DF in dataframes dict to match the current state loaded from "df" key
                st.session_state.dataframes[st.session_state.active_df_key] = st.session_state.df.copy()

            # Restore dtypes explicitly
            original_dtypes = state_loaded.get("dtypes", {})
            for key, df_to_fix in st.session_state.dataframes.items():
                for col, dtype in original_dtypes.items():
                    if col in df_to_fix.columns:
                        try:
                            if "datetime" in dtype:  # force datetime
                                df_to_fix[col] = pd.to_datetime(df_to_fix[col], errors="coerce",
                                                                infer_datetime_format=True)
                            elif "int" in dtype:
                                # Use 'Int64' (nullable integer) for robustness against NaN/nulls
                                df_to_fix[col] = pd.to_numeric(df_to_fix[col], errors="coerce").astype("Int64")
                            elif "float" in dtype:
                                df_to_fix[col] = pd.to_numeric(df_to_fix[col], errors="coerce")
                            elif "bool" in dtype:
                                df_to_fix[col] = df_to_fix[col].astype("bool")
                            else:  # keep as object/string
                                # Use pandas 'string' dtype for better performance/handling of NaNs in strings
                                df_to_fix[col] = df_to_fix[col].astype("string")
                        except Exception:
                            pass
                st.session_state.dataframes[key] = df_to_fix

            # Set the final active DataFrame with fixed dtypes
            st.session_state.df = st.session_state.dataframes[st.session_state.active_df_key].copy()

            # Restore metadata
            st.session_state.history = state_loaded.get("history", [])
            st.session_state.macro_actions = state_loaded.get("macro_actions", [])
            st.session_state.processed_files = state_loaded.get("processed_files", [])

            # Charts restore (Keep as is)
            st.session_state.chart_data = []
            for chart_item in state_loaded.get("chart_data", []):
                fig_bytes = base64.b64decode(chart_item["fig_bytes"])
                fig = plt.figure()
                plt.imshow(plt.imread(io.BytesIO(fig_bytes)))
                plt.axis("off")
                st.session_state.chart_data.append(
                    {
                        "fig": fig,
                        "insight": chart_item["insight"],
                        "conclusion": chart_item["conclusion"],
                    }
                )
                plt.close(fig)

            # --- CRITICAL FIX: Correctly initialize df_history to ensure transformations are present ---
            # Initialize all history lists with a copy of the *original* state (if needed for rollback to base)
            st.session_state.df_history = {
                key: [st.session_state.original_df.copy()]  # Use a safe base for other keys
                for key in st.session_state.dataframes.keys()
            }


            st.session_state.df_history[st.session_state.active_df_key] = []

            st.session_state.redo_history = []
            st.session_state.kpi_suggestions = None
            st.session_state.session_restored = True

            st.success(f"Session '{session_name}' loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading session state: {e}")
    else:
        st.warning(f"No saved session found with the name '{session_name}'.")


def delete_session_by_name(session_name, username):
    # This function is updated to use the username for data isolation
    user_dir = os.path.join(SESSION_STATE_DIR, username)
    filename = os.path.join(user_dir, f"{session_name}.json")

    # First, verify the session exists for the user in the database
    if not load_user_session(username, session_name):
        st.warning(f"No saved session found with the name '{session_name}'.")
        return

    if os.path.exists(filename):
        os.remove(filename)
        # Also remove from the database
        delete_user_session(username, session_name)
        st.success(f"Session '{session_name}' deleted successfully!")
        st.rerun()
    else:
        st.warning(f"No saved session found with the name '{session_name}'.")


def convert_condition_to_python(cond_str, col_name):
    cond_str = cond_str.replace("and", "&").replace("or", "|")
    pattern = r'([<>!=]=?|==)\s*([\d\.]+)'
    cond_str = re.sub(pattern, rf"(df['{col_name}'].astype(float) \1 \2)", cond_str)
    return cond_str


def generate_pdf_report(df, history, kpi_suggestions, chart_data, conclusion):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Data Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"**Report Date:** {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
    story.append(Paragraph(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns", styles['Normal']))
    story.append(Spacer(1, 24))

    story.append(Paragraph("AI-Powered Key Performance Indicator (KPI) Insights", styles['h2']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(kpi_suggestions, styles['Normal']))
    story.append(Spacer(1, 24))

    story.append(Paragraph("Conclusion and Recommendations for Stakeholders", styles['h2']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(conclusion, styles['Normal']))
    story.append(Spacer(1, 24))

    story.append(Paragraph("Data Cleaning & Transformation History", styles['h2']))
    story.append(Spacer(1, 12))
    for entry in history:
        story.append(Paragraph(entry, styles['Normal']))
        story.append(Spacer(1, 6))
    story.append(Spacer(1, 24))

    story.append(Paragraph("Generated Visualizations", styles['h2']))
    story.append(Spacer(1, 12))
    for i, data in enumerate(chart_data):
        fig = data['fig']
        insight = data.get('insight', 'No insight provided.')
        conclusion = data.get('conclusion', 'No conclusion provided.')
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img = Image(img_buffer, width=450, height=300)
        story.append(img)
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"**Insight:** {insight}", styles['Normal']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"**Conclusion:** {conclusion}", styles['Normal']))
        story.append(Spacer(1, 18))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# Locate and modify the load_to_mysql function:

def load_to_mysql(df, host, user, password, database, table_name, if_exists='replace'):
    try:

        encoded_password = urllib.parse.quote_plus(password)

        # --- STEP 1: Connect to the server WITHOUT specifying the database ---
        # Note the use of encoded_password in the connection string
        engine_no_db = create_engine(f'mysql+mysqlconnector://{user}:{encoded_password}@{host}/')

        # --- STEP 2: Create Database if it doesn't exist ---
        # Note: Need 'from sqlalchemy import text' at the top of app.py
        from sqlalchemy import text
        with engine_no_db.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {database}"))
            conn.commit()

            # --- STEP 3: Connect to the database and load data ---
        engine = create_engine(f'mysql+mysqlconnector://{user}:{encoded_password}@{host}/{database}')

        # Load the data
        df.to_sql(name=table_name, con=engine, if_exists=if_exists, index=False)
        return True, f"Successfully created database '{database}' (if needed) and loaded {len(df)} rows to MySQL table '{table_name}'."
    except Exception as e:
        return False, f"MySQL Load Error: {e}. Ensure the host '{host}' is correct and the user '{user}' has sufficient privileges."


# Locate and modify the load_to_snowflake function:

def load_to_snowflake(df, user, password, account, warehouse, database, schema, table_name, if_exists='replace'):
    # --- FIX: Create a mutable copy and handle problematic dtypes before load ---
    df_load = df.copy()

    # Identify non-numeric columns that might have been incorrectly treated as numeric during schema inference
    # (e.g., columns containing car models, which often fail due to Snowflake's strictness).
    # We force 'object' (string) type for known string columns that might contain mixed types.

    # Based on the error, force 'model' and any column that should be text to be object/string.
    for col in df_load.columns:
        # Check if the column is currently a string/object but contains values that might confuse schema inference,
        # or if it's one of the columns known to cause trouble (like 'model').
        if col.lower() in ['model', 'body', 'car', 'registration', 'drive']:
            df_load[col] = df_load[col].astype(str)
        # Handle cases where nulls cause integer columns to become floats and then fail
        elif pd.api.types.is_float_dtype(df_load[col]):
            # Use nullable integer ('Int64') or round to clean up numbers
            pass  # Allowing floats is usually okay unless precision is an issue

    try:
        # --- (Existing password encoding remains here) ---
        import urllib.parse
        encoded_password = urllib.parse.quote_plus(password)

        # Configure the Snowflake connection URL (use standard 'SNOWFLAKE' database for initial connection)
        snowflake_url_initial = URL(
            user=user, password=encoded_password, account=account, warehouse=warehouse,
            database="SNOWFLAKE",
            schema="INFORMATION_SCHEMA"
        )

        engine_initial = create_engine(snowflake_url_initial)

        #  STEP 1: Connect and create Database/Schema if they don't exist
        from sqlalchemy import text
        with engine_initial.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {database}"))
            conn.execute(text(f"USE DATABASE {database}"))
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            conn.commit()

        #  STEP 2: Connect to the newly created database and load data
        snowflake_url_final = URL(
            user=user, password=encoded_password, account=account, warehouse=warehouse,
            database=database, schema=schema
        )
        engine_final = create_engine(snowflake_url_final)

        # Load the modified DataFrame (df_load)
        with engine_final.connect() as conn_final:
            # We use 'replace' here, so the schema is correctly re-inferred as VARCHAR
            df_load.to_sql(table_name, con=conn_final, if_exists=if_exists, index=False)

        return True, f"Successfully created database/schema (if needed) and loaded {len(df_load)} rows to Snowflake table '{table_name}'."

    except Exception as e:
        return False, f"Snowflake Load Error: {e}. Please check the data types in your DataFrame and target table."


# Session State Initialization

if "df" not in st.session_state:
    st.session_state.df = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "history" not in st.session_state:
    st.session_state.history = []
if "df_history" not in st.session_state:
    st.session_state.df_history = {"Original Data": []}
if "chart_data" not in st.session_state:
    st.session_state.chart_data = []
if "viz_params" not in st.session_state:
    st.session_state.viz_params = {'chart_type': None, 'x_col': None, 'y_col': None}
if "temp_fill_value" not in st.session_state:
    st.session_state.temp_fill_value = None
if "kpi_suggestions" not in st.session_state:
    st.session_state.kpi_suggestions = None
if "conclusion_text" not in st.session_state:
    st.session_state.conclusion_text = None
if "macro_actions" not in st.session_state:
    st.session_state.macro_actions = []
if "macro_file_data" not in st.session_state:
    st.session_state.macro_file_data = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'last_uploaded_files' not in st.session_state:
    st.session_state.last_uploaded_files = None
if 'outlier_col_selected' not in st.session_state:
    st.session_state.outlier_col_selected = None
if 'outliers_found' not in st.session_state:
    st.session_state.outliers_found = False
if 'outlier_df' not in st.session_state:
    st.session_state.outlier_df = pd.DataFrame()
if 'outlier_count' not in st.session_state:
    st.session_state.outlier_count = 0
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'session_restored' not in st.session_state:
    st.session_state.session_restored = False
if 'incremental_fill_option' not in st.session_state:
    st.session_state.incremental_fill_option = 'Recalculate on Entire Dataset'
if 'ai_code' not in st.session_state:
    st.session_state.ai_code = ""
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
if 'active_df_key' not in st.session_state:
    st.session_state.active_df_key = "Original Data"
if 'last_ai_code' not in st.session_state:
    st.session_state.last_ai_code = None
# --- ADDED FOR AI CHAT FEATURE ---
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = None
if 'show_code' not in st.session_state:
    st.session_state.show_code = False
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
# ------------------------------------
if 'aws_access_key_id' not in st.session_state:
    st.session_state.aws_access_key_id = ''
if 'aws_secret_access_key' not in st.session_state:
    st.session_state.aws_secret_access_key = ''
if 'aws_region' not in st.session_state:
    st.session_state.aws_region = 'us-east-1'
if 'mysql_host' not in st.session_state:
    st.session_state.mysql_host = ''
if 'mysql_user' not in st.session_state:
    st.session_state.mysql_user = ''
if 'mysql_password' not in st.session_state:
    st.session_state.mysql_password = ''
if 'mysql_database' not in st.session_state:
    st.session_state.mysql_database = ''
if 'gdrive_credentials_ok' not in st.session_state:
    st.session_state.gdrive_credentials_ok = False
if 'preview_df' not in st.session_state:
    st.session_state.preview_df = None
if 'preview_mode' not in st.session_state:
    st.session_state.preview_mode = False
if 'final_join_params' not in st.session_state:
    st.session_state.final_join_params = None

# --- NEW: Session State for Authentication ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
# Add this to your main Session State Initialization block:
if 'last_sql_code' not in st.session_state:
    st.session_state.last_sql_code = ""
if 'show_sql_code' not in st.session_state:
    st.session_state.show_sql_code = False


# Authentication and Main UI Control

if not st.session_state.logged_in:
    st.title("Welcome to the Data Analysis App")
    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        st.subheader("Login to Your Account")
        st.markdown("Enter your username and password to log in to your account.")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if login_username and login_password:
                success, message = verify_and_login(login_username, login_password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please enter both a username and password.")

    with signup_tab:
        st.subheader("Create a New Account")
        st.markdown("Sign up to create your user account and begin your data analysis journey.")
        signup_username = st.text_input("Choose a Username", key="signup_username")
        signup_email = st.text_input("Email Address", key="signup_email")
        signup_password = st.text_input("Create a Password", type="password", key="signup_password")

        if st.button("Create Account"):
            if signup_username and signup_email and signup_password:
                success, message = signup_user(signup_username, signup_email, signup_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.error("Please fill in all fields.")

else:
    # --- LOGGED-IN VIEW ---
    st.sidebar.title(f"Hello, {st.session_state.username}!")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.df = None
        st.session_state.original_df = None
        st.session_state.history = []
        st.session_state.macro_actions = []
        st.session_state.dataframes = {}
        st.session_state.active_df_key = "Original Data"
        st.session_state.chart_data = []
        st.session_state.kpi_suggestions = None
        st.session_state.conclusion_text = None
        st.session_state.processed_files = []
        st.info("You have been logged out.")
        st.rerun()

    # File Upload and Session Management

    st.sidebar.title("Data Upload")
    st.sidebar.markdown("Choose a data source to upload a dataset for analysis.")
    if st.session_state.df is not None:
        if st.sidebar.button("Undo Last Action"):
            if st.session_state.df_history[st.session_state.active_df_key]:
                st.session_state.df = st.session_state.df_history[st.session_state.active_df_key].pop()
                st.session_state.history.pop()
                st.session_state.macro_actions.pop()
                st.success("Undo successful. Reverted to previous state.")
                st.rerun()
        else:
            st.sidebar.button("Undo Last Action", disabled=True, help="No actions to undo.")

    if st.session_state.df is not None:
        with st.sidebar.expander("Data Preview", expanded=False):
            st.markdown("See a quick glimpse of your loaded data here.")
            st.dataframe(st.session_state.df.head())

    upload_mode = st.sidebar.radio(
        "Choose Upload Mode:",
        ("Local Upload", "Folder (Merge Files)", "AWS S3", "Google Drive", "MySQL Database"),
        key="upload_mode_radio"
    )

    df = None
    uploaded_file = None
    uploaded_files = []

    if upload_mode == "Local Upload":
        st.sidebar.markdown(
            "Upload a single file from your local machine. Supported formats: CSV, Excel, Parquet, JSON, XML, ORC.")
        uploaded_file = st.sidebar.file_uploader(
            "Upload a single dataset",
            type=["csv", "xlsx", "xls", "parquet", "json", "xml", "orc"]
        )
    elif upload_mode == "Folder (Merge Files)":
        st.sidebar.markdown(
            "Upload multiple files to merge them into a single dataset. All files must have the same column structure.")
        uploaded_files = st.sidebar.file_uploader(
            "Upload multiple files to merge (same schema)",
            type=["csv", "xlsx", "xls", "json", "xml"],
            accept_multiple_files=True
        )
    elif upload_mode == "AWS S3":
        st.sidebar.markdown("---")
        st.sidebar.subheader("S3 Configuration")
        st.sidebar.markdown(
            "Connect to your AWS S3 bucket to load data directly. Make sure your credentials are correct and the file key points to a valid file.")
        aws_access_key_id_input = st.sidebar.text_input("AWS Access Key ID", type="password",
                                                        value=st.session_state.aws_access_key_id)
        aws_secret_access_key_input = st.sidebar.text_input("AWS Secret Access Key", type="password",
                                                            value=st.session_state.aws_secret_access_key)
        aws_region_input = st.sidebar.text_input("AWS Region", value=st.session_state.aws_region)
        bucket_name = st.sidebar.text_input("S3 Bucket Name")
        file_key = st.sidebar.text_input("S3 File Key (e.g., 'data/my_file.csv')")
        s3_file_format = st.sidebar.selectbox("File Format", ["csv", "xlsx", "json", "parquet", "xml"])
        if st.sidebar.button("Load from S3"):
            if bucket_name and file_key and aws_access_key_id_input and aws_secret_access_key_input and aws_region_input:
                st.session_state.aws_access_key_id = aws_access_key_id_input
                st.session_state.aws_secret_access_key = aws_secret_access_key_input
                st.session_state.aws_region = aws_region_input
                with st.spinner('Loading data from S3...'):
                    df = load_data_from_s3(bucket_name, file_key, s3_file_format, st.session_state.aws_access_key_id,
                                           st.session_state.aws_secret_access_key, st.session_state.aws_region)
                    if df is not None and not df.empty:
                        st.session_state.df = df
                        st.session_state.original_df = df.copy()
                        st.session_state.dataframes["Original Data"] = st.session_state.original_df.copy()
                        st.session_state.active_df_key = "Original Data"
                        st.session_state.history = [
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded file from S3: {file_key}"
                        ]
                        st.session_state.df_history["Original Data"] = [df.copy()]
                        st.rerun()
                    else:
                        st.error("Failed to load data from S3. Please check your credentials and file path.")
            else:
                st.sidebar.warning("Please enter all S3 credentials and file details.")

    elif upload_mode == "Google Drive":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Google Drive/Sheets Configuration")
        st.info(
            "Upload your service account key file (`credentials.json`) and share the sheet with the service account email.")
        gdrive_key_file = st.sidebar.file_uploader("Upload your credentials.json file", type="json")
        gdrive_sheet_id = st.sidebar.text_input("Google Sheet File ID (from the URL)")
        if st.sidebar.button("Load from Google Drive"):
            if gdrive_key_file and gdrive_sheet_id:
                try:
                    credentials_data = json.load(gdrive_key_file)
                    st.session_state.gdrive_credentials_ok = True
                    with st.spinner('Loading data from Google Drive...'):
                        df = load_data_from_gdrive(credentials_data, gdrive_sheet_id)
                        if df is not None and not df.empty:
                            st.session_state.df = df
                            st.session_state.original_df = df.copy()
                            st.session_state.dataframes["Original Data"] = st.session_state.original_df.copy()
                            st.session_state.active_df_key = "Original Data"
                            st.session_state.history = [
                                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded data from Google Sheet ID: {gdrive_sheet_id}"
                            ]
                            st.session_state.df_history["Original Data"] = [df.copy()]
                            st.rerun()
                        else:
                            st.error("Failed to load data from Google Drive. Check file ID and permissions.")
                except Exception as e:
                    st.error(f"Error with Google Drive credentials: {e}")
            else:
                st.sidebar.warning("Please upload the credentials file and enter the Sheet ID.")

    elif upload_mode == "MySQL Database":
        st.sidebar.markdown("---")
        st.sidebar.subheader("MySQL Configuration")
        st.sidebar.markdown("Connect to your MySQL database to load data using a custom SQL query.")
        mysql_host_input = st.sidebar.text_input("Host", value=st.session_state.mysql_host)
        mysql_user_input = st.sidebar.text_input("User", value=st.session_state.mysql_user)
        mysql_password_input = st.sidebar.text_input("Password", type="password", value=st.session_state.mysql_password)
        mysql_database_input = st.sidebar.text_input("Database Name", value=st.session_state.mysql_database)
        mysql_query_input = st.sidebar.text_area("SQL Query", "SELECT * FROM your_table;", height=150)
        if st.sidebar.button("Load from MySQL"):
            if all([mysql_host_input, mysql_user_input, mysql_database_input, mysql_query_input]):
                st.session_state.mysql_host = mysql_host_input
                st.session_state.mysql_user = mysql_user_input
                st.session_state.mysql_password = mysql_password_input
                st.session_state.mysql_database = mysql_database_input
                with st.spinner('Loading data from MySQL...'):
                    df = load_data_from_mysql(
                        st.session_state.mysql_host,
                        st.session_state.mysql_user,
                        st.session_state.mysql_password,
                        st.session_state.mysql_database,
                        mysql_query_input
                    )
                    if df is not None and not df.empty:
                        st.session_state.df = df
                        st.session_state.original_df = df.copy()
                        st.session_state.dataframes["Original Data"] = st.session_state.original_df.copy()
                        st.session_state.active_df_key = "Original Data"
                        st.session_state.history = [
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded data from MySQL table using a custom query."
                        ]
                        st.session_state.df_history["Original Data"] = [df.copy()]
                        st.rerun()
                    else:
                        st.error("Failed to load data from MySQL. Please check your credentials and query.")
            else:
                st.sidebar.warning("Please enter all MySQL connection details and a query.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Session Management")
    st.sidebar.markdown(
        "Save your current work or load a previously saved session. Your sessions are private and linked to your account.")
    session_name_to_save = st.sidebar.text_input("Enter session name to save:")
    if st.sidebar.button("Save Current Session"):
        save_session_with_name(session_name_to_save, st.session_state.username)

    saved_sessions = get_user_saved_sessions(st.session_state.username)
    selected_session = st.sidebar.selectbox("Select session to load/delete:", ["None"] + saved_sessions)
    load_col, delete_col = st.sidebar.columns(2)
    with load_col:
        if st.button("Load Selected Session"):
            if selected_session and selected_session != "None":
                load_session_by_name(selected_session, st.session_state.username)
            else:
                st.warning("Please select a session to load.")
    with delete_col:
        if st.button("Delete Selected Session"):
            if selected_session and selected_session != "None":
                delete_session_by_name(selected_session, st.session_state.username)
            else:
                st.warning("Please select a session to delete.")

    # New DataFrame Management UI

    st.sidebar.markdown("---")
    st.sidebar.subheader(" Manage DataFrames")

    # Creation Section
    with st.sidebar.expander("Create New DataFrame", expanded=False):
        if st.session_state.df is not None:
            df_name = st.text_input("New DataFrame Name:", key="new_df_name")
            creation_method = st.radio("Creation Method:", ["From Current View", "From Filter", "From Grouped Data"])

            new_df_to_create = None

            # --- FROM FILTER LOGIC ---
            if creation_method == "From Filter":
                filter_col = st.selectbox("Filter Column", ["None"] + list(st.session_state.df.columns),
                                          key="filter_col_df_creation")
                if filter_col != "None":
                    filter_operator = st.selectbox("Operator", ["==", "!=", ">", "<", ">=", "<=", "contains"],
                                                   key="filter_op_df_creation")
                    filter_value = st.text_input("Value", key="filter_val_df_creation")

                if st.button("Create Filtered DF"):
                    if df_name and filter_col != "None" and filter_value:
                        try:
                            # Logic to apply the filter and create the new df
                            if filter_operator == "contains":
                                new_df_to_create = st.session_state.df[
                                    st.session_state.df[filter_col].astype(str).str.contains(filter_value, case=False,
                                                                                             na=False)]
                            else:
                                # Use query for robust numeric/string filtering
                                if pd.api.types.is_numeric_dtype(st.session_state.df[filter_col]):
                                    expression = f"`{filter_col}` {filter_operator} {filter_value}"
                                else:
                                    expression = f"`{filter_col}` {filter_operator} '{filter_value}'"
                                new_df_to_create = st.session_state.df.query(expression)
                        except Exception as e:
                            st.error(f"Error applying filter: {e}")
                    else:
                        st.warning("Please fill in name, column, and value.")

            # --- FROM GROUPED DATA LOGIC ---
            elif creation_method == "From Grouped Data":
                group_cols_df = st.multiselect("Group by Columns", st.session_state.df.columns, key="groupby_cols_df")
                agg_col_df = st.selectbox("Aggregation Column",
                                          ["None"] + list(st.session_state.df.select_dtypes(include='number').columns),
                                          key="agg_col_df")
                agg_func_df = st.selectbox("Aggregation Function", ["count", "sum", "mean", "median", "min", "max"],
                                           key="agg_func_df")
                if st.button("Create Grouped DF"):
                    if df_name and group_cols_df and agg_col_df != "None" and agg_func_df:
                        try:
                            new_df_to_create = st.session_state.df.groupby(group_cols_df)[agg_col_df].agg(
                                agg_func_df).reset_index()
                        except Exception as e:
                            st.error(f"Error creating grouped data: {e}")
                    else:
                        st.warning("Please select group by columns, aggregation column, and a function.")

            # --- FROM CURRENT VIEW LOGIC ---
            else:  # From Current View
                if st.button("Create Copy of Current View"):
                    if df_name:
                        new_df_to_create = st.session_state.df.copy()
                    else:
                        st.warning("Please enter a name for the new DataFrame.")

            # --- COMMIT NEW DF TO SESSION STATE ---
            if new_df_to_create is not None and df_name:
                if df_name in st.session_state.dataframes:
                    st.error(f"A DataFrame named '{df_name}' already exists. Choose a different name.")
                else:
                    st.session_state.dataframes[df_name] = new_df_to_create
                    st.session_state.df_history[df_name] = [new_df_to_create.copy()]
                    st.session_state.history.append(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Created a new DataFrame: '{df_name}'.")
                    st.success(f"✅ DataFrame '{df_name}' created and set as active!")
                    st.session_state.df = new_df_to_create.copy()
                    st.session_state.active_df_key = df_name
                    st.rerun()
        else:
            st.info("Upload data first to create derivative DataFrames.")

    # Switching Section
    st.sidebar.markdown("#### Active DataFrames")

    if st.session_state.df is not None:
        current_active_df = st.session_state.active_df_key
        st.sidebar.success(f"Active: **{current_active_df}** ({len(st.session_state.df)} rows)")

        # Generate switch buttons for all non-active dataframes
        for key in st.session_state.dataframes:
            if key != current_active_df:
                if st.sidebar.button(f"Switch to '{key}'"):
                    st.session_state.df = st.session_state.dataframes[key].copy()
                    st.session_state.active_df_key = key

                    # Ensure a history list exists for the newly active DF
                    if key not in st.session_state.df_history:
                        st.session_state.df_history[key] = [st.session_state.df.copy()]

                    st.session_state.history.append(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Switched to DataFrame '{key}'.")
                    st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Automate Your Workflow")
    st.sidebar.markdown(
        "Load a macro script to automatically apply a sequence of data transformations to your current dataset.")
    macro_file = st.sidebar.file_uploader("Load Macro Script", type=["json"])
    if macro_file:
        st.session_state.macro_file_data = json.load(macro_file)

    if st.sidebar.button("Run Macro"):
        if st.session_state.df is None:
            st.error("Please upload a dataset or load a session first.")
        elif st.session_state.macro_file_data is None:
            st.error("Please load a macro script first.")
        else:
            st.info("Running macro on the current dataset...")
            new_df, macro_log, success = run_macro(st.session_state.df, st.session_state.macro_file_data)
            if success:
                st.session_state.df = new_df
                st.session_state.df_history[st.session_state.active_df_key].append(new_df.copy())
                st.session_state.history.extend(macro_log)
                st.session_state.macro_actions.extend(st.session_state.macro_file_data)
                st.success("Macro executed successfully! The dataset has been transformed.")
                st.rerun()
            else:
                st.error("Macro execution halted due to an error.")
                st.sidebar.subheader("Macro Execution Log")
                for entry in macro_log:
                    st.sidebar.write(entry)
                st.stop()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Incremental Data Loading")
    st.sidebar.markdown(
        "Merge new files with your existing dataset and re-apply transformations to handle new data efficiently.")
    incremental_files = st.sidebar.file_uploader(
        "Upload new files to merge with existing data",
        type=["csv", "xlsx", "xls", "json", "xml"],
        accept_multiple_files=True
    )
    st.session_state.incremental_fill_option = st.sidebar.radio(
        "How to handle fill values?",
        ('Recalculate on Entire Dataset', 'Apply Previous Statistics to New Data'),
        help="Recalculate: Recalculates mean/median/mode from the combined dataset. Apply Previous: Uses the values from the last session for new data.",
        key="incremental_fill_option_radio"
    )

    if st.sidebar.button("Merge New Files"):
        if st.session_state.df is None:
            st.error("Please upload an initial dataset or load a session first to enable incremental loading.")
        elif not incremental_files:
            st.warning("Please select at least one new file to merge.")
        else:
            st.info("Merging new files with the existing dataset...")

            # --- 1. DEFINE MACRO ACTION CATEGORIES FOR 3-STEP FLOW ---
            column_level_initial_actions = [
                'remove_columns', 'split_column', 'change_dtype', 'merge_columns',
                'create_column_date_part', 'create_column_date_add_subtract'
            ]
            row_level_cleaning_actions = [
                'remove_duplicates', 'fill_nulls', 'remove_rows_by_condition',
                'remove_outliers'
            ]
            dependent_column_actions = [
                'create_column_arithmetic',
                'create_column_single_arithmetic',
                'create_column',  # Conditional column creation
            ]

            # Filter saved macro actions into the three lists
            column_macro_initial = [action for action in st.session_state.macro_actions if
                                    action['action'] in column_level_initial_actions]
            row_macro_cleaning = [action for action in st.session_state.macro_actions if
                                  action['action'] in row_level_cleaning_actions]
            column_macro_dependent = [action for action in st.session_state.macro_actions if
                                      action['action'] in dependent_column_actions]

            # Combine cleaning and recalculation for the final macro execution
            full_row_recalc_macro = row_macro_cleaning + column_macro_dependent


            new_dfs = []
            new_files_to_process = []
            skipped_files = []
            current_processed_files = st.session_state.get('processed_files', [])
            target_columns = st.session_state.df.columns.tolist()  # Schema of the existing master DF

            # --- FILE PROCESSING LOOP (STEP 1: Initial Column Alignment) ---
            for f in incremental_files:
                if f.name not in current_processed_files:
                    try:
                        new_df = load_data(f)
                        if new_df is not None:
                            #  STEP 1: Initial Column Macro (Schema Alignment)
                            temp_df_step1, macro_log_step1, success_step1 = run_macro(new_df, column_macro_initial)

                            # Add Dependent columns (as NULLs) to the new DF to align schema before merge.
                            temp_df_schema = temp_df_step1
                            for action in column_macro_dependent:
                                col_name = action['params'].get('new_col') or action['params'].get('new_name')
                                if col_name and col_name not in temp_df_schema.columns:
                                    temp_df_schema[col_name] = pd.NA

                            # Check final schema consistency with the master
                            is_schema_consistent = success_step1 and (
                                        set(temp_df_schema.columns) == set(target_columns))

                            if is_schema_consistent:
                                # Enforce the exact target column order before merging (CRUCIAL)
                                temp_df_schema = temp_df_schema.reindex(columns=target_columns)

                                new_dfs.append(temp_df_schema)
                                new_files_to_process.append(f.name)
                                st.session_state.history.extend(macro_log_step1)

                            else:
                                st.error(f"Schema mismatch for {f.name} after initial alignment. Skipping file.")
                                skipped_files.append(f.name)

                    except Exception as e:
                        st.error(f"Error loading or transforming incremental file {f.name}: {e}")
                        skipped_files.append(f.name)

            # --- MERGING AND FINAL RECALCULATION (STEPS 2 & 3) ---
            if new_dfs:
                # 1. Merge the existing master DF with the newly processed DFs
                combined_df = pd.concat([st.session_state.df] + new_dfs, ignore_index=True)
                final_df = combined_df

                #  STEP 2 & 3: Run Row Cleaning AND Dependent Column Recalculation on the Merged Data
                st.info("Applying cleaning and dependent calculations to the merged dataset...")
                final_df, row_macro_log, success = run_macro(combined_df, full_row_recalc_macro)

                if success:
                    st.session_state.df = final_df
                    st.success(
                        f"Successfully merged {len(new_dfs)} new files and applied full transformation workflow.")
                    st.session_state.history.extend(row_macro_log)
                    st.session_state.history.append(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Merged and processed new files: {', '.join(new_files_to_process)}"
                    )
                    st.session_state.processed_files.extend(new_files_to_process)
                    st.session_state.original_df = st.session_state.df.copy()
                    st.session_state.df_history[st.session_state.active_df_key].append(st.session_state.df.copy())
                    st.rerun()
                else:
                    st.error("Error applying combined row-cleaning and dependent column transformations after merge.")
                    # You can add a detailed log display here for debugging if needed

            else:
                st.warning("No new files were successfully loaded or merged.")
                if skipped_files:
                    st.warning(f"Skipped files: {', '.join(skipped_files)}")


    if st.session_state.df is None and (uploaded_file or uploaded_files):
        if uploaded_file:
            if uploaded_file != st.session_state.get('last_uploaded_file'):
                st.session_state.df = None
                st.session_state.last_uploaded_file = uploaded_file
                st.session_state.last_uploaded_files = None
        elif uploaded_files:
            if uploaded_files != st.session_state.get('last_uploaded_files'):
                st.session_state.df = None
                st.session_state.last_uploaded_files = uploaded_files
                st.session_state.last_uploaded_file = None

        if st.session_state.df is None:
            if uploaded_file:
                try:
                    st.session_state.df = load_data(uploaded_file)
                    if st.session_state.df is not None:
                        st.session_state.original_df = st.session_state.df.copy()
                        st.session_state.dataframes["Original Data"] = st.session_state.original_df.copy()
                        st.session_state.active_df_key = "Original Data"
                        st.session_state.history = [
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Uploaded single file: {uploaded_file.name}"]
                        st.session_state.processed_files = [uploaded_file.name]
                        st.session_state.df_history[st.session_state.active_df_key] = [st.session_state.df.copy()]
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to load the file. Error: {e}")
                    st.session_state.df = None
            elif uploaded_files:
                merged_df, skipped = load_and_merge_data(uploaded_files)
                if merged_df is not None:
                    st.session_state.df = merged_df
                    st.session_state.original_df = merged_df.copy()
                    st.session_state.dataframes["Original Data"] = st.session_state.original_df.copy()
                    st.session_state.active_df_key = "Original Data"
                    st.session_state.history = [
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Merged {len(merged_df)} rows from {len(uploaded_files)} files."
                    ]
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    if skipped:
                        st.session_state.history.append(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipped files due to errors: {', '.join(skipped)}")
                    st.session_state.df_history[st.session_state.active_df_key] = [st.session_state.df.copy()]
                    st.rerun()

    if st.session_state.session_restored:
        st.subheader("Session Restored")
        st.info(
            f"Successfully loaded previous session with **{len(st.session_state.df)} rows** and **{len(st.session_state.df.columns)} columns**.")
        st.write("---")
        st.write(" Restored Data Preview")
        st.dataframe(st.session_state.df.head())
        st.write(" Restored History Log")
        for log in reversed(st.session_state.history):
            st.write(f"- {log}")
        st.session_state.session_restored = False

    if st.session_state.df is not None:
        df = st.session_state.df

        st.markdown("<h2 style='text-align: center;'>Understanding the Data</h2>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### Dataset Overview")
        st.markdown(
            "This section provides a summary of your data, including its size, column types, and a statistical overview.")
        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

        st.write("### Data Preview")
        with st.expander("Click to view top 5 rows"):
            st.dataframe(df.head())

        st.write("### Data Types of Each Column")
        st.markdown(
            "Verify the data type of each column. Changing data types is useful for performing the correct operations, like calculations on numbers or date-based analysis.")
        st.dataframe(df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'DataType'}))
        st.write("### Data Types of Each Column")
        st.markdown(
            "Verify the data type of each column. Changing data types is useful for performing the correct operations, like calculations on numbers or date-based analysis.")
        st.dataframe(df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'DataType'}))

        # THE NULL VALUES CODE SNIPPET
        st.write("### Null Values Per Column")
        st.markdown("This table shows the count of missing (null/NaN) values for each column.")
        null_counts = df.isnull().sum()
        st.dataframe(null_counts[null_counts > 0].to_frame(name="Null Count"))
        if null_counts.sum() == 0:
            st.info("No null values found in the dataset. Data quality is excellent! ")
        #END OF NULL VALUES CODE SNIPPET

        st.write("Statistical Summary")
        st.markdown(
            "Get a quick statistical summary of your numerical columns (mean, min, max, etc.) and a count of unique values for all columns.")
        st.write(df.describe(include="all").transpose())

        st.write("Statistical Summary")
        st.markdown(
            "Get a quick statistical summary of your numerical columns (mean, min, max, etc.) and a count of unique values for all columns.")
        st.write(df.describe(include="all").transpose())

        st.write("### Distinct Values")
        st.markdown(
            "Explore the unique values within each column to identify potential data entry issues or categories.")
        for col in df.columns:
            with st.expander(f"Show distinct values for '{col}'"):
                st.write(f"*{df[col].nunique()} unique values*")
                st.write(df[col].unique())

        st.write("Sort Dataset")
        st.markdown(
            "Rearrange your data based on the values in a specific column, either in ascending or descending order.")
        col_to_sort = st.selectbox("Select column to sort by", ["None"] + list(df.columns))
        sort_order = st.radio("Order", ["Ascending", "Descending"], key="sort_order_main")
        if st.button("Sort Dataset"):
            if col_to_sort != "None":
                ascending = sort_order == "Ascending"
                perform_action(
                    lambda d: d.sort_values(by=col_to_sort, ascending=ascending),
                    f"Sorted dataset by '{col_to_sort}' in {sort_order.lower()} order.",
                    {"action": "sort", "params": {"col": col_to_sort, "ascending": ascending}}
                )
            else:
                st.warning("Please select a column to sort by.")

        st.markdown("---")

        # Custom Aggregation

        st.subheader("Custom Aggregation")
        st.markdown(
            "Analyze the frequency of values, optionally grouped by other columns. "
            "Use Value Counts to see frequency distributions, or Aggregate to compute numeric aggregations."
        )

        #  DYNAMIC UPDATE
        # The code now directly uses the main DataFrame from your session state (st.session_state.df).
        # This is the currently active DataFrame in your app.
        df_for_agg = st.session_state.df


        agg_type = st.radio(
            "Choose analysis type:",
            ["Value Counts", "Aggregate"],
            key="agg_mode",
            horizontal=True
        )

        aggregation_mapping = {}

        if agg_type == "Aggregate":
            st.info("In Aggregate mode, you must select numeric columns for calculation.")
            numeric_cols = df_for_agg.select_dtypes(include=np.number).columns.tolist()

            target_cols_agg = st.multiselect(
                "Select numeric target column(s) for aggregation:",
                options=numeric_cols,
                key="agg_target"
            )
            groupby_cols_agg = st.multiselect(
                "Select columns to group by (optional):",
                options=df_for_agg.columns.tolist(),
                key="agg_groupby"
            )

            if target_cols_agg:
                agg_funcs = st.multiselect(
                    "Select aggregation function(s):",
                    ["mean", "median", "sum", "min", "max", "count", "std", "var"],
                    default=["mean"],
                    key="agg_func_select"
                )
                if agg_funcs:
                    for col in target_cols_agg:
                        aggregation_mapping[col] = agg_funcs

        else:  # Value Counts
            target_cols_agg = st.multiselect(
                "Select target column(s) to count (optional):",
                options=df_for_agg.columns.tolist(),
                key="agg_target_vc"
            )
            groupby_cols_agg = st.multiselect(
                "Select columns to group by (optional):",
                options=df_for_agg.columns.tolist(),
                key="agg_groupby_vc"
            )

        # Your Sorting and Button Logic
        sort_order_agg = st.radio("Sort order:", ["Descending", "Ascending"], key="sort_order_agg", horizontal=True)

        sort_by_options = []
        if groupby_cols_agg:
            sort_by_options.extend(groupby_cols_agg)
        if agg_type == "Value Counts":
            sort_by_options.append("count")
        elif agg_type == "Aggregate" and aggregation_mapping:
            for col, funcs in aggregation_mapping.items():
                sort_by_options.extend([f"{col}_{func}" for func in funcs])

        sort_by_col_agg = st.selectbox("Sort result by:", ["None"] + sort_by_options, key="sort_by_col_agg")

        if st.button("Apply Analysis", use_container_width=True, type="primary"):
            if not target_cols_agg and not groupby_cols_agg:
                st.warning("Please select at least one target or group-by column.")
            else:
                try:
                    agg_result = None
                    if agg_type == "Value Counts":
                        cols_to_group = groupby_cols_agg + target_cols_agg
                        if not cols_to_group: st.warning("Please select columns to count."); st.stop()
                        agg_result = df_for_agg.groupby(cols_to_group).size().reset_index(name="count")
                    else:  # Aggregate
                        if not aggregation_mapping: st.warning("Please select a target and a function."); st.stop()
                        if groupby_cols_agg:
                            agg_result = df_for_agg.groupby(groupby_cols_agg).agg(aggregation_mapping)
                        else:
                            agg_result = df_for_agg[list(aggregation_mapping.keys())].agg(aggregation_mapping)
                        agg_result = agg_result.reset_index()

                    if agg_result is not None:
                        if isinstance(agg_result.columns, pd.MultiIndex):
                            agg_result.columns = ["_".join(map(str, col)).strip() for col in
                                                  agg_result.columns.values]

                        if sort_by_col_agg != "None" and sort_by_col_agg in agg_result.columns:
                            agg_result = agg_result.sort_values(by=sort_by_col_agg,
                                                                ascending=(sort_order_agg == "Ascending"))

                        st.write("#### Analysis Result:")
                        st.dataframe(agg_result, use_container_width=True)

                        # --- Your AI Conclusion Logic can be placed here if needed ---

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")


        st.markdown("---")

        # Joining Datasets

        import re






        st.markdown("---")
        st.write("### 🔗 Joining Datasets")
        st.markdown(
            "Upload a second dataset. The process is two-step: **Preview** the joined result, then **Select Final Columns**.")

        # The join file uploader is always present
        join_file = st.file_uploader("Upload dataset to join with",
                                     type=["csv", "xlsx", "xls", "parquet", "json", "xml", "orc"],
                                     key='join_file_uploader')

        if st.session_state.df is not None:
            # Use a single flag to control the flow state: either viewing the uploader, or viewing the preview.
            if join_file or st.session_state.preview_mode:
                try:
                    # Load the file if it's new, otherwise use the temporary file data (if Streamlit holds it)
                    df_to_join = load_data(join_file) if join_file else None

                    if df_to_join is not None:
                        st.write("### 1. Configure Join & Preview")

                        # Check for self-join
                        is_self_join = st.session_state.df.equals(df_to_join)

                        join_col_1, join_col_2, join_col_3 = st.columns(3)

                        with join_col_1:
                            join_type = st.selectbox("Join Type", ["inner", "left", "right", "outer"],
                                                     key="final_join_type")

                        # Dynamic key selection based on self-join status
                        with join_col_2:
                            if not is_self_join:
                                on_col_1 = st.selectbox("Key in Current Data:", st.session_state.df.columns,
                                                        key="join_key_1")
                                join_on = {
                                    on_col_1: st.selectbox("Key in New Data:", df_to_join.columns, key="join_key_2")}
                            else:
                                st.info("Self-Join Detected.")
                                join_on = st.selectbox("Key in Both Datasets:", st.session_state.df.columns,
                                                       key="join_key_self")

                        with join_col_3:
                            st.markdown("#####")  # Spacer
                            if st.button("Preview Join Result"):

                                # Perform the merge based on join type and self-join status
                                if is_self_join:
                                    preview_df = st.session_state.df.merge(df_to_join, on=join_on, how=join_type,
                                                                           suffixes=('', '_new'))
                                    params_on_cols = join_on
                                else:
                                    # FIX APPLIED HERE: Explicitly convert dict_keys/dict_values to lists
                                    left_keys = list(join_on.keys())
                                    right_keys = list(join_on.values())

                                    preview_df = st.session_state.df.merge(df_to_join, left_on=left_keys,
                                                                           right_on=right_keys, how=join_type)
                                    params_on_cols = f"{left_keys[0]}={right_keys[0]}"  # Simplified for logging

                                st.session_state.preview_df = preview_df
                                st.session_state.preview_mode = True
                                st.session_state.final_join_params = {
                                    "file_name": join_file.name,
                                    "join_type": join_type,
                                    "on_cols": params_on_cols  # Store simplified keys
                                }
                                st.rerun()

                    # --- Step 2: Column Selection and Finalization ---
                    if st.session_state.preview_mode and st.session_state.preview_df is not None:
                        st.write("### 2. Select Final Columns")
                        st.markdown(
                            f"**Previewed Result Shape:** {st.session_state.preview_df.shape[0]} rows, {st.session_state.preview_df.shape[1]} columns")

                        selected_columns = st.multiselect(
                            "Select ALL columns you wish to keep in the final DataFrame:",
                            options=st.session_state.preview_df.columns.tolist(),
                            default=st.session_state.preview_df.columns.tolist(),
                            key="final_cols_select"
                        )

                        if st.button("Finalize and Use Joined Dataset", type="primary"):
                            if selected_columns:
                                final_df_joined = st.session_state.preview_df[selected_columns].copy()

                                # Use the stored parameters for logging
                                params = st.session_state.final_join_params
                                log_text = (
                                    f"Joined dataset with '{params['file_name']}' using a {params['join_type']} join "
                                    f"on '{params['on_cols']}', resulting in {final_df_joined.shape[1]} columns.")


                                # Function to update the main DF
                                def update_df_with_join(d):
                                    # d is the old st.session_state.df, we return the final_df_joined to replace it
                                    return final_df_joined


                                perform_action(
                                    update_df_with_join,
                                    log_text,
                                    {"action": "join", "params": params}
                                )

                                # Clear preview mode after commit
                                st.session_state.preview_mode = False
                                st.session_state.preview_df = None
                                st.session_state.final_join_params = None
                                st.rerun()

                            else:
                                st.warning("Please select at least one column to finalize the join.")

                        st.dataframe(st.session_state.preview_df.head(10), use_container_width=True)
                        if st.button("Cancel Join and Return to Current Data"):
                            st.session_state.preview_mode = False
                            st.session_state.preview_df = None
                            st.session_state.final_join_params = None
                            st.rerun()


                except Exception as e:
                    st.error(f"Failed to process join file or configuration: {e}")

        #  AI QUESTION SECTION


        st.write("### Ask a Question")
        st.markdown(
            "Use this powerful feature to ask any question about your data in natural language. "
            "The AI will generate code and provide a human-readable answer."
        )
        st.markdown("You can then request the Python or MySQL code used for the answer.")

        language_map = {
            "English": "en-US", "Spanish": "es-ES", "French": "fr-FR",
            "German": "de-DE", "Hindi": "hi-IN", "Chinese (Mandarin)": "zh-CN",
            "Japanese": "ja-JP", "Arabic": "ar-SA"
        }

        # --- Language Selection Fix ---
        default_language = list(language_map.keys())[0]

        selected_language = st.selectbox(
            "Choose Language for Voice Input",
            list(language_map.keys()),
            index=0
        )

        selected_language_code = language_map.get(selected_language, language_map[default_language])

        # --- Input Handling ---
        question = None
        query_from_text = st.text_input("Enter your question about the data:")

        if st.button("Ask AI with Your Voice"):
            question = recognize_speech(selected_language_code)

        elif query_from_text and query_from_text != st.session_state.get("last_question", ""):
            question = query_from_text

        # --- Unified AI Processing ---
        if question:
            st.session_state.last_question = question
            st.session_state.show_code = False
            st.session_state.show_sql_code = False
            st.session_state.last_sql_code = ""

            try:
                with st.spinner("Thinking..."):
                    sdf = SmartDataframe(df, config={"llm": llm})
                    response = sdf.chat(str(question))  # ✅ Ensure always a string
                    st.session_state.ai_response = str(response)

                    # --- Python Code Capture ---
                    if hasattr(sdf, 'last_code_generated') and sdf.last_code_generated:
                        st.session_state.last_ai_code = sdf.last_code_generated

                        # --- SQL Generation ---
                        sql_prompt = f"""
                        Given the data and the following Python analysis code:
                        ```python
                        {st.session_state.last_ai_code}
                        ```
                        Write the equivalent MySQL query for a table named `data_table`.
                        Return only the SQL code inside ```sql ... ``` fencing.
                        Do not include explanations.
                        """

                        sql_response = str(sdf.chat(sql_prompt))

                        # Try to capture SQL safely
                        sql_match = re.search(r"```sql\n(.*?)```", sql_response, re.DOTALL | re.IGNORECASE)
                        if sql_match:
                            st.session_state.last_sql_code = sql_match.group(1).strip()
                        else:
                            sql_match = re.search(r"(SELECT\s.*?;)", sql_response, re.DOTALL | re.IGNORECASE)
                            if sql_match:
                                st.session_state.last_sql_code = sql_match.group(1).strip()
                            elif 'SELECT' in sql_response.upper():
                                st.session_state.last_sql_code = sql_response.strip()
                            else:
                                st.session_state.last_sql_code = ""  # no SQL generated
                    else:
                        st.session_state.last_ai_code = "# Code not available or skipped by LLM."
                        st.session_state.last_sql_code = ""

            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.ai_response = None
                st.session_state.last_ai_code = "# Error during processing prevented code generation."
                st.session_state.last_sql_code = ""

        # --- Decoupled Display Logic ---
        if st.session_state.get("ai_response"):
            st.write(f"> Your question: *{st.session_state.last_question}*")
            st.write("#### AI Response:")
            st.write(st.session_state.ai_response)

            code_col, sql_col = st.columns(2)

            # Python Code Button
            with code_col:
                if st.session_state.last_ai_code and not st.session_state.last_ai_code.startswith("#"):
                    if st.button("Know Python Code", key="btn_know_python"):
                        st.session_state.show_code = not st.session_state.show_code
                        st.session_state.show_sql_code = False
                        st.rerun()

            # MySQL Code Button (only if valid SQL exists)
            with sql_col:
                if st.session_state.last_sql_code:
                    if st.button("Know MySQL Code", key="btn_know_sql"):
                        st.session_state.show_sql_code = not st.session_state.show_sql_code
                        st.session_state.show_code = False
                        st.rerun()

            # Display Python Code
            if st.session_state.get("show_code", False):
                st.write("---")
                st.write("#### Python Code used to generate response:")
                st.code(st.session_state.last_ai_code, language='python')

            # Display MySQL Code
            if st.session_state.get("show_sql_code", False) and st.session_state.last_sql_code:
                st.write("---")
                st.write("#### Equivalent MySQL Code:")
                st.code(st.session_state.last_sql_code, language='sql')


        # END: CORRECTED AI QUESTION SECTION

        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>Row Operations</h2>", unsafe_allow_html=True)
        st.markdown(
            "Apply transformations to your data on a row-by-row basis. This includes removing duplicates, handling missing values, and identifying outliers.")
        st.write("### Remove Duplicate Rows")
        st.markdown("Eliminate rows that have identical values across all columns.")
        if st.button("Remove Duplicate Rows"):
            before = len(df)
            perform_action(
                lambda d: d.drop_duplicates(),
                f"Removed {before - len(df.drop_duplicates())} duplicate rows.",
                {"action": "remove_duplicates"}
            )

        st.write("### Dealing with Null Values")
        st.markdown(
            "Fill in missing (null) values in a selected column using various strategies like mean, median, mode, or a custom value.")
        cols_with_nulls = df.columns[df.isnull().any()].tolist()
        if cols_with_nulls:
            fill_col = st.selectbox("Select column to fill NULLs", ["None"] + cols_with_nulls)
        else:
            st.info("No columns with null values detected.")
            fill_col = "None"
        if fill_col != "None":
            is_numeric = pd.api.types.is_numeric_dtype(df[fill_col])
            fill_options = ["Custom Value", "Fill with Previous", "Fill with Next", "AI Recommendation"]
            if is_numeric:
                fill_options = ["Mean", "Median", "Mode"] + fill_options
            else:
                fill_options = ["Mode"] + fill_options
            fill_strategy = st.selectbox("Choose fill strategy", fill_options)
            group_cols = None
            if st.checkbox("Conditional fill (group by other columns)"):
                group_cols = st.multiselect("Group by columns", df.columns)
            custom_value = None
            if fill_strategy == "Custom Value":
                custom_value = st.text_input("Enter custom value")
            if st.button("Apply Fill"):
                try:
                    value_to_fill = None
                    if fill_strategy == "Mean":
                        if group_cols:
                            value_to_fill = df.groupby(group_cols)[fill_col].mean().to_dict()
                            perform_action(
                                lambda d: d.assign(
                                    **{fill_col: d.groupby(group_cols)[fill_col].transform(
                                        lambda x: x.fillna(x.mean()))}),
                                f"Conditionally filled '{fill_col}' with Mean based on {', '.join(group_cols)}.",
                                {"action": "fill_nulls",
                                 "params": {"col": fill_col, "strategy": "Mean", "group_cols": group_cols}},
                                fill_value=value_to_fill
                            )
                        else:
                            value_to_fill = df[fill_col].mean()
                            perform_action(
                                lambda d: d.assign(**{fill_col: d[fill_col].fillna(value_to_fill)}),
                                f"Filled '{fill_col}' with global Mean.",
                                {"action": "fill_nulls", "params": {"col": fill_col, "strategy": "Mean"}},
                                fill_value=value_to_fill
                            )
                    elif fill_strategy == "Median":
                        if group_cols:
                            value_to_fill = df.groupby(group_cols)[fill_col].median().to_dict()
                            perform_action(
                                lambda d: d.assign(**{
                                    fill_col: d.groupby(group_cols)[fill_col].transform(
                                        lambda x: x.fillna(x.median()))}),
                                f"Conditionally filled '{fill_col}' with Median based on {', '.join(group_cols)}.",
                                {"action": "fill_nulls",
                                 "params": {"col": fill_col, "strategy": "Median", "group_cols": group_cols}},
                                fill_value=value_to_fill
                            )
                        else:
                            value_to_fill = df[fill_col].median()
                            perform_action(
                                lambda d: d.assign(**{fill_col: d[fill_col].fillna(value_to_fill)}),
                                f"Filled '{fill_col}' with global Median.",
                                {"action": "fill_nulls", "params": {"col": fill_col, "strategy": "Median"}},
                                fill_value=value_to_fill
                            )
                    elif fill_strategy == "Mode":
                        if group_cols:
                            value_to_fill = df.groupby(group_cols)[fill_col].apply(
                                lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()
                            perform_action(
                                lambda d: d.assign(**{fill_col: d.groupby(group_cols)[fill_col].transform(
                                    lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x))}),
                                f"Conditionally filled '{fill_col}' with Mode based on {', '.join(group_cols)}.",
                                {"action": "fill_nulls",
                                 "params": {"col": fill_col, "strategy": "Mode", "group_cols": group_cols}},
                                fill_value=value_to_fill
                            )
                        else:
                            value_to_fill = df[fill_col].mode().iloc[0] if not df[fill_col].mode().empty else None
                            perform_action(
                                lambda d: d.assign(**{fill_col: d[fill_col].fillna(value_to_fill)}),
                                f"Filled '{fill_col}' with global Mode.",
                                {"action": "fill_nulls", "params": {"col": fill_col, "strategy": "Mode"}},
                                fill_value=value_to_fill
                            )
                    elif fill_strategy == "Custom Value":
                        if custom_value is not None:
                            value_to_fill = custom_value
                            perform_action(
                                lambda d: d.assign(**{fill_col: d[fill_col].fillna(custom_value)}),
                                f"Filled '{fill_col}' with custom value '{custom_value}'.",
                                {"action": "fill_nulls",
                                 "params": {"col": fill_col, "strategy": "Custom Value", "value": custom_value}},
                                fill_value=value_to_fill
                            )
                        else:
                            st.warning("Please enter a custom value.")
                    elif fill_strategy == "AI Recommendation":
                        ai_value = ai_fill_recommendation(df, fill_col)
                        value_to_fill = ai_value
                        perform_action(
                            lambda d: d.assign(**{fill_col: d[fill_col].fillna(ai_value)}),
                            f"Filled '{fill_col}' with AI-recommended value.",
                            {"action": "fill_nulls", "params": {"col": fill_col, "strategy": "AI Recommendation"}},
                            fill_value=value_to_fill
                        )
                    elif fill_strategy == "Fill with Previous":
                        value_to_fill = 'Previous row value'
                        perform_action(
                            lambda d: d.assign(**{fill_col: d[fill_col].fillna(method='ffill')}),
                            f"Filled '{fill_col}' with previous values.",
                            {"action": "fill_nulls", "params": {"col": fill_col, "strategy": "Fill with Previous"}},
                            fill_value=value_to_fill
                        )
                    elif fill_strategy == "Fill with Next":
                        value_to_fill = 'Next row value'
                        perform_action(
                            lambda d: d.assign(**{fill_col: d[fill_col].fillna(method='bfill')}),
                            f"Filled '{fill_col}' with next values.",
                            {"action": "fill_nulls", "params": {"col": fill_col, "strategy": "Fill with Next"}},
                            fill_value=value_to_fill
                        )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error applying fill: {e}")
            if st.button("Show Rows with Remaining Nulls"):
                remaining_nulls_df = df[df[fill_col].isnull()]
                if not remaining_nulls_df.empty:
                    st.write(f"### Rows where '{fill_col}' still has null values:")
                    st.dataframe(remaining_nulls_df)
                else:
                    st.info(f"No remaining null values found in '{fill_col}'.")

        st.write("### Outlier Detection and Removal")
        st.markdown(
            "Use the Interquartile Range (IQR) method to detect and remove outliers from your numeric data. Outliers can skew your analysis.")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            outlier_col = st.selectbox("Select a numeric column to check for outliers", ["None"] + numeric_cols,
                                       key='outlier_col_select')
            if outlier_col != "None":
                if st.button(f"Find Outliers in '{outlier_col}'"):
                    Q1 = df[outlier_col].quantile(0.25)
                    Q3 = df[outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]

                    st.session_state.outlier_count = len(outliers)
                    st.session_state.outliers_found = not outliers.empty
                    st.session_state.outlier_df = outliers

                    if st.session_state.outliers_found:
                        st.write(f"Found **{st.session_state.outlier_count}** outliers in '{outlier_col}'.")
                        st.write("First 5 rows of detected outliers:")
                        st.dataframe(st.session_state.outlier_df.head())
                    else:
                        st.info(f"No outliers found in '{outlier_col}' based on the IQR method.")
                    st.rerun()

                if st.session_state.outliers_found:
                    outlier_action = st.radio(
                        "Choose action for outliers:",
                        ["Remove Outliers", "Replace with Median", "Replace with Mode"],
                        key='outlier_action_select'
                    )

                    if st.button(f"Apply '{outlier_action}' to '{outlier_col}'"):
                        def handle_outliers_func(d):
                            current_df = d.copy()
                            Q1 = current_df[outlier_col].quantile(0.25)
                            Q3 = current_df[outlier_col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            mask = (current_df[outlier_col] < lower_bound) | (current_df[outlier_col] > upper_bound)
                            if outlier_action == "Remove Outliers":
                                current_df = current_df.loc[~mask].reset_index(drop=True)
                            elif outlier_action == "Replace with Median":
                                median_val = current_df[outlier_col].median()
                                current_df.loc[mask, outlier_col] = median_val
                            elif outlier_action == "Replace with Mode":
                                mode_val = current_df[outlier_col].mode().iloc[0] if not current_df[
                                    outlier_col].mode().empty else None
                                if mode_val is not None:
                                    current_df.loc[mask, outlier_col] = mode_val
                            return current_df

                        perform_action(
                            handle_outliers_func,
                            f"{outlier_action} applied on outliers in '{outlier_col}'.",
                            {"action": "handle_outliers", "params": {"col": outlier_col, "method": outlier_action}}
                        )
        else:
            st.info("No numeric columns available for outlier detection.")


        # Replace Specific Cell Values

        import re

        st.write("### Replace Specific Cell Values")
        st.markdown(
            "Replace specific values in a column, either by manually entering old/new values or by applying a condition.\n\n"
            "*Condition examples:* >18, <=100, ==Yes, !=No, >=2 & <=4 (use & or and for AND, | or or for OR)."
        )
        replace_col = st.selectbox("Select column to replace values", ["None"] + list(st.session_state.df.columns),
                                   key='replace_col_select')
        if replace_col != "None":
            replace_method = st.radio("Choose replacement method:", ["Manual", "By Condition"], key="replace_method")

            # Manual Replacement
            if replace_method == "Manual":
                st.info("Enter comma-separated values. Matching is case-insensitive (trimmed).")
                old_values = st.text_input("Values to replace (comma-separated)", key="manual_old_values")
                new_values = st.text_input("New values (comma-separated, same count as old)", key="manual_new_values")
                if st.button("Replace Value(s)"):
                    if old_values and new_values:
                        old_list = [val.strip() for val in old_values.split(',')]
                        new_list = [val.strip() for val in new_values.split(',')]
                        if len(old_list) != len(new_list):
                            st.error("Please ensure the number of old values matches the number of new values.")
                        else:
                            replace_dict = {k.lower(): v for k, v in zip(old_list, new_list)}

                            def _manual_replace(d):
                                new_df = d.copy()
                                # normalize strings for matching, but write the replacement as provided
                                new_df[replace_col] = new_df[replace_col].astype(str).str.strip().apply(
                                    lambda x: replace_dict.get(x.lower(), x)
                                )
                                return new_df

                            perform_action(
                                lambda d: _manual_replace(d),
                                f"Replaced values in '{replace_col}' using manual mapping: {replace_dict}",
                                {"action": "replace_values",
                                 "params": {"col": replace_col, "method": "manual", "map": replace_dict}}
                            )
                    else:
                        st.warning("Please provide old and new values.")

            #  Conditional Replacement
            else:
                st.info(
                    "Example: >18 => sets numeric rows where value>18. Combine using & (AND) or | (OR): >=2 & <=4.")
                condition_str = st.text_input(
                    f"Enter condition for '{replace_col}' (e.g., '>100', '>=2 & <=4', \"==Yes\", \"contains abc\")",
                    key="cond_input")
                replacement_val = st.text_input("Enter replacement value", key="cond_replacement")

                def _build_mask_from_condition(series: pd.Series, cond_str: str) -> pd.Series:
                    """Return boolean mask for rows matching condition string.
                        Supports numeric operators: >, <, >=, <=, ==, !=
                        Supports combining with & / and (AND) or | / or (OR)
                        For non-numeric, supports '==value', '!=value', and 'contains substring' (use 'contains abc').
                    """
                    cond = cond_str.strip()
                    if cond == "":
                        raise ValueError("Empty condition")

                    # decide combining operator
                    if re.search(r'\s*(?:&|\band\b)\s*', cond, flags=re.IGNORECASE):
                        parts = re.split(r'\s*(?:&|\band\b)\s*', cond, flags=re.IGNORECASE)
                        combiner = 'and'
                    elif re.search(r'\s*(?:\||\bor\b)\s*', cond, flags=re.IGNORECASE):
                        parts = re.split(r'\s*(?:\||\bor\b)\s*', cond, flags=re.IGNORECASE)
                        combiner = 'or'
                    else:
                        parts = [cond]
                        combiner = None

                    def _single_part_mask(part: str) -> pd.Series:
                        p = part.strip()
                        # contains pattern e.g., contains abc
                        m_contains = re.match(r'^(?:contains)\s+["\']?(.*[^"\'])["\']?$', p, flags=re.IGNORECASE)
                        if m_contains:
                            substr = m_contains.group(1).strip()
                            return series.astype(str).str.contains(re.escape(substr), case=False, na=False)

                        # numeric / comparison operators
                        m = re.match(r'^(>=|<=|==|!=|>|<)\s*(.+)$', p)
                        if m:
                            op = m.group(1)
                            val_str = m.group(2).strip().strip('\'"')

                            #  Try conversion to numeric always for comparison operators
                            # This allows comparison on a column that is currently 'object' but contains numbers
                            # (mixed types or converted after the first replacement)
                            s_converted = pd.to_numeric(series, errors='coerce')

                            # If the conversion resulted in mostly non-numeric data (e.g. if the whole column is now strings)
                            # AND the operator is numeric (<, >), we must switch to the string comparison logic.
                            is_comparison_operator = op not in ('==', '!=')

                            # Check if it's primarily numeric OR if we are using ==/!= (which works on strings)
                            if s_converted.notna().sum() > 0 or not is_comparison_operator:
                                if not is_comparison_operator:  # == or !=
                                    # String comparison logic (case-insensitive)
                                    if op in ('==', '!='):
                                        cmp = series.astype(str).str.strip().str.lower() == val_str.lower()
                                        return cmp if op == '==' else ~cmp
                                else:  # Numeric comparison logic
                                    try:
                                        val = float(val_str)
                                    except Exception:
                                        raise ValueError(f"Cannot parse numeric value: {val_str}")

                                    if op == '>':
                                        return s_converted > val
                                    if op == '<':
                                        return s_converted < val
                                    if op == '>=':
                                        return s_converted >= val
                                    if op == '<=':
                                        return s_converted <= val
                                    if op == '==':
                                        return s_converted == val
                                    if op == '!=':
                                        return s_converted != val

                            # Fallback for remaining cases that failed numeric comparison (e.g. non-numeric column with <)
                            if is_comparison_operator:
                                raise ValueError(
                                    "Operator not supported for non-numeric column. Use == or != or 'contains'.")

                        # If input doesn't match operator form, treat as equality string for non-numeric
                        if not pd.api.types.is_numeric_dtype(series):
                            target = p.strip().strip('\'"').lower()
                            return series.astype(str).str.strip().str.lower() == target

                        raise ValueError(f"Could not parse condition part: '{part}'")

                    # build initial mask
                    masks = []
                    for part in parts:
                        masks.append(_single_part_mask(part))

                    # combine masks
                    if combiner == 'and':
                        result = masks[0]
                        for m in masks[1:]:
                            result = result & m
                        return result
                    elif combiner == 'or':
                        result = masks[0]
                        for m in masks[1:]:
                            result = result | m
                        return result
                    else:
                        return masks[0]

                if st.button("Replace Value(s) by Condition"):
                    if condition_str and replacement_val:
                        try:
                            def conditional_replace(d):
                                new_df = d.copy()
                                original_dtype = new_df[replace_col].dtype
                                mask = _build_mask_from_condition(new_df[replace_col], condition_str)
                                new_df.loc[mask, replace_col] = replacement_val
                                # Try to convert back to original dtype
                                try:
                                    new_df[replace_col] = new_df[replace_col].astype(original_dtype)
                                except:
                                    st.warning(
                                        f"Could not convert '{replace_col}' back to original type after conditional replace.")
                                    pass
                                return new_df

                            perform_action(
                                lambda d: conditional_replace(d),
                                f"Replaced values in '{replace_col}' where '{condition_str}' with '{replacement_val}'",
                                {"action": "replace_values",
                                 "params": {"col": replace_col, "method": "condition", "condition": condition_str,
                                            "replacement": replacement_val}}
                            )

                        except Exception as e:
                            st.error(f"Error applying conditional replace: {e}")
        st.write("### Remove Rows by Condition")
        st.markdown(
            "Remove rows that meet a specific condition, such as values greater than a certain number or containing a specific text string.")
        col_to_filter = st.selectbox("Select column to filter on", ["None"] + list(df.columns), key='filter_col_select')
        if col_to_filter != "None":
            operator = st.selectbox("Select operator", ["<", ">", "<=", ">=", "==", "!=", "contains"])
            filter_value = st.text_input("Enter value")
            if st.button("Remove Rows"):
                if filter_value:
                    try:
                        if operator == "contains":
                            perform_action(
                                lambda d: d[
                                    ~d[col_to_filter].astype(str).str.contains(filter_value, case=False, na=False)],
                                f"Removed rows where '{col_to_filter}' contains '{filter_value}'.",
                                {"action": "remove_rows_by_condition",
                                 "params": {"col": col_to_filter, "operator": operator, "value": filter_value}}
                            )
                        elif pd.api.types.is_numeric_dtype(df[col_to_filter]):
                            filter_value = float(filter_value)
                            if operator == "<":
                                perform_action(
                                    lambda d: d[d[col_to_filter] >= filter_value],
                                    f"Removed rows where '{col_to_filter}' < {filter_value}.",
                                    {"action": "remove_rows_by_condition",
                                     "params": {"col": col_to_filter, "operator": operator, "value": filter_value}}
                                )
                            elif operator == ">":
                                perform_action(
                                    lambda d: d[d[col_to_filter] <= filter_value],
                                    f"Removed rows where '{col_to_filter}' > {filter_value}.",
                                    {"action": "remove_rows_by_condition",
                                     "params": {"col": col_to_filter, "operator": operator, "value": filter_value}}
                                )
                            elif operator == "<=":
                                perform_action(
                                    lambda d: d[d[col_to_filter] > filter_value],
                                    f"Removed rows where '{col_to_filter}' <= {filter_value}.",
                                    {"action": "remove_rows_by_condition",
                                     "params": {"col": col_to_filter, "operator": operator, "value": filter_value}}
                                )
                            elif operator == ">=":
                                perform_action(
                                    lambda d: d[d[col_to_filter] < filter_value],
                                    f"Removed rows where '{col_to_filter}' >= {filter_value}.",
                                    {"action": "remove_rows_by_condition",
                                     "params": {"col": col_to_filter, "operator": operator, "value": filter_value}}
                                )
                            elif operator == "==":
                                perform_action(
                                    lambda d: d[d[col_to_filter] != filter_value],
                                    f"Removed rows where '{col_to_filter}' == {filter_value}.",
                                    {"action": "remove_rows_by_condition",
                                     "params": {"col": col_to_filter, "operator": operator, "value": filter_value}}
                                )
                            elif operator == "!=":
                                perform_action(
                                    lambda d: d[d[col_to_filter] == filter_value],
                                    f"Removed rows where '{col_to_filter}' != {filter_value}.",
                                    {"action": "remove_rows_by_condition",
                                     "params": {"col": col_to_filter, "operator": operator, "value": filter_value}}
                                )
                        else:
                            st.warning(
                                "This operator is only valid for numeric columns. Please choose a numeric column or use 'contains'.")
                    except ValueError:
                        st.error("Invalid value. Please enter a number for this type of condition.")
                    except Exception as e:
                        st.error(f"Error applying filter: {e}")
                else:
                    st.warning("Please enter a value to filter on.")

        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>Column Operations</h2>", unsafe_allow_html=True)
        st.markdown(
            "Apply transformations to your data on a column-by-column basis. This includes changing data types, splitting columns, and creating new columns from existing ones.")
        st.write("### Change Column Data Types")
        st.markdown(
            "Convert a column's data type to the correct format for analysis. For example, convert a text column containing numbers to an integer or float type.")
        col_to_change = st.selectbox("Select column to change type", ["None"] + list(df.columns))
        if col_to_change != "None":
            new_type = st.selectbox("Select new data type", ["int", "float", "string", "datetime"])
            if st.button("Change Data Type"):
                try:
                    if new_type == "int":
                        perform_action(
                            lambda d: d.assign(**{col_to_change: d[col_to_change].astype(int)}),
                            f"Changed column '{col_to_change}' to {new_type}.",
                            {"action": "change_dtype", "params": {"col": col_to_change, "new_type": "int"}}
                        )
                    elif new_type == "float":
                        perform_action(
                            lambda d: d.assign(**{col_to_change: d[col_to_change].astype(float)}),
                            f"Changed column '{col_to_change}' to {new_type}.",
                            {"action": "change_dtype", "params": {"col": col_to_change, "new_type": "float"}}
                        )
                    elif new_type == "string":
                        perform_action(
                            lambda d: d.assign(**{col_to_change: d[col_to_change].astype(str)}),
                            f"Changed column '{col_to_change}' to {new_type}.",
                            {"action": "change_dtype", "params": {"col": col_to_change, "new_type": "string"}}
                        )
                    elif new_type == "datetime":
                        perform_action(
                            lambda d: d.assign(**{col_to_change: pd.to_datetime(d[col_to_change], errors="coerce")}),
                            f"Changed column '{col_to_change}' to {new_type}.",
                            {"action": "change_dtype", "params": {"col": col_to_change, "new_type": "datetime"}}
                        )
                except Exception as e:
                    st.error(f"Failed to change data type: {e}")

        st.write("### Remove Columns")
        st.markdown(
            "Permanently drop one or more columns from your dataset. This can help simplify your data and focus on what's important.")
        cols_to_remove = st.multiselect("Select columns to remove", df.columns)
        if st.button("Remove Selected Columns"):
            if cols_to_remove:
                perform_action(
                    lambda d: d.drop(columns=cols_to_remove),
                    f"Removed columns: {', '.join(cols_to_remove)}",
                    {"action": "remove_columns", "params": {"columns": cols_to_remove}}
                )
            else:
                st.warning("Please select at least one column to remove.")

        st.write("### Split Column")
        st.markdown(
            "Break a single column into multiple new columns. You can split by a specific delimiter (like a comma) or by a fixed number of characters.")
        split_col = st.selectbox("Select column to split:", ["None"] + list(df.columns))
        if split_col != "None":
            split_method = st.radio("Choose split method:", ["By Delimiter", "By Number of Characters"])

            if split_method == "By Delimiter":
                delimiter = st.text_input("Enter the delimiter (e.g., ',' or '-')")
                new_col_names = st.text_input("Enter new column names (comma-separated, e.g., 'col1,col2')")
                if st.button("Split Column"):
                    if delimiter and new_col_names:
                        new_names_list = [name.strip() for name in new_col_names.split(',')]
                        try:
                            split_data = df[split_col].astype(str).str.split(delimiter, expand=True)
                            split_data.columns = new_names_list
                            perform_action(
                                lambda d: pd.concat([d, split_data], axis=1),
                                f"Split column '{split_col}' into {len(new_names_list)} new columns: {new_col_names} using '{delimiter}'.",
                                {"action": "split_column",
                                 "params": {"col": split_col, "method": "By Delimiter", "delimiter": delimiter,
                                            "new_names": new_names_list}}
                            )
                        except Exception as e:
                            st.error(f"Error splitting column: {e}")
                    else:
                        st.warning("Please provide a delimiter and new column names.")
            else:  # By Number of Characters
                num_chars_input = st.text_input(
                    "Enter number of characters for each new column (comma-separated, e.g., '3,4,5')")
                new_col_names = st.text_input("Enter new column names (comma-separated, e.g., 'part1,part2,part3')")
                if st.button("Split Column"):
                    if num_chars_input and new_col_names:
                        try:
                            num_chars_list = [int(n.strip()) for n in num_chars_input.split(',')]
                            new_names_list = [name.strip() for name in new_col_names.split(',')]
                            if len(num_chars_list) != len(new_names_list):
                                st.error("Number of characters must match the number of new column names.")
                            else:
                                def split_by_chars_func(d):
                                    new_df = d.copy()

                                    def split_by_chars(text, lengths):
                                        parts = []
                                        current_pos = 0
                                        for length in lengths:
                                            parts.append(text[current_pos:current_pos + length])
                                            current_pos += length
                                        return parts

                                    split_data = pd.DataFrame(
                                        new_df[split_col].astype(str).apply(
                                            lambda x: split_by_chars(x, num_chars_list)).tolist(),
                                        index=new_df.index)
                                    split_data.columns = new_names_list
                                    return pd.concat([new_df, split_data], axis=1)

                                perform_action(
                                    split_by_chars_func,
                                    f"Split column '{split_col}' into {len(new_names_list)} new columns: {new_col_names} by character count.",
                                    {"action": "split_column",
                                     "params": {"col": split_col, "method": "By Number of Characters",
                                                "num_chars": num_chars_list, "new_names": new_names_list}}
                                )
                        except ValueError:
                            st.error("Invalid number of characters. Please enter a comma-separated list of integers.")
                        except Exception as e:
                            st.error(f"Error splitting column: {e}")
                    else:
                        st.warning("Please provide number of characters and new column names.")

        st.write(" Merge Columns")
        st.markdown("Combine multiple columns into a single new column using a specified delimiter.")
        merge_cols = st.multiselect("Select columns to merge:", df.columns)
        if merge_cols:
            new_merged_col_name = st.text_input("Enter the new merged column name:")
            merge_delimiter = st.text_input("Enter the delimiter to merge with (e.g., ',' or '-')")
            if st.button("Merge Columns"):
                if len(merge_cols) >= 2 and new_merged_col_name and merge_delimiter:
                    perform_action(
                        lambda d: d.assign(
                            **{new_merged_col_name: d[merge_cols].astype(str).agg(merge_delimiter.join, axis=1)}),
                        f"Merged columns {', '.join(merge_cols)} into a new column '{new_merged_col_name}'.",
                        {"action": "merge_columns",
                         "params": {"cols": merge_cols, "new_name": new_merged_col_name, "delimiter": merge_delimiter}}
                    )
                else:
                    st.warning("Please select at least two columns and provide a new name and delimiter.")

        st.write(" Create New Column from Conditions")
        st.markdown(
            "Create a new categorical column based on conditions applied to an existing column. This is useful for grouping data into meaningful categories.")
        col_to_categorize = st.selectbox("Select a column to create conditions on", ["None"] + list(df.columns),
                                         key='cat_col_select')
        if col_to_categorize != "None":
            new_category_col = st.text_input("Enter new column name (e.g., 'Rating_category')")
            num_conditions = st.number_input("Number of conditions", min_value=1, value=1, step=1, key='num_cond_input')
            conditions = []
            labels = []
            st.write(" Define Conditions and Labels")
            st.info("Example format for conditions:\n"
                    "- <2\n"
                    "- >=2 & <=4\n"
                    "- >4\n"
                    "Labels can be any text like Low, Between2and4, High.")
            for i in range(int(num_conditions)):
                with st.expander(f"Condition {i + 1}"):
                    condition_text = st.text_input(f"If '{col_to_categorize}' is ...", key=f"condition_{i}")
                    label_text = st.text_input(f"Then label as ...", key=f"label_{i}")
                    conditions.append(condition_text)
                    labels.append(label_text)
            final_label = st.text_input("Final label (for values not matching any of the above conditions)",
                                        key="final_label")
            if st.button("Create Categorical Column"):
                if new_category_col and final_label and all(labels) and all(conditions):
                    def categorize_data(d):
                        new_df = d.copy()
                        new_df[new_category_col] = final_label
                        for i in range(len(conditions)):
                            cond_text = convert_condition_to_python(conditions[i], col_to_categorize)
                            label = labels[i]
                            try:
                                mask = eval(cond_text)
                                new_df.loc[mask, new_category_col] = label
                            except Exception as e:
                                st.warning(f"Skipping condition '{conditions[i]}' due to error: {e}")
                        return new_df

                    perform_action(
                        lambda d: categorize_data(d),
                        f"Created a new column '{new_category_col}' based on '{col_to_categorize}'.",
                        {"action": "create_column",
                         "params": {"col": col_to_categorize, "new_col": new_category_col, "conditions": conditions,
                                    "labels": labels, "final_label": final_label}}
                    )
                else:
                    st.warning("Please fill in all required fields and define at least one condition.")

        st.write(" Create New Column with Arithmetic Operations")
        st.markdown("Create a new column by applying mathematical operations to one or more existing columns.")
        col_names = list(df.columns)
        new_col_name = st.text_input("New column name", key="new_col_name_arithmetic")
        operation_type = st.radio("Choose operation type:", ["Combine multiple columns", "Perform on a single column"],
                                  key="op_type_radio")

        if operation_type == "Combine multiple columns":
            selected_cols = st.multiselect("Select columns for calculation", col_names, key="selected_cols_arithmetic")
            operation = st.selectbox("Choose operation", ["Add (+)", "Subtract (-)", "Multiply (*)", "Divide (/)"],
                                     key="arithmetic_op")
            if st.button("Create New Column"):
                if not new_col_name or not selected_cols or len(selected_cols) < 2:
                    st.warning("Please provide a new column name and select at least two columns.")
                else:
                    try:
                        op_symbol = operation.split(" ")[1].replace('(', '').replace(')', '')
                        new_series = df[selected_cols[0]].copy()
                        for col in selected_cols[1:]:
                            if op_symbol == '+':
                                new_series += df[col]
                            elif op_symbol == '-':
                                new_series -= df[col]
                            elif op_symbol == '*':
                                new_series *= df[col]
                            elif op_symbol == '/':
                                new_series /= df[col]

                        perform_action(
                            lambda d: d.assign(**{new_col_name: new_series}),
                            f"Created new column '{new_col_name}' by {operation.lower().strip().replace('(', '').replace(')', '')}ing {', '.join(selected_cols)}.",
                            {"action": "create_column_arithmetic",
                             "params": {"new_col": new_col_name, "cols": selected_cols, "operation": op_symbol}}
                        )
                    except Exception as e:
                        st.error(f"Error performing arithmetic operation: {e}")
        else:  # Perform on a single column
            selected_col = st.selectbox("Select column for calculation", col_names, key="selected_col_single")

            calculation_method = st.radio(
                "Calculation Method:",
                ["Arithmetic Operation with a Number", "Extract Date Part", "Add/Subtract Date Parts"],
                key="single_col_calc_method"
            )

            if calculation_method == "Arithmetic Operation with a Number":
                is_numeric = pd.api.types.is_numeric_dtype(df[selected_col])
                if not is_numeric:
                    st.warning(f"Column '{selected_col}' is not numeric. Please select a different column or method.")
                else:
                    factor = st.number_input("Enter a number to perform the operation with:", value=0.0, format="%f",
                                             key="factor_single")
                    operation = st.selectbox("Choose operation",
                                             ["Add by", "Subtract by", "Multiply by", "Divide by"],
                                             key="single_op")
                    if st.button("Create New Column"):
                        if not new_col_name or not selected_col:
                            st.warning("Please provide a new column name and select a column.")
                        else:
                            try:
                                new_series = df[selected_col].copy()
                                if operation == "Add by":
                                    new_series += factor
                                elif operation == "Subtract by":
                                    new_series -= factor
                                elif operation == "Multiply by":
                                    new_series *= factor
                                elif operation == "Divide by":
                                    new_series /= factor

                                perform_action(
                                    lambda d: d.assign(**{new_col_name: new_series}),
                                    f"Created new column '{new_col_name}' by performing '{operation}' on '{selected_col}' with factor {factor}.",
                                    {"action": "create_column_single_arithmetic",
                                     "params": {"new_col": new_col_name, "col": selected_col, "operation": operation,
                                                "factor": factor}}
                                )
                            except Exception as e:
                                st.error(f"Error performing single column operation: {e}")

            elif calculation_method == "Extract Date Part":
                date_col_names = [col for col in df.columns if
                                  pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_string_dtype(
                                      df[col])]
                date_col_to_use = st.selectbox("Select a date/string column:", ["None"] + date_col_names,
                                               key="date_part_col_select")
                if date_col_to_use != "None":
                    part_to_extract = st.selectbox("Extract:", ["year", "month", "day", "month_name"],
                                                   key="date_part_select")
                    if st.button("Create Date Part Column"):
                        if not new_col_name or not date_col_to_use:
                            st.warning("Please provide a new column name and select a date column.")
                        else:
                            try:
                                def extract_date_part(d):
                                    new_df = d.copy()
                                    temp_series = pd.to_datetime(new_df[date_col_to_use], errors='coerce')

                                    if part_to_extract == 'month_name':
                                        new_df[new_col_name] = temp_series.dt.month_name()
                                    elif part_to_extract == 'year':
                                        new_df[new_col_name] = temp_series.dt.year
                                    elif part_to_extract == 'day':
                                        new_df[new_col_name] = temp_series.dt.day
                                    elif part_to_extract == 'month':
                                        new_df[new_col_name] = temp_series.dt.month
                                    return new_df

                                perform_action(
                                    lambda d: extract_date_part(d),
                                    f"Created a new column '{new_col_name}' by extracting '{part_to_extract}' from '{date_col_to_use}'.",
                                    {"action": "create_column_date_part",
                                     "params": {"new_col": new_col_name, "col": date_col_to_use,
                                                "part": part_to_extract}}
                                )
                            except Exception as e:
                                st.error(f"Error extracting date part: {e}")

            elif calculation_method == "Add/Subtract Date Parts":
                date_col_names = [col for col in df.columns if
                                  pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_string_dtype(
                                      df[col])]
                date_col_to_use = st.selectbox("Select a date/string column:", ["None"] + date_col_names,
                                               key="date_add_col_select")
                if date_col_to_use != "None":
                    value_to_add = st.number_input("Value to add/subtract:", value=0, step=1, key="date_add_value")
                    unit = st.selectbox("Unit:", ["Days", "Months", "Years"], key="date_add_unit")
                    operation = st.selectbox("Operation:", ["Add", "Subtract"], key="date_add_subtract_op")
                    if st.button("Create Modified Date Column"):
                        if not new_col_name or not date_col_to_use:
                            st.warning("Please provide a new column name and select a date column.")
                        else:
                            try:
                                def add_subtract_date_parts(d):
                                    new_df = d.copy()
                                    temp_series = pd.to_datetime(new_df[date_col_to_use], errors='coerce')

                                    if operation == "Add":
                                        if unit == "Days":
                                            new_series = temp_series + pd.to_timedelta(value_to_add, unit='d')
                                        elif unit == "Months":
                                            new_series = temp_series + pd.DateOffset(months=value_to_add)
                                        elif unit == "Years":
                                            new_series = temp_series + pd.DateOffset(years=value_to_add)
                                    elif operation == "Subtract":
                                        if unit == "Days":
                                            new_series = temp_series - pd.to_timedelta(value_to_add, unit='d')
                                        elif unit == "Months":
                                            new_series = temp_series - pd.DateOffset(months=value_to_add)
                                        elif unit == "Years":
                                            new_series = temp_series - pd.DateOffset(years=value_to_add)

                                    new_df[new_col_name] = new_series
                                    return new_df

                                perform_action(
                                    lambda d: add_subtract_date_parts(d),
                                    f"Created new column '{new_col_name}' by '{operation}ing' {value_to_add} {unit.lower()} from '{date_col_to_use}'.",
                                    {"action": "create_column_date_add_subtract",
                                     "params": {"new_col": new_col_name, "col": date_col_to_use, "value": value_to_add,
                                                "unit": unit, "operation": operation}}
                                )
                            except Exception as e:
                                st.error(f"Error performing date arithmetic: {e}")

        st.markdown("---")
        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>Generate Visualization</h2>", unsafe_allow_html=True)
        st.markdown(
            "Create compelling charts and graphs to visualize trends, distributions, and relationships in your data.")

        # Chart Type Selection
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Count Plot", "Histogram", "Box Plot", "Heatmap"],
            key="chart_type_select"
        )

        # --- Standard Plotting ---
        st.markdown("##### Standard Plotting")
        st.info(
            "Configure filters, aggregation, sorting, and top-N limits to prepare your data for plotting."
        )

        # Filtering (Kept as is)
        filter_col_viz = st.selectbox("Filter on column (optional)", ["None"] + list(df.columns), key="filter_col_viz")
        filtered_df = df.copy()

        if filter_col_viz != "None":
            unique_vals = list(df[filter_col_viz].unique())
            filter_val = st.selectbox(
                f"Select value to filter '{filter_col_viz}' by", ["None"] + unique_vals, key="filter_val_viz"
            )
            if filter_val != "None":
                filtered_df = filtered_df[filtered_df[filter_col_viz] == filter_val]
                st.write(f"Applying filter: *{len(filtered_df)}* rows remaining.")
                if filtered_df.empty:
                    st.warning("The filter resulted in an empty dataset. Please adjust your filter.")

        # Chart title (Kept as is)
        chart_title = st.text_input("Enter chart title:", key="chart_title")

        # --- NEW AXIS LABEL INPUTS ---
        x_label = st.text_input("Enter X-axis Label (Optional):", key="x_axis_label_input")
        y_label = st.text_input("Enter Y-axis Label (Optional):", key="y_axis_label_input")
        # ------------------------------

        x_col, y_col, hue_col, agg_method = None, None, None, None
        is_agg_applied = False

        # Axis and aggregation options
        if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Count Plot", "Histogram", "Heatmap"]:

            x_col_options = ["None"] + filtered_df.columns.tolist()
            x_col = st.selectbox("Select X-axis Column", x_col_options, key="chart_x_col")

            if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot"]:
                y_col = st.selectbox("Select Y-axis Column", ["None"] + list(filtered_df.columns), key="chart_y_col")

                if y_col != "None" and pd.api.types.is_numeric_dtype(filtered_df[y_col]):
                    agg_method = st.selectbox(
                        "Select aggregation method for the Y-axis",
                        ["mean", "sum", "count", "min", "max", "median"],
                        key="standard_agg_method"
                    )
                    is_agg_applied = True
                elif y_col != "None":
                    st.warning(f"Aggregation is not available for non-numeric column '{y_col}'.")

            hue_col = st.selectbox(
                "Select a column for 'Hue' (optional)", ["None"] + filtered_df.columns.tolist(), key="chart_hue_col"
            )

            if chart_type in ["Count Plot", "Histogram", "Box Plot"]:
                if x_col == "None" and y_col == "None":
                    st.warning(f"Please select at least one column (X or Y) for a {chart_type}.")

        elif chart_type == "Pie Chart":
            x_col = st.selectbox("Select Column for Pie Chart", filtered_df.columns, key="chart_x_col")
            y_col = 'Count'
        elif chart_type == "Heatmap":
            st.info("Heatmap will show correlations between all numeric columns.")

        # --- Feature 5: Sorting and Top-N (Head) ---
        sortable_cols = []
        if x_col != "None":
            sortable_cols.append(x_col)
        if y_col != "None" and y_col != 'Count' and y_col != 'Correlation Value':
            sortable_cols.append(y_col)
        sortable_cols.append("Count")

        sort_col_viz = st.selectbox("Sort chart bars/points by (value or count):", ["None"] + sortable_cols,
                                    key="sort_col_viz")
        sort_order_viz = st.radio("Sort Order:", ["Descending", "Ascending"], key="sort_order_viz")
        head_n_viz = st.number_input("Show Top N values (0 for all):", min_value=0, step=1, key="head_n_viz")

        # --- Generate Plot Button ---
        if st.button("Generate Standard Plot"):
            if filtered_df.empty:
                st.error("Cannot generate plot on an empty (filtered) dataset.")
            elif chart_type == "Pie Chart" and x_col is None:
                st.error("Please select a column for the Pie Chart.")
            elif chart_type == "Scatter Plot" and (x_col == 'None' or y_col == 'None'):
                st.error("Scatter plots require both X-axis and Y-axis columns.")
            elif chart_type in ["Bar Chart", "Line Chart", "Box Plot"] and x_col == 'None' and y_col == 'None':
                st.error(f"{chart_type} requires at least one column (X or Y) to plot.")
            else:
                fig = plt.figure(figsize=(10, 6))
                plot_df = filtered_df.copy()
                final_plot_data_for_ai = plot_df

                primary_col = x_col if x_col != 'None' else y_col
                y_col_for_ai = y_col

                try:
                    # --- Aggregation and Counting ---
                    if chart_type in ["Bar Chart", "Line Chart"] and is_agg_applied:
                        group_cols = [c for c in [x_col, hue_col] if c != 'None']
                        final_plot_data_for_ai = plot_df.groupby(group_cols)[y_col].agg(agg_method).reset_index()
                        y_col_for_ai = y_col

                    elif chart_type in ["Bar Chart", "Count Plot", "Pie Chart"]:
                        if x_col != 'None':
                            count_col = x_col
                        elif y_col != 'None':
                            count_col = y_col
                        else:
                            count_col = None

                        if count_col:
                            if hue_col != 'None':
                                final_plot_data_for_ai = plot_df.groupby([count_col, hue_col]).size().reset_index(
                                    name='count')
                            else:
                                final_plot_data_for_ai = plot_df[count_col].value_counts().to_frame().reset_index()
                                final_plot_data_for_ai.columns = [count_col, 'count']

                            x_col_for_plot = count_col
                            y_col_for_ai = 'count'

                    # --- Apply Sorting and Head-N ---
                    sort_by_col = None
                    if sort_col_viz == 'Count' and 'count' in final_plot_data_for_ai.columns:
                        sort_by_col = 'count'
                    elif sort_col_viz != 'None' and sort_col_viz in final_plot_data_for_ai.columns:
                        sort_by_col = sort_col_viz

                    if sort_by_col:
                        final_plot_data_for_ai = final_plot_data_for_ai.sort_values(
                            by=sort_by_col,
                            ascending=(sort_order_viz == "Ascending")
                        )

                        if head_n_viz > 0:
                            final_plot_data_for_ai = final_plot_data_for_ai.head(head_n_viz)

                    # Determine the Order List for Categorical Plots
                    order_values = None
                    order_col = x_col if x_col != 'None' else y_col

                    if sort_by_col and order_col != 'None' and order_col in final_plot_data_for_ai.columns:
                        order_values = final_plot_data_for_ai[order_col].tolist()

                    # --- Plotting Logic ---
                    if chart_type == "Bar Chart":
                        y_to_plot = y_col_for_ai if y_col_for_ai != y_col else y_col

                        ax = sns.barplot(data=final_plot_data_for_ai, x=x_col, y=y_to_plot,
                                         hue=hue_col if hue_col != "None" else None,
                                         order=order_values)

                        plt.ylabel(
                            f"{agg_method.capitalize()} of {y_col}" if is_agg_applied and not y_label else "Count")
                        for container in ax.containers:
                            ax.bar_label(container)

                    elif chart_type == "Line Chart":
                        if x_col == 'None' or y_col == 'None':
                            st.error("Line charts require both X-axis and Y-axis columns.")
                            plt.close(fig)
                            st.stop()
                        sns.lineplot(data=final_plot_data_for_ai, x=x_col, y=y_col,
                                     hue=hue_col if hue_col != "None" else None)
                        plt.ylabel(f"{agg_method.capitalize()} of {y_col}")

                    elif chart_type == "Scatter Plot":
                        sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=hue_col if hue_col != "None" else None)

                    elif chart_type == "Box Plot":
                        x_for_plot = x_col if x_col != 'None' else None
                        y_for_plot = y_col if y_col != 'None' else None

                        sns.boxplot(data=plot_df, x=x_for_plot, y=y_for_plot,
                                    hue=hue_col if hue_col != "None" else None,
                                    order=order_values if x_for_plot else None)

                    elif chart_type == "Pie Chart":
                        counts = final_plot_data_for_ai['count']
                        labels = final_plot_data_for_ai[x_col]
                        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
                        plt.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left')

                    elif chart_type == "Count Plot":
                        x_to_plot = x_col if x_col != 'None' else y_col

                        ax = sns.barplot(data=final_plot_data_for_ai, x=x_to_plot, y='count',
                                         hue=hue_col if hue_col != "None" else None,
                                         order=order_values)

                        plt.ylabel("Count")
                        for container in ax.containers:
                            ax.bar_label(container)

                    elif chart_type == "Histogram":
                        x_to_plot = x_col if x_col != 'None' else y_col
                        sns.histplot(data=plot_df, x=x_to_plot, hue=hue_col if hue_col != "None" else None, kde=True)

                    elif chart_type == "Heatmap":
                        numeric_df = plot_df.select_dtypes(include=['number'])
                        if len(numeric_df.columns) < 2:
                            st.error("Heatmap requires at least two numeric columns.")
                            plt.close(fig)
                            st.stop()
                        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
                        final_plot_data_for_ai = numeric_df.corr()
                        x_col = 'Correlation Matrix'
                        y_col_for_ai = 'Correlation Value'

                    #  AI/Download Logic
                    plt.title(chart_title if chart_title else f"{chart_type} of {primary_col}")
                    plt.xticks(rotation=45, ha='right')

                    #APPLY AXIS LABELS
                    if x_label:
                        plt.xlabel(x_label)
                    if y_label:
                        plt.ylabel(y_label)


                    plt.tight_layout()
                    st.pyplot(fig)

                    # Store the figure for potential AI generation/reporting
                    st.session_state.current_chart_figure = fig
                    st.session_state.current_chart_data = {
                        "df_for_chart": final_plot_data_for_ai,
                        "chart_type": chart_type,
                        "x_col": x_col,
                        "y_col": y_col_for_ai,
                        "hue_col": hue_col
                    }
                    # Reset AI analysis state
                    st.session_state.show_ai_analysis = False
                    st.session_state.ai_analysis_result = None

                except Exception as e:
                    st.error(f"An error occurred while generating the plot: {e}")
                    plt.close(fig)

        # --- Feature 1: Conditional AI Insight Generation ---

        if st.session_state.get('current_chart_figure') is not None:
            if st.button("Give Insights and Conclusion (Uses API Credits)"):
                st.session_state.show_ai_analysis = True

                with st.spinner("Generating AI analysis..."):
                    params = st.session_state.current_chart_data
                    ai_analysis = get_ai_chart_analysis(
                        df_for_chart=params['df_for_chart'],
                        chart_type=params['chart_type'],
                        x_col=params['x_col'],
                        y_col=params['y_col'],
                        hue_col=params['hue_col'],
                        llm_instance=llm
                    )
                    st.session_state.ai_analysis_result = ai_analysis

                chart_data_to_store = {
                    "fig": st.session_state.current_chart_figure,
                    "insight": ai_analysis['insight'],
                    "conclusion": ai_analysis['conclusion'],
                    "generated_ai": True
                }

                st.session_state.chart_data.append(chart_data_to_store)

                perform_viz_action(
                    f"Generated standard '{params['chart_type']}' for {params['x_col']} with AI Analysis.",
                    {"action": "chart",
                     "params": {"type": params['chart_type'], "x": params['x_col'], "y": params['y_col']}}
                )
                st.rerun()

        if st.session_state.get('ai_analysis_result') is not None and st.session_state.show_ai_analysis:
            ai_analysis = st.session_state.ai_analysis_result
            st.info(f"**Insight:** {ai_analysis['insight']}")
            st.success(f"**Conclusion:** {ai_analysis['conclusion']}")

            if st.session_state.get('current_chart_figure') is not None:
                chart_bytes = io.BytesIO()
                st.session_state.current_chart_figure.savefig(chart_bytes, format='png', bbox_inches='tight')
                st.download_button(
                    label="Download Chart",
                    data=chart_bytes.getvalue(),
                    file_name=f"{chart_title or chart_type}.png",
                    mime="image/png"
                )
                plt.close(st.session_state.current_chart_figure)

            st.session_state.current_chart_figure = None
            st.session_state.ai_analysis_result = None
            st.session_state.current_chart_data = None
            st.session_state.show_ai_analysis = False

        st.markdown("---")
        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>Summary, KPI and Report Generation</h2>", unsafe_allow_html=True)
        st.markdown("Get a final summary of your data, including key insights and a professional PDF report.")

        # NEW CONTROL CHECKBOX
        generate_ai_summary = st.checkbox(
            "Include AI-Powered KPI & Conclusion in Report (Uses API Credits)",
            value=True,
            key="include_ai_summary_checkbox"
        )

        #  Display Area for Statistical Review
        if st.button("View Statistical Summary & Nulls"):
            # This button remains separate and does NOT call the API
            st.write("### Updated Statistical Summary")
            st.write(df.describe(include="all").transpose())
            st.write("### Null Values After Cleaning")
            st.write(df.isnull().sum())


        if st.button("Download Full Analysis Report"):

            kpi_suggestions = "KPI and Conclusion generation was skipped by user selection."
            conclusion = "Report generated without AI insights (user selection)."

            # --- CONDITIONAL AI GENERATION ---
            if generate_ai_summary:
                try:
                    with st.spinner("Generating AI Summary and Report..."):
                        sdf = SmartDataframe(df, config={"llm": llm})

                        # API Call 1: KPI Suggestions
                        kpi_prompt = (
                            f"Generate a detailed list of **Key Performance Indicators (KPIs)** based ONLY on the dataset. The response MUST be formatted as a numbered list of actionable points. Include specific data points, percentages, top categories, and strategic implications to support the claims."
                        )
                        kpi_suggestions = sdf.chat(kpi_prompt)

                        # API Call 2: Conclusion
                        conclusion_prompt = (
                            f"Based on the following KPI suggestions: {kpi_suggestions}. Please provide a brief, actionable conclusion for stakeholders. "
                            f"The conclusion should focus on 1-2 strategic recommendations."
                        )
                        conclusion = sdf.chat(conclusion_prompt)

                    st.success("AI analysis complete! Report ready for download.")

                except Exception as e:
                    st.error(f"Error generating AI content: {e}. Report generated without AI sections.")
                    kpi_suggestions = f"AI Generation Failed: {e}"
                    conclusion = "AI Conclusion Failed."

            # ---------------------------------

            # 4. Store results temporarily (Using the final values, whether generated or skipped)
            st.session_state.kpi_suggestions = kpi_suggestions
            st.session_state.conclusion_text = conclusion

            try:
                # 5. Generate and trigger download
                report_content = generate_pdf_report(
                    df,
                    st.session_state.history,
                    kpi_suggestions,
                    st.session_state.chart_data,
                    conclusion
                )

                st.download_button(
                    label="Download PDF Report",
                    data=report_content,
                    file_name="Data_Analysis_Report.pdf",
                    mime="application/pdf",
                    key="download_report"
                )

                # Optionally display the final outcome
                if generate_ai_summary:
                    st.subheader("Generated AI Insights (for Review)")
                    st.markdown(f"**KPI Suggestions:**")
                    st.markdown(kpi_suggestions)
                    st.markdown(f"**Conclusion:** {conclusion}")

            except Exception as e:
                st.error(f"Error during report finalization: {e}")

        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>Downloading Cleaned Data</h2>", unsafe_allow_html=True)
        st.markdown("Download your cleaned and transformed dataset as a CSV file for use in other applications.")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Cleaned Dataset",
            csv,
            "cleaned_dataset.csv",
            "text/csv",
            key='download-csv'
        )

        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>Load (L) Final Data</h2>", unsafe_allow_html=True)
        st.markdown("Load your cleaned and transformed dataset directly into a production database.")

        load_target = st.radio("Choose Target Database:", ["MySQL/RDS", "Snowflake"], key="load_target")
        table_name = st.text_input("Target Table Name (e.g., cleaned_data)", key="target_table_name")
        load_mode = st.radio("Load Mode:", ["Replace Existing Table", "Append to Existing Table"], key="load_mode")
        if_exists = 'replace' if load_mode == "Replace Existing Table" else 'append'

        if load_target == "MySQL/RDS":
            st.subheader("MySQL/RDS Credentials")
            # Using session state variables defined in the Upload section for persistence
            mysql_host_load = st.text_input("Host", value=st.session_state.mysql_host, key="mysql_host_load")
            mysql_user_load = st.text_input("User", value=st.session_state.mysql_user, key="mysql_user_load")
            mysql_password_load = st.text_input("Password", type="password", value=st.session_state.mysql_password,
                                                key="mysql_password_load")
            mysql_database_load = st.text_input("Database Name", value=st.session_state.mysql_database,
                                                key="mysql_database_load")

            if st.button("Load to MySQL/RDS"):
                if table_name and mysql_host_load and mysql_user_load and mysql_database_load:
                    with st.spinner("Loading data to MySQL..."):
                        success, message = load_to_mysql(
                            df=df,
                            host=mysql_host_load,
                            user=mysql_user_load,
                            password=mysql_password_load,
                            database=mysql_database_load,
                            table_name=table_name,
                            if_exists=if_exists
                        )
                        if success:
                            st.success(message)
                            st.session_state.history.append(
                                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded data to MySQL table: {table_name}")
                        else:
                            st.error(message)

        elif load_target == "Snowflake":
            st.subheader("Snowflake Credentials")
            sf_user = st.text_input("User", key="sf_user")
            sf_password = st.text_input("Password", type="password", key="sf_password")
            sf_account = st.text_input("Account Identifier (e.g., xyz12345)", key="sf_account")
            sf_warehouse = st.text_input("Warehouse (e.g., COMPUTE_WH)", key="sf_warehouse")
            sf_database = st.text_input("Database", key="sf_database")
            sf_schema = st.text_input("Schema", key="sf_schema")

            if st.button("Load to Snowflake"):
                if table_name and sf_user and sf_password and sf_account and sf_warehouse and sf_database and sf_schema:
                    with st.spinner("Loading data to Snowflake..."):
                        success, message = load_to_snowflake(
                            df=df,
                            user=sf_user,
                            password=sf_password,
                            account=sf_account,
                            warehouse=sf_warehouse,
                            database=sf_database,
                            schema=sf_schema,
                            table_name=table_name,
                            if_exists=if_exists
                        )
                        if success:
                            st.success(message)
                            st.session_state.history.append(
                                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded data to Snowflake table: {table_name}")
                        else:
                            st.error(message)


        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>History Log</h3>", unsafe_allow_html=True)
        st.markdown("Review the history of all the transformations and operations you have performed on your data.")
        for entry in reversed(st.session_state.history):
            st.write(entry)

        st.sidebar.markdown("---")
        st.sidebar.write("Save your current steps as a repeatable macro script.")
        if st.sidebar.download_button(
                "Download Macro Script",
                json.dumps(st.session_state.macro_actions, indent=2),
                "data_macro.json",
                "application/json",
                key="download_macro_script"
        ):
            st.sidebar.success("Macro script downloaded! You can re-run these steps on similar data.")

        if st.button("Reset Dataset"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.history = [
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset reset to original state."]
            st.session_state.df_history[st.session_state.active_df_key] = [st.session_state.df.copy()]
            st.session_state.redo_history = []
            st.session_state.chart_data = []
            st.session_state.kpi_suggestions = None
            st.session_state.conclusion_text = None
            st.session_state.macro_actions = []
            st.session_state.last_uploaded_file = None
            st.session_state.last_uploaded_files = None
            st.rerun()

    else:
        st.info(" Upload or load a dataset to get started.")
