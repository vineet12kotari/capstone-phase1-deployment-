import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
import re
import json
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

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

        # --- NEW, MORE PRECISE PROMPT FOR INSIGHT (Incorporating Heatmap Context) ---
        insight_prompt = (
            f"You are a data analyst. Your task is to provide one factual, data-driven insight. "
            f"Base your analysis STRICTLY on the data summary below. {heatmap_context} "
            f"The data was used for this visualization: {chart_context}. "
            f"Describe the most significant pattern, comparison, or finding from these numbers. "
            f"For example: 'The count for Category A (X) is Y% higher than for Category B (Z)'. "
            f"Your response must be a single, concise sentence. Here is the data:\n{data_summary}"
        )
        insight = sdf_local.chat(insight_prompt)

        # --- NEW, MORE PRECISE PROMPT FOR CONCLUSION ---
        conclusion_prompt = (
            f"Based *only* on the following insight: '{insight}'. "
            f"Provide a brief, strategic conclusion for a business stakeholder. What is the business implication? "
            f"Your response MUST BE A TEXTUAL SUMMARY. Do not generate a chart or a file path."
        )
        conclusion = sdf_local.chat(conclusion_prompt)

        # --- ADDED SAFEGUARD: Check for and handle file path responses ---
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


# -----------------------------
# Helper Functions (Keep these as they are, no need to change them)
# -----------------------------
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

            # Ensure the active_df_key has the loaded DF (the fully transformed one)
            if st.session_state.active_df_key not in st.session_state.dataframes:
                st.session_state.dataframes[st.session_state.active_df_key] = st.session_state.df.copy()
            else:
                # Update the DF in dataframes dict to match the current state loaded from "df" key
                st.session_state.dataframes[st.session_state.active_df_key] = st.session_state.df.copy()

            # Restore dtypes explicitly (Good logic, keeping it)
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

            # The *active* history list should contain the loaded transformed state.
            # When an action is performed, the current df is copied to the history *before* the action.
            # To simulate the start of a session after transformations, we should clear the
            # history for the active key so the *next* operation records the loaded state as its 'undo' point.
            # However, since perform_action *prepends* the current state before applying the change,
            # clearing the list works well, provided st.session_state.df is the loaded state.
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


# -----------------------------
# Session State Initialization
# -----------------------------
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

# --- NEW: Session State for Authentication ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# -----------------------------
# Authentication and Main UI Control
# -----------------------------
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
    # -----------------------------
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
    # Locate the 'Merge New Files' button logic (around line 1400 in the provided script)
    # ...

    if st.sidebar.button("Merge New Files"):
        if st.session_state.df is None:
            st.error("Please upload an initial dataset or load a session first to enable incremental loading.")
        elif not incremental_files:
            st.warning("Please select at least one new file to merge.")
        else:
            st.info("Merging new files with the existing dataset...")
            column_level_actions = [
                'remove_columns', 'split_column', 'change_dtype', 'merge_columns',
                'create_column', 'sort', 'replace_values', 'aggregate',
                # 'aggregate' is typically excluded from schema changes
                'create_column_arithmetic', 'create_column_single_arithmetic',
                'create_column_date_part', 'create_column_date_add_subtract'
            ]
            row_level_actions = [
                'remove_duplicates', 'fill_nulls', 'remove_rows_by_condition',
                'remove_outliers'
            ]
            column_macro = [action for action in st.session_state.macro_actions if
                            action['action'] in column_level_actions]
            row_macro = [action for action in st.session_state.macro_actions if action['action'] in row_level_actions]
            new_dfs = []
            new_files_to_process = []
            skipped_files = []
            current_processed_files = st.session_state.get('processed_files', [])

            # Define the target column order once
            target_columns = st.session_state.df.columns.tolist()

            for f in incremental_files:
                if f.name not in current_processed_files:
                    try:
                        new_df = load_data(f)
                        if new_df is not None:
                            # Apply column-level transformations to the new data
                            temp_col_df, macro_log, success = run_macro(new_df, column_macro)

                            # --- START OF FIX: Relaxing schema check and enforcing column order ---

                            # 1. Check if the set of columns matches (ignoring order)
                            is_schema_match = success and (set(temp_col_df.columns) == set(target_columns))

                            if is_schema_match:
                                # 2. Enforce the exact target column order before merging
                                temp_col_df = temp_col_df.reindex(columns=target_columns)
                                new_dfs.append(temp_col_df)
                                new_files_to_process.append(f.name)
                                st.session_state.history.extend(macro_log)
                            else:
                                st.error(
                                    f"Schema mismatch for {f.name} after applying column-level transformations. Skipping file. "
                                    f"New columns: {set(temp_col_df.columns) - set(target_columns) or 'None'}")
                                skipped_files.append(f.name)
                    # --- END OF FIX ---
                    except Exception as e:
                        st.error(f"Error loading or transforming incremental file {f.name}: {e}")
                        skipped_files.append(f.name)

            if new_dfs:
                combined_df = pd.concat([st.session_state.df] + new_dfs, ignore_index=True)
                # ... (rest of the block continues unchanged)
                final_df = combined_df
                if st.session_state.incremental_fill_option == 'Recalculate on Entire Dataset':
                    final_df, row_macro_log, success = run_macro(combined_df, row_macro)
                    st.session_state.history.extend(row_macro_log)
                    if not success:
                        st.error("Error applying row-level transformations after merge. Check logs.")
                st.session_state.df = final_df
                st.success(f"Successfully merged {len(new_dfs)} new files and applied transformations.")
                st.session_state.history.append(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Merged and processed new files: {', '.join(new_files_to_process)}"
                )
                st.session_state.processed_files.extend(new_files_to_process)
                st.session_state.original_df = st.session_state.df.copy()
                st.session_state.df_history[st.session_state.active_df_key].append(st.session_state.df.copy())
                st.rerun()
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
        st.write("### Restored Data Preview")
        st.dataframe(st.session_state.df.head())
        st.write("### Restored History Log")
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

        # --- ADD THE NULL VALUES CODE SNIPPET HERE ---
        st.write("### Null Values Per Column")
        st.markdown("This table shows the count of missing (null/NaN) values for each column.")
        null_counts = df.isnull().sum()
        st.dataframe(null_counts[null_counts > 0].to_frame(name="Null Count"))
        if null_counts.sum() == 0:
            st.info("No null values found in the dataset. Data quality is excellent! ")
        # --- END OF NULL VALUES CODE SNIPPET ---

        st.write("### Statistical Summary")
        st.markdown(
            "Get a quick statistical summary of your numerical columns (mean, min, max, etc.) and a count of unique values for all columns.")
        st.write(df.describe(include="all").transpose())

        st.write("### Statistical Summary")
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

        st.write("### Sort Dataset")
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
        # -----------------------------
        # Custom Aggregation (updated)
        # -----------------------------
        st.subheader("Custom Aggregation")
        st.markdown(
            "Analyze the frequency of values, optionally grouped by other columns. "
            "Use *Value Counts* to see frequency distributions, or *Aggregate* to compute numeric aggregations."
        )
        st.info(
            "If you choose Aggregate without group-by, aggregation will be applied across the whole dataset (per column)."
        )

        target_cols_agg = st.multiselect(
            "Select target column(s) for aggregation (optional - leave empty to count group combinations)",
            df.columns, key="agg_target"
        )
        groupby_cols_agg = st.multiselect(
            "Select columns to group by (optional)",
            df.columns, key="agg_groupby"
        )

        # --- MODIFIED LOGIC START ---

        # Determine if at least one selected target column is numeric
        is_any_target_numeric = bool(target_cols_agg) and any(
            pd.api.types.is_numeric_dtype(df[col]) for col in target_cols_agg
        )

        # If no group-by columns are selected, we can only do aggregate if we have numeric targets
        # If group-by columns are selected, we can always do Value Counts (count)
        allow_numeric_aggregate = bool(groupby_cols_agg) and is_any_target_numeric

        # Default to Value Counts, but offer Aggregate if numeric targets and/or group-by columns are selected
        agg_type = "Value Counts"
        if is_any_target_numeric or groupby_cols_agg:
            agg_options = ["Value Counts"]
            if is_any_target_numeric:
                agg_options.append("Aggregate")

            agg_type = st.radio("Choose aggregation type:", agg_options, key="agg_mode")

        aggregation_mapping = {}

        # If user picked Aggregate, choose function(s)
        if agg_type == "Aggregate" and is_any_target_numeric:

            # Filter for only numeric target columns to apply functions like mean/median/sum
            numeric_target_cols = [col for col in target_cols_agg if pd.api.types.is_numeric_dtype(df[col])]

            # Allow user to select multiple aggregation functions
            agg_funcs = st.multiselect(
                "Select aggregation function(s) to apply to target column(s)",
                ["mean", "median", "sum", "min", "max", "count", "std", "var"],
                default=["mean"] if "mean" in ["mean", "median", "sum", "min", "max", "count", "std", "var"] else None,
                key="agg_func_select"
            )

            if agg_funcs and numeric_target_cols:
                # Create mapping where each numeric target column gets all selected functions
                for col in numeric_target_cols:
                    aggregation_mapping[col] = agg_funcs

                # If there are non-numeric targets, we can only count them
                non_numeric_target_cols = [col for col in target_cols_agg if not pd.api.types.is_numeric_dtype(df[col])]
                for col in non_numeric_target_cols:
                    if "count" in agg_funcs:
                        aggregation_mapping[col] = "count"

        # --- MODIFIED LOGIC END ---

        # Sorting UI (This part remains mostly the same, but dynamically updated)
        sort_order_agg = st.radio("Sort order for the result", ["Descending", "Ascending"], key="sort_order_agg")

        # Prepare sort options depending on result shape
        sort_by_options = []
        if groupby_cols_agg:
            sort_by_options.extend(groupby_cols_agg)

        if agg_type == "Value Counts":
            sort_by_options.append("count")
        elif agg_type == "Aggregate" and aggregation_mapping:
            # Use the first column/function pair for default sorting options
            for col, funcs in aggregation_mapping.items():
                if isinstance(funcs, list):
                    sort_by_options.extend([f"{col}_{func}" for func in funcs])
                else:  # Handle single function case from old logic for robustness, though new is list
                    sort_by_options.append(f"{col}_{funcs}")

        sort_by_col_agg = st.selectbox("Sort the result by this column:", ["None"] + sort_by_options,
                                       key="sort_by_col_agg")

        # ... (The rest of the `if st.button("Apply Analysis"):` logic handles the DataFrame aggregation
        #      and is robust enough to handle the updated `aggregation_mapping` structure.)

        if st.button("Apply Analysis"):
            if not target_cols_agg and not groupby_cols_agg:
                st.warning("Please select at least one target column or at least one group-by column.")
            else:
                try:
                    # ----- VALUE COUNTS -----
                    if agg_type == "Value Counts":
                        if groupby_cols_agg:
                            # If target selected -> counts per (groupby... , target)
                            if target_cols_agg:
                                target_col = target_cols_agg[0]  # use first target for value counts
                                agg_result = (
                                    df.groupby(groupby_cols_agg + [target_col])
                                    .size()
                                    .reset_index(name="count")
                                )
                            else:
                                # no target; counts of each group combination
                                agg_result = (
                                    df.groupby(groupby_cols_agg)
                                    .size()
                                    .reset_index(name="count")
                                )
                        else:
                            # no groupby; simple value_counts on target
                            target_col = target_cols_agg[0]
                            agg_result = df[target_col].value_counts().to_frame(name="count").reset_index()
                            agg_result.columns = [target_col, "count"]

                    # ----- AGGREGATE -----
                    else:  # agg_type == "Aggregate"
                        if groupby_cols_agg:
                            # groupby + aggregation mapping
                            agg_result = df.groupby(groupby_cols_agg).agg(aggregation_mapping).reset_index()
                        else:
                            # No group-by: apply aggregation to the entire dataframe for selected columns
                            # result will be 1-row summary: e.g., col1_mean, col2_mean ...
                            temp = df[list(aggregation_mapping.keys())].agg(aggregation_mapping)
                            # temp might be a Series (if single func + multiple cols) or DataFrame (if multi-func). Normalize to DataFrame
                            if isinstance(temp, pd.Series):
                                agg_result = temp.to_frame().T
                            else:
                                agg_result = pd.DataFrame(temp)
                                # if columns are MultiIndex, flatten them
                                if isinstance(agg_result.columns, pd.MultiIndex):
                                    agg_result.columns = ["_".join(map(str, col)).strip() for col in
                                                          agg_result.columns.values]
                                agg_result = agg_result.reset_index(drop=True)

                    # Ensure DataFrame type
                    agg_result = pd.DataFrame(agg_result)

                    # Flatten MultiIndex columns if present
                    if isinstance(agg_result.columns, pd.MultiIndex):
                        agg_result.columns = ["_".join([str(c) for c in col]).strip() for col in
                                              agg_result.columns.values]

                    # Sorting if requested
                    if sort_by_col_agg != "None" and sort_by_col_agg in agg_result.columns:
                        agg_result = agg_result.sort_values(by=sort_by_col_agg,
                                                            ascending=(sort_order_agg == "Ascending"))
                    else:
                        # default sort for counts
                        if "count" in agg_result.columns:
                            agg_result = agg_result.sort_values(by="count", ascending=(sort_order_agg == "Ascending"))

                    st.write("#### Analysis Result:")
                    st.dataframe(agg_result)

                    # --- AI conclusion (safe) ---
                    try:
                        openai_key = os.getenv("OPENAI_API_KEY", None)
                        if openai_key:
                            # only call pandasai if agg_result is a non-empty DataFrame
                            if isinstance(agg_result, pd.DataFrame) and not agg_result.empty:
                                sdf_local = SmartDataframe(agg_result, config={"llm": OpenAI(api_token=openai_key)})
                                agg_conclusion_prompt = (
                                    "Based on this aggregation table, provide a concise and actionable conclusion for stakeholders. "
                                    "Mention the key findings and any notable trends or outliers."
                                )
                                agg_conclusion = sdf_local.chat(agg_conclusion_prompt)
                                st.info(f"Conclusion: {agg_conclusion}")
                            else:
                                st.info("No valid aggregation result to generate AI conclusion.")
                        else:
                            st.info("AI not available (OPENAI_API_KEY not set). Skipping automatic conclusions.")
                    except Exception as e:
                        st.warning(f"AI conclusion not available: {e}")

                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    st.write("Detailed error:", e)
        st.markdown("---")
        # -----------------------------
        # Joining Datasets (improved)
        # -----------------------------
        import re





        # -------------------------------------------------------------------
        # START: CORRECTED AI QUESTION SECTION
        # -------------------------------------------------------------------
        # ... (rest of the code remains unchanged)


        # -------------------------------------------------------------------
        # START: CORRECTED AI QUESTION SECTION
        # -------------------------------------------------------------------
        st.write("### Ask a Question")
        st.markdown(
            "Use this powerful feature to ask any question about your data in natural language. "
            "The AI will generate code and provide a human-readable answer."
        )

        language_map = {
            "English": "en-US", "Spanish": "es-ES", "French": "fr-FR",
            "German": "de-DE", "Hindi": "hi-IN", "Chinese (Mandarin)": "zh-CN",
            "Japanese": "ja-JP", "Arabic": "ar-SA"
        }
        selected_language = st.selectbox("Choose Language for Voice Input", list(language_map.keys()), index=0)
        selected_language_code = language_map[selected_language]

        # --- Input Handling ---
        question = None
        query_from_text = st.text_input("Enter your question about the data:")

        if st.button("Ask AI with Your Voice"):
            question = recognize_speech(selected_language_code)

        # Process text input only if it's new
        elif query_from_text and query_from_text != st.session_state.last_question:
            question = query_from_text

        # --- Unified AI Processing ---
        # If there's a new question, process it
        if question:
            st.session_state.last_question = question
            st.session_state.show_code = False  # Reset code visibility for new question
            try:
                with st.spinner("Thinking..."):
                    sdf = SmartDataframe(df, config={"llm": llm})
                    response = sdf.chat(question)
                    st.session_state.ai_response = response

                    # --- THIS IS THE CORRECTED PART ---
                    # Use the 'last_code_generated' attribute which is available in newer versions
                    if hasattr(sdf, 'last_code_generated') and sdf.last_code_generated:
                        st.session_state.last_ai_code = sdf.last_code_generated
                    else:
                        st.session_state.last_ai_code = "# Code not available for this query or no code was generated."
                    # --- END OF CORRECTION ---

            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.ai_response = None

        # --- Decoupled Display Logic ---
        # This section's job is ONLY to display whatever is in the session_state.
        if st.session_state.ai_response:
            st.write(f"> Your question: *{st.session_state.last_question}*")
            st.write("#### AI Response:")
            st.write(st.session_state.ai_response)

            if st.session_state.last_ai_code:
                # This button's only job is to flip the 'show_code' boolean state
                if st.button("Know Code"):
                    st.session_state.show_code = not st.session_state.show_code

            # The code is displayed here ONLY if the 'show_code' state is True
            if st.session_state.show_code:
                st.write("---")
                st.write("#### Code used to generate response:")
                st.code(st.session_state.last_ai_code, language='python')
        # -------------------------------------------------------------------
        # END: CORRECTED AI QUESTION SECTION
        # -------------------------------------------------------------------
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

        # -----------------------------
        # Replace Specific Cell Values
        # -----------------------------
        import re  # make sure this import is available at top of file (you can also keep it here)

        st.write("### Replace Specific Cell Values")
        st.markdown(
            "Replace specific values in a column, either by manually entering old/new values or by applying a condition.\n\n"
            "*Condition examples:* >18, <=100, ==Yes, !=No, >=2 & <=4 (use & or and for AND, | or or for OR)."
        )
        replace_col = st.selectbox("Select column to replace values", ["None"] + list(st.session_state.df.columns),
                                   key='replace_col_select')
        if replace_col != "None":
            replace_method = st.radio("Choose replacement method:", ["Manual", "By Condition"], key="replace_method")

            # ---------- Manual Replacement ----------
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

            # ---------- Conditional Replacement ----------
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

                            # --- MODIFIED: Try conversion to numeric always for comparison operators ---
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

        st.write("### Merge Columns")
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

        st.write("### Create New Column from Conditions")
        st.markdown(
            "Create a new categorical column based on conditions applied to an existing column. This is useful for grouping data into meaningful categories.")
        col_to_categorize = st.selectbox("Select a column to create conditions on", ["None"] + list(df.columns),
                                         key='cat_col_select')
        if col_to_categorize != "None":
            new_category_col = st.text_input("Enter new column name (e.g., 'Rating_category')")
            num_conditions = st.number_input("Number of conditions", min_value=1, value=1, step=1, key='num_cond_input')
            conditions = []
            labels = []
            st.write("#### Define Conditions and Labels")
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

        st.write("### Create New Column with Arithmetic Operations")
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
        st.markdown("<h2 style='text-align: center;'>Generate Visualization</h2>", unsafe_allow_html=True)
        st.markdown(
            "Create compelling charts and graphs to visualize trends, distributions, and relationships in your data.")
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Count Plot", "Histogram", "Box Plot", "Heatmap"],
            key="chart_type_select"
        )

        st.markdown("##### Advanced Plotting with Grouped Data")
        st.info(
            "Create a temporary aggregated DataFrame and plot it. Useful for charts like 'Average Price by Hotel Type'.")
        temp_plot_name = st.text_input("Temporary plot name (e.g., 'Hotel_Price_Dist')")
        temp_groupby = st.multiselect("Group by columns", df.columns, key="temp_groupby")
        temp_agg_col = st.selectbox("Aggregation column", ["None"] + list(df.select_dtypes(include='number').columns),
                                    key="temp_agg_col")
        temp_agg_func = st.selectbox("Aggregation function", ["mean", "sum", "count", "min", "max"],
                                     key="temp_agg_func")

        sort_advanced_by = st.selectbox("Sort data by", ["None"] + [temp_agg_col], key="sort_advanced_by")
        sort_advanced_order = st.radio("Sort order", ["Descending", "Ascending"], key="sort_advanced_order")
        head_n = st.number_input("Show top N rows (0 for all)", min_value=0, step=1, key="head_n")

        st.markdown("---")  # Add a separator before the plot buttons

        if st.button("Generate Advanced Plot"):
            if temp_plot_name and temp_groupby and temp_agg_col != "None":
                try:
                    # Create the aggregated DataFrame that will be used for plotting and AI analysis
                    temp_df = df.groupby(temp_groupby)[temp_agg_col].agg(temp_agg_func).reset_index()

                    if sort_advanced_by != "None" and sort_advanced_by in temp_df.columns:
                        temp_df = temp_df.sort_values(by=sort_advanced_by,
                                                      ascending=(sort_advanced_order == "Ascending"))
                    if head_n > 0:
                        temp_df = temp_df.head(head_n)

                    if len(temp_groupby) > 1:
                        temp_df[temp_plot_name] = temp_df[temp_groupby].astype(str).agg(' - '.join, axis=1)
                    else:
                        temp_df[temp_plot_name] = temp_df[temp_groupby[0]]

                    st.subheader(f"Aggregated Data for Plot: {temp_plot_name}")
                    st.dataframe(temp_df)

                    fig = plt.figure(figsize=(10, 6))

                    if chart_type == "Bar Chart":
                        ax = sns.barplot(data=temp_df, x=temp_plot_name, y=temp_agg_col)
                        for container in ax.containers:
                            ax.bar_label(container)
                    elif chart_type == "Line Chart":
                        sns.lineplot(data=temp_df, x=temp_plot_name, y=temp_agg_col)
                    elif chart_type == "Scatter Plot":
                        sns.scatterplot(data=temp_df, x=temp_plot_name, y=temp_agg_col)
                    elif chart_type == "Pie Chart":
                        plt.pie(temp_df[temp_agg_col], labels=temp_df[temp_plot_name], autopct='%1.1f%%', startangle=90,
                                pctdistance=0.85)
                        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
                    else:
                        st.warning(f"Chart type '{chart_type}' is not supported for advanced plotting.")
                        plt.close(fig)  # Close the figure if not used
                        st.stop()

                    plt.title(f"{temp_agg_func.capitalize()} of {temp_agg_col} by {', '.join(temp_groupby)}")
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)

                    # --- Generate and Display AI Analysis ---
                    with st.spinner("Generating AI analysis..."):
                        ai_analysis = get_ai_chart_analysis(
                            df_for_chart=temp_df,
                            chart_type=chart_type,
                            x_col=temp_plot_name,
                            y_col=temp_agg_col,
                            llm_instance=llm
                        )

                        st.info(f"**Insight:** {ai_analysis['insight']}")
                        st.success(f"**Conclusion:** {ai_analysis['conclusion']}")

                        # Store the complete chart data for reporting
                        chart_data_to_store = {
                            "fig": fig,
                            "insight": ai_analysis['insight'],
                            "conclusion": ai_analysis['conclusion']
                        }
                        perform_viz_action(
                            f"Generated '{chart_type}' for {temp_agg_col} by {', '.join(temp_groupby)}",
                            {"action": "chart", "params": {"type": chart_type, "x": temp_plot_name, "y": temp_agg_col}},
                            chart_data=chart_data_to_store
                        )

                    chart_bytes = io.BytesIO()
                    fig.savefig(chart_bytes, format='png', bbox_inches='tight')
                    st.download_button(
                        label="Download Chart",
                        data=chart_bytes.getvalue(),
                        file_name=f"{temp_plot_name}.png",
                        mime="image/png"
                    )
                    plt.close(fig)

                except Exception as e:
                    st.error(f"Error creating advanced plot: {e}")
            else:
                st.warning("Please fill in all fields to create an advanced plot.")

        st.markdown("---")
        # --- Standard Plotting ---
        # --- Standard Plotting ---
        st.markdown("##### Standard Plotting")
        st.info(
            "You can apply a filter and then group the data before plotting. (e.g., plot 'Survived' for 'Adults' only)"
        )

        # Filtering
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

        # Chart title
        chart_title = st.text_input("Enter chart title:", key="chart_title")

        # Grouping (only available for certain charts in standard mode)
        grouped_cols_viz = []  # Simplified for clarity

        x_col, y_col, hue_col, agg_method = None, None, None, None

        # Axis and aggregation options
        if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Count Plot", "Histogram"]:

            # MODIFICATION: Added "None" to X-axis options
            x_col_options = ["None"] + filtered_df.columns.tolist()
            x_col = st.selectbox("Select X-axis Column", x_col_options, key="chart_x_col")

            if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot"]:
                y_col = st.selectbox("Select Y-axis Column", ["None"] + list(filtered_df.columns), key="chart_y_col")

                # Show aggregation only if a Y-column is selected and is numeric
                if y_col != "None" and pd.api.types.is_numeric_dtype(filtered_df[y_col]):
                    agg_method = st.selectbox(
                        "Select aggregation method for the Y-axis",
                        ["mean", "sum", "count", "min", "max", "median"],
                        key="standard_agg_method"
                    )
                elif y_col != "None":
                    st.warning(f"Aggregation is not available for non-numeric column '{y_col}'.")

            hue_col = st.selectbox(
                "Select a column for 'Hue' (optional)", ["None"] + filtered_df.columns.tolist(), key="chart_hue_col"
            )

            # Data validation for single-variable plots
            if chart_type in ["Count Plot", "Histogram", "Box Plot"]:
                # Ensure at least one axis is selected for single-variable plots
                if x_col == "None" and y_col == "None":
                    st.warning(f"Please select at least one column (X or Y) for a {chart_type}.")


        elif chart_type == "Pie Chart":
            x_col = st.selectbox("Select Column for Pie Chart", filtered_df.columns, key="chart_x_col")
        elif chart_type == "Heatmap":
            st.info("Heatmap will show correlations between all numeric columns.")

        # --- Generate Plot Button ---
        if st.button("Generate Standard Plot"):
            if filtered_df.empty:
                st.error("Cannot generate plot on an empty (filtered) dataset.")
            # Check for minimum required columns based on chart type
            elif chart_type == "Pie Chart" and x_col is None:
                st.error("Please select a column for the Pie Chart.")
            elif chart_type == "Scatter Plot" and (x_col == 'None' or y_col == 'None'):
                st.error("Scatter plots require both X-axis and Y-axis columns.")
            elif chart_type in ["Bar Chart", "Line Chart", "Box Plot"] and x_col == 'None' and y_col == 'None':
                st.error(f"{chart_type} requires at least one column (X or Y) to plot.")
            else:
                fig = plt.figure(figsize=(10, 6))
                plot_df = filtered_df.copy()
                final_plot_data_for_ai = plot_df  # Default to the filtered df

                # Determine the primary plotting column for single-variable charts (Box/Count/Hist)
                primary_col = x_col if x_col != 'None' else y_col

                # Set a default Y column for AI context if it's a count-based plot
                y_col_for_ai = y_col
                if y_col == 'None' and primary_col != 'None' and chart_type in ["Bar Chart", "Count Plot", "Histogram",
                                                                                "Pie Chart"]:
                    y_col_for_ai = 'count'
                elif y_col == 'None' and chart_type == 'Box Plot' and primary_col != 'None':
                    y_col_for_ai = 'value'

                try:
                    # --- Plotting Logic ---
                    if chart_type == "Bar Chart":
                        if y_col != 'None' and pd.api.types.is_numeric_dtype(plot_df[y_col]):
                            final_plot_data_for_ai = plot_df.groupby(x_col)[y_col].agg(agg_method).reset_index()
                            ax = sns.barplot(data=final_plot_data_for_ai, x=x_col, y=y_col,
                                             hue=hue_col if hue_col != "None" else None)
                            plt.ylabel(f"{agg_method.capitalize()} of {y_col}")
                        else:  # Count plot style
                            final_plot_data_for_ai = plot_df[x_col].value_counts().to_frame().reset_index()
                            final_plot_data_for_ai.columns = [x_col, 'count']
                            y_col_for_ai = 'count'  # Set y_col for AI analysis
                            ax = sns.barplot(data=final_plot_data_for_ai, x=x_col, y='count',
                                             hue=hue_col if hue_col != "None" else None)
                            plt.ylabel("Count")
                        for container in ax.containers:
                            ax.bar_label(container)

                    elif chart_type == "Line Chart":
                        if x_col == 'None' or y_col == 'None' or not pd.api.types.is_numeric_dtype(plot_df[y_col]):
                            st.error("Line charts require both X-axis and a numeric Y-axis column for aggregation.")
                            plt.close(fig)
                            st.stop()

                        final_plot_data_for_ai = plot_df.groupby(x_col)[y_col].agg(
                            agg_method).reset_index().sort_values(by=x_col)
                        sns.lineplot(data=final_plot_data_for_ai, x=x_col, y=y_col,
                                     hue=hue_col if hue_col != "None" else None)
                        plt.ylabel(f"{agg_method.capitalize()} of {y_col}")

                    elif chart_type == "Scatter Plot":
                        sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=hue_col if hue_col != "None" else None)

                    elif chart_type == "Box Plot":
                        # If x_col is None, use y_col as the main variable
                        if x_col == 'None':
                            sns.boxplot(data=plot_df, y=y_col if y_col != 'None' else None,
                                        hue=hue_col if hue_col != "None" else None)
                            x_col = y_col  # Use y_col as primary for AI analysis
                            y_col_for_ai = 'value'
                        # If y_col is None, use x_col as the main variable
                        elif y_col == 'None':
                            sns.boxplot(data=plot_df, y=x_col,
                                        hue=hue_col if hue_col != "None" else None)
                            x_col = x_col  # x_col is already the primary
                            y_col_for_ai = 'value'
                        # If both are set, plot Y against X (grouped distribution)
                        else:
                            sns.boxplot(data=plot_df, x=x_col, y=y_col,
                                        hue=hue_col if hue_col != "None" else None)
                            y_col_for_ai = y_col

                    elif chart_type == "Pie Chart":
                        counts = plot_df[x_col].value_counts()
                        final_plot_data_for_ai = counts.to_frame().reset_index()
                        final_plot_data_for_ai.columns = [x_col, 'count']
                        y_col_for_ai = 'count'
                        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
                        plt.legend(counts.index, bbox_to_anchor=(1.05, 1), loc='upper left')

                    elif chart_type == "Count Plot":
                        if x_col != 'None':
                            ax = sns.countplot(data=plot_df, x=x_col, hue=hue_col if hue_col != "None" else None,
                                               order=plot_df[x_col].value_counts().index)
                            y_col_for_ai = 'count'
                            for container in ax.containers:
                                ax.bar_label(container)
                        elif y_col != 'None':  # Plot count vertically
                            ax = sns.countplot(data=plot_df, y=y_col, hue=hue_col if hue_col != "None" else None,
                                               order=plot_df[y_col].value_counts().index)
                            x_col = y_col  # Use y_col as primary for AI analysis
                            y_col_for_ai = 'count'
                        else:
                            st.error("Count Plot requires at least one column (X or Y).")
                            plt.close(fig)
                            st.stop()

                    elif chart_type == "Histogram":
                        if x_col != 'None':
                            sns.histplot(data=plot_df, x=x_col, hue=hue_col if hue_col != "None" else None, kde=True)
                        elif y_col != 'None':
                            sns.histplot(data=plot_df, y=y_col, hue=hue_col if hue_col != "None" else None, kde=True)
                            x_col = y_col  # Use y_col as primary for AI analysis
                        else:
                            st.error("Histogram requires at least one column (X or Y).")
                            plt.close(fig)
                            st.stop()
                        y_col_for_ai = 'frequency'  # for AI analysis context

                    elif chart_type == "Heatmap":
                        numeric_df = plot_df.select_dtypes(include=['number'])
                        if len(numeric_df.columns) < 2:
                            st.error("Heatmap requires at least two numeric columns.")
                            plt.close(fig)
                            st.stop()
                        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
                        final_plot_data_for_ai = numeric_df.corr()  # Use correlation matrix for AI
                        x_col = 'Correlation Matrix'
                        y_col_for_ai = 'Correlation Value'

                    # --- Final Touches and Display ---
                    plt.title(chart_title if chart_title else f"{chart_type} of {primary_col}")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)

                    # --- Generate and Display AI Analysis ---
                    with st.spinner("Generating AI analysis..."):
                        ai_analysis = get_ai_chart_analysis(
                            df_for_chart=final_plot_data_for_ai,
                            chart_type=chart_type,
                            x_col=x_col,
                            y_col=y_col_for_ai,
                            hue_col=hue_col,
                            llm_instance=llm
                        )

                        st.info(f"**Insight:** {ai_analysis['insight']}")
                        st.success(f"**Conclusion:** {ai_analysis['conclusion']}")

                        # Store the complete chart data for reporting
                        chart_data_to_store = {
                            "fig": fig,
                            "insight": ai_analysis['insight'],
                            "conclusion": ai_analysis['conclusion']
                        }
                        perform_viz_action(
                            f"Generated standard '{chart_type}' for {x_col}",
                            {"action": "chart", "params": {"type": chart_type, "x": x_col, "y": y_col}},
                            chart_data=chart_data_to_store
                        )

                    # --- Download Button ---
                    chart_bytes = io.BytesIO()
                    fig.savefig(chart_bytes, format='png', bbox_inches='tight')
                    st.download_button(
                        label="Download Chart",
                        data=chart_bytes.getvalue(),
                        file_name=f"{chart_title or chart_type}.png",
                        mime="image/png"
                    )
                    plt.close(fig)

                except Exception as e:
                    st.error(f"An error occurred while generating the plot: {e}")
                    plt.close(fig)
        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>Summary, KPI and Report Generation</h2>", unsafe_allow_html=True)
        st.markdown("Get a final summary of your data, including key insights and a professional PDF report.")
        st.info(
            "Note: Please click 'View Final Summary & KPIs' before attempting to download the report to ensure all sections are populated.")

        if st.button("View Final Summary & KPIs"):
            # --- ADD THIS LINE ---
            # Create the SmartDataframe instance here to make it available for this section.
            sdf = SmartDataframe(df, config={"llm": llm})
            # ---------------------

            st.write("### Updated Statistical Summary")
            st.write(df.describe(include="all").transpose())
            st.write("### Null Values After Cleaning")
            st.write(df.isnull().sum())
            st.write("### AI-Powered KPI Suggestions")

            with st.spinner("Generating KPI suggestions..."):
                try:
                    kpi_suggestions = sdf.chat(
                        f"Generate a detailed, paragraph-based conclusion for stakeholders based on the dataset. Include key trends (e.g., year-by-year changes), top-performing categories, revenue/rating distributions, and any notable correlations. The report should be easy for a business user to understand, and it must include specific numbers, percentages, and data points to support the claims.")
                    st.session_state.kpi_suggestions = kpi_suggestions
                    st.write(f"Suggested KPIs: {kpi_suggestions}")
                except Exception as e:
                    st.error(f"Error generating KPI suggestions: {e}")

            st.subheader("AI-Powered Conclusion")
            st.info("The AI-generated conclusion will focus on actionable insights for stakeholders.")

            with st.spinner("Generating final conclusion..."):
                try:
                    conclusion_prompt = f"Based on the following KPI suggestions: {st.session_state.kpi_suggestions}. Please provide a brief, actionable conclusion for stakeholders. For example, if the KPI suggests 'Region 2 has the highest sales,' the conclusion should be 'Stakeholders should focus on marketing efforts in Region 2 to maximize sales.'"
                    conclusion = sdf.chat(conclusion_prompt)
                    st.session_state.conclusion_text = conclusion
                    st.write(f"**Conclusion:** {conclusion}")
                except Exception as e:
                    st.error(f"Error generating AI conclusion: {e}")

        if st.button("Download Full Analysis Report"):
            try:
                if 'kpi_suggestions' not in st.session_state or 'conclusion_text' not in st.session_state or not st.session_state.kpi_suggestions:
                    st.error(
                        "Please generate KPI suggestions and the conclusion first by clicking 'View Final Summary & KPIs'.")
                else:
                    kpi_suggestions = st.session_state.kpi_suggestions
                    conclusion = st.session_state.conclusion_text
                    report_content = generate_pdf_report(
                        df, st.session_state.history, kpi_suggestions,
                        st.session_state.chart_data, conclusion
                    )
                    st.download_button(
                        label="Download PDF Report",
                        data=report_content,
                        file_name="Data_Analysis_Report.pdf",
                        mime="application/pdf",
                        key="download_report"
                    )
            except Exception as e:
                st.error(f"Error generating PDF report: {e}")

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