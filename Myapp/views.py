from django.views.decorators.csrf import ensure_csrf_cookie
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
import re
import itertools
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from django.db.models import Count
from django.db.models.functions import TruncMonth, ExtractMonth
import os
import glob
import warnings
import traceback
from django.views.decorators.csrf import csrf_exempt
import joblib

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

def create_lstm_model():
    """Create LSTM model with proper session management"""
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=(3, 1)),
        LSTM(32, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def parse_log_line(line):
    """Enhanced log parsing with multiple pattern support"""
    patterns = [
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (\w+)\s+\[(.*?)\] - (.*)$",
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (\w+) \[(.*?)\] - (.*)$",
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (\w+) \[(.*?)\] - (.*)$",
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) (\w+) \[(.*?)\] (.*)$"
    ]
    
    for pattern in patterns:
        match = re.match(pattern, line)
        if match:
            try:
                dt_str = match.group(1)
                if ',' in dt_str:
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S,%f")
                else:
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                level = match.group(2).strip()
                thread = match.group(3).strip()
                message = match.group(4).strip()
                return dt, level, thread, message
            except (ValueError, IndexError) as e:
                continue
    return None, None, None, None


def get_logs_from_session(request, level=None):
    """
    Helper function to get and filter logs from the session.
    `level` can be 'INFO', 'WARN', 'ERROR', or None (for all).
    """
    log_content = request.session.get('uploaded_log_file')
    if not log_content:
        return ["Log file not found in session. Please upload a file from the 'Predictions' page first."]

    lines = log_content.strip().split('\n')
    
    if not level:
        return lines # Return all lines if no level is specified

    filtered_lines = []
    # Define keywords for each level
    keywords = {
        'INFO': ['INFO'],
        'WARN': ['WARN', 'WARNING'],
        'ERROR': ['ERROR']
    }
    
    if level in keywords:
        search_terms = keywords[level]
        for line in lines:
            # Case-insensitive search for keywords
            if any(term in line.upper() for term in search_terms):
                filtered_lines.append(line)
    
    return filtered_lines if filtered_lines else [f"No logs found for level: {level}"]

def prepare_sequences(series, steps=3):
    """Prepare sequences for LSTM training"""
    X, y = [], []
    for i in range(len(series) - steps):
        X.append(series[i:i + steps])
        y.append(series[i + steps])
    return np.array(X), np.array(y)

def process_log_data_for_charts(log_data):
    """Process log data to create chart data"""
    if not log_data:
        return {}
    
    lines = [line.strip() for line in log_data.splitlines() if line.strip()]
    parsed = []
    
    for line in lines:
        parsed_line = parse_log_line(line)
        if parsed_line[0] is not None:
            parsed.append(parsed_line)
    
    if not parsed:
        return {}
    
    log_df = pd.DataFrame(parsed, columns=["date", "level", "thread", "message"])
    log_df = log_df.dropna()
    
    if log_df.empty:
        return {}
    
    # Create monthly data
    log_df["month"] = log_df["date"].dt.to_period("M")
    log_df["month_str"] = log_df["date"].dt.strftime("%Y-%m")
    
    # Monthly counts by level
    monthly_counts = log_df.groupby(['month_str', 'level']).size().reset_index(name='count')
    
    # Monthly message counts
    monthly_message_counts = log_df.groupby(['month_str', 'message']).size().reset_index(name='count')
    
    # Error analysis
    error_logs = log_df[log_df["level"] == "ERROR"]
    error_monthly = error_logs.groupby(['month_str', 'message']).size().reset_index(name='count')
    
    # Temperature and battery data (simulated for demo)
    temperature_data = {
        'months': monthly_counts['month_str'].unique().tolist(),
        'values': [25 + (i * 5) % 40 for i in range(len(monthly_counts['month_str'].unique()))]
    }
    
    battery_data = {
        'charge': [100 - (i * 10) % 80 for i in range(len(monthly_counts['month_str'].unique()))],
        'time_remaining': [120 - (i * 15) % 100 for i in range(len(monthly_counts['month_str'].unique()))]
    }
    
    return {
        'monthly_counts': monthly_counts,
        'monthly_message_counts': monthly_message_counts,
        'error_monthly': error_monthly,
        'temperature_data': temperature_data,
        'battery_data': battery_data,
        'log_levels': log_df['level'].unique().tolist(),
        'total_logs': len(log_df),
        'date_range': {
            'start': log_df['date'].min().strftime('%Y-%m-%d'),
            'end': log_df['date'].max().strftime('%Y-%m-%d')
        }
    }

def LSTM_Forecast_from_session(request, input_msg):
    """LSTM forecasting logic with enhanced error handling"""
    result = {
        'error_msg': None, 
        'forecast_table': None, 
        'forecast_actual': None,
        'forecast_predicted': None,
        'level_guess': None,
        'combined_error_data': None,
        'monthly_trend_data': None
    }
    
    try:
        tf.keras.backend.clear_session()
        
        log_data = request.session.get('uploaded_log_file')
        if not log_data:
            result['error_msg'] = 'No log file found in session. Please upload a log file first.'
            return result
            
        if len(log_data.strip()) == 0:
            result['error_msg'] = 'Log file appears to be empty. Please upload a valid log file.'
            return result
        
        # Parse log data
        lines = [line.strip() for line in log_data.splitlines() if line.strip()]
        if not lines:
            result['error_msg'] = 'No valid log lines found in the uploaded file.'
            return result
            
        parsed = []
        for line in lines:
            parsed_line = parse_log_line(line)
            if parsed_line[0] is not None:
                parsed.append(parsed_line)
        
        if not parsed:
            result['error_msg'] = 'No valid log entries could be parsed. Please check the log format.'
            return result
            
        log_df = pd.DataFrame(parsed, columns=["date", "level", "thread", "message"])
        log_df = log_df.dropna()
        
        if log_df.empty:
            result['error_msg'] = 'No valid log entries found after parsing.'
            return result
            
        log_df["month"] = log_df["date"].dt.to_period("M")
        
        # Define time periods
        monthly_index = pd.period_range("2025-01", "2025-05", freq="M")
        future_months = pd.period_range("2025-06", "2025-12", freq="M")
        full_index = monthly_index.append(future_months)
        
        # Create monthly trend data for the specific message
        if input_msg and input_msg.strip():
            input_matches = log_df[log_df["message"].str.strip().str.lower() == input_msg.strip().lower()]
            
            if not input_matches.empty:
                monthly_trend = input_matches.groupby(input_matches["date"].dt.to_period("M")).size()
                trend_data = []
                for month in monthly_trend.index:
                    trend_data.append({
                        'month': str(month),
                        'count': int(monthly_trend[month])
                    })
                result['monthly_trend_data'] = trend_data
        
        # Individual message forecast
        if input_msg and input_msg.strip():
            input_matches = log_df[log_df["message"].str.strip().str.lower() == input_msg.strip().lower()]
            
            if input_matches.empty:
                result['error_msg'] = f"No exact matching messages found for: '{input_msg}'"
                return result
            
            level_guess = input_matches['level'].mode().iloc[0] if not input_matches.empty else 'UNKNOWN'
            result['level_guess'] = level_guess
            
            msg_monthly = input_matches.groupby("month").size().reindex(monthly_index, fill_value=0)
            
            if msg_monthly.sum() < 4:
                result['error_msg'] = f"Not enough data points for the input message to forecast (found {msg_monthly.sum()}, need at least 4)."
                return result
            
            # LSTM Model for individual message
            try:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_msg = scaler.fit_transform(msg_monthly.values.reshape(-1, 1))
                X_msg, y_msg = prepare_sequences(scaled_msg, steps=3)
                
                if X_msg.shape[0] > 0:
                    model = create_lstm_model()
                    
                    model.fit(
                        X_msg.reshape(X_msg.shape[0], X_msg.shape[1], 1), 
                        y_msg, 
                        epochs=10, 
                        verbose=0,
                        batch_size=1
                    )
                    
                    # Forecasting
                    input_seq = scaled_msg[-3:].flatten()
                    msg_preds = []
                    
                    for _ in range(7):
                        pred = model.predict(input_seq.reshape(1, 3, 1), verbose=0)
                        msg_preds.append(pred[0][0])
                        input_seq = np.append(input_seq[1:], pred[0][0])
                    
                    forecast_vals_msg = scaler.inverse_transform(np.array(msg_preds).reshape(-1, 1))
                    forecast_vals_msg = np.maximum(forecast_vals_msg.flatten(), 0).round().astype(int)
                    
                    msg_forecast_series = pd.Series(forecast_vals_msg, index=future_months)
                    msg_full = pd.concat([msg_monthly, msg_forecast_series])
                    
                    # Prepare data for frontend
                    result['forecast_table'] = [
                        {"Month": month.to_timestamp(), "Count": int(count)} 
                        for month, count in msg_full.items()
                    ]
                    
                    result['forecast_actual'] = {
                        'months': [str(m) for m in monthly_index],
                        'values': [int(x) for x in msg_monthly.values]
                    }
                    
                    result['forecast_predicted'] = {
                        'months': [str(m) for m in future_months],
                        'values': [int(x) for x in forecast_vals_msg]
                    }
                    
                    del model
                    tf.keras.backend.clear_session()
                    
            except Exception as e:
                result['error_msg'] = f"Error in LSTM forecasting: {str(e)}"
                tf.keras.backend.clear_session()
                return result
        
        # Combined ERROR forecast
        error_logs = log_df[log_df["level"] == "ERROR"]
        
        if not error_logs.empty:
            unique_error_messages = error_logs["message"].unique()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            combined_data = []
            color_index = 0
            
            for msg in unique_error_messages:
                try:
                    monthly_counts = error_logs[error_logs["message"] == msg].groupby("month").size().reindex(monthly_index, fill_value=0)
                    
                    if monthly_counts.sum() < 2:
                        continue
                    
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled = scaler.fit_transform(monthly_counts.values.reshape(-1, 1))
                    X, y = prepare_sequences(scaled, steps=3)
                    
                    if X.shape[0] == 0:
                        continue
                    
                    model_id = re.sub(r'[^a-zA-Z0-9]', '_', input_msg[:20])
                    model_path = os.path.join("PhaseI", "models", f"model_{model_id}.h5")
                    scaler_path = model_path.replace(".h5", "_scaler.gz")
                    if os.path.exists(model_path) and os.path.exists(scaler_path):
                        model = load_model(model_path)
                        scaler = joblib.load(scaler_path)
                    else:
                        model = create_lstm_model()
                        model.fit(
                            X.reshape(X.shape[0], X.shape[1], 1), 
                            y, 
                            epochs=10, 
                            verbose=0,
                            batch_size=1
                        )
                    
                    # Forecasting
                    input_seq = scaled[-3:].flatten()
                    preds = []
                    
                    for _ in range(7):
                        pred = model.predict(input_seq.reshape(1, 3, 1), verbose=0)
                        preds.append(pred[0][0])
                        input_seq = np.append(input_seq[1:], pred[0][0])
                    
                    forecast_vals = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
                    forecast_vals = np.maximum(forecast_vals.flatten(), 0).round().astype(int)
                    
                    label = msg[:30] + ("..." if len(msg) > 30 else "")
                    color = colors[color_index % len(colors)]
                    
                    combined_data.append({
                        'label': label,
                        'full_message': msg,
                        'color': color,
                        'actual_values': [int(x) for x in monthly_counts.values],
                        'predicted_values': [int(x) for x in forecast_vals],
                        'actual_months': [str(m) for m in monthly_index],
                        'predicted_months': [str(m) for m in future_months],
                        'all_months': [str(m) for m in full_index],
                        'months': [str(m) for m in full_index], 
                        'values': [int(x) for x in monthly_counts.values] + [int(x) for x in forecast_vals] 
                    })
                    
                    color_index += 1
                    del model
                    tf.keras.backend.clear_session()
                    
                except Exception as e:
                    print(f"Error processing message '{msg}': {e}")
                    tf.keras.backend.clear_session()
                    continue
            
            result['combined_error_data'] = combined_data
            
            if not combined_data:
                result['error_msg'] = 'No ERROR messages had sufficient data for forecasting (minimum 2 occurrences required).'
    
    except Exception as e:
        result['error_msg'] = f"Unexpected error processing log file: {str(e)}"
        print(f"Exception in LSTM_Forecast_from_session: {e}")
        traceback.print_exc()
        tf.keras.backend.clear_session()
    
    return result

@ensure_csrf_cookie
def Result_Forecast(request, user_name):
    """Main view for handling forecast results"""
    forecast_table = None
    error_msg = None
    level_guess = None
    input_msg = None
    forecast_actual = None
    forecast_predicted = None
    combined_error_data = None
    forecast_percent_change = None
    forecast_percent_color = '#fff'
    monthly_trend_data = None
    chart_data = {}
    
    # Additional statistics
    total_historical = 0
    total_predicted = 0
    
    # Check if session has log file
    if 'uploaded_log_file' not in request.session:
        context = {
            'user_name': user_name,
            'error_msg': 'No log file found in session. Please upload a log file first.',
        }
        return render(request, 'result.html', context)
    
    # Process log data for charts
    log_data = request.session.get('uploaded_log_file')
    chart_data = process_log_data_for_charts(log_data)
    
    # Load combined error data even without POST (for initial page load)
    try:
        # Pass an empty string for input_msg to only get combined error data
        initial_forecast_result = LSTM_Forecast_from_session(request, "")
        combined_error_data = initial_forecast_result.get('combined_error_data')
        # Capture any initial errors, but don't display them unless no data is found
        if initial_forecast_result.get('error_msg') and not combined_error_data:
            error_msg = initial_forecast_result.get('error_msg')
    except Exception as e:
        error_msg = f"Error loading combined error data: {str(e)}"

    if request.method == 'POST':
        input_msg = request.POST.get('input_msg', '').strip()
        
        if not input_msg:
            error_msg = 'Please enter a log message to forecast.'
        else:
            # Get forecast results for the specific message
            forecast_result = LSTM_Forecast_from_session(request, input_msg)
            
            forecast_table = forecast_result.get('forecast_table')
            error_msg = forecast_result.get('error_msg')
            level_guess = forecast_result.get('level_guess')
            forecast_actual = forecast_result.get('forecast_actual')
            forecast_predicted = forecast_result.get('forecast_predicted')
            monthly_trend_data = forecast_result.get('monthly_trend_data')
            
            # The combined_error_data is already loaded, but you can update it if the function returns it
            if forecast_result.get('combined_error_data'):
                combined_error_data = forecast_result.get('combined_error_data')

            # Calculate totals
            if forecast_actual and forecast_actual.get('values'):
                total_historical = sum(forecast_actual['values'])
            if forecast_predicted and forecast_predicted.get('values'):
                total_predicted = sum(forecast_predicted['values'])
            
            # Calculate percent change for badge
            if forecast_actual and forecast_predicted and len(forecast_actual['values']) > 0 and len(forecast_predicted['values']) > 0:
                try:
                    actual_last = forecast_actual['values'][-1]
                    predicted_first = forecast_predicted['values'][0]
                    if actual_last != 0:
                        forecast_percent_change = ((predicted_first - actual_last) / abs(actual_last)) * 100
                        forecast_percent_color = '#22c55e' if forecast_percent_change > 0 else "#ebebeb"
                except (IndexError, ZeroDivisionError):
                    forecast_percent_change = None
                    forecast_percent_color = '#fff'

    context = {
        'user_name': user_name,
        'forecast_table': forecast_table,
        'forecast_actual': forecast_actual,
        'forecast_predicted': forecast_predicted,
        'error_msg': error_msg,
        'level_guess': level_guess,
        'input_msg': input_msg,
        'forecast_percent_change': forecast_percent_change,
        'forecast_percent_color': forecast_percent_color,
        'combined_error_data': combined_error_data,
        'total_historical': total_historical,
        'total_predicted': total_predicted,
        'monthly_trend_data': monthly_trend_data,
        'chart_data': chart_data,
    }
    
    return render(request, 'result.html', context)

def get_combined_error_forecast(request):
    """
    An API-style view that computes and returns the combined error forecast data.
    """
    if 'uploaded_log_file' not in request.session:
        return JsonResponse({'error': 'No log file found in session.'}, status=400)

    # We pass an empty string because we only want the combined error part
    forecast_result = LSTM_Forecast_from_session(request, "")

    if forecast_result.get('error_msg') and not forecast_result.get('combined_error_data'):
         return JsonResponse({'error': forecast_result['error_msg']}, status=500)

    return JsonResponse({
        'combined_error_data': forecast_result.get('combined_error_data')
    })


def Prediction_Task(request, user_name, calling_request):
    """Handle file upload for predictions"""
    uploaded_file_name = None
    uploaded_file_type = None
    upload_error = None
    
    if request.method == 'POST' and 'log_file' in request.FILES:
        try:
            log_file = request.FILES['log_file']
            uploaded_file_name = log_file.name
            uploaded_file_type = log_file.content_type
            
            # Validate file size (max 10MB)
            if log_file.size > 10 * 1024 * 1024:
                upload_error = "File size too large. Maximum 10MB allowed."
            else:
                file_content = log_file.read().decode('utf-8', errors='ignore')
                if len(file_content.strip()) == 0:
                    upload_error = "File appears to be empty."
                else:
                    request.session['uploaded_log_file'] = file_content
                    request.session.modified = True
                    
        except Exception as e:
            upload_error = f"Error uploading file: {str(e)}"
    
    context = {
        'user_name': user_name,
        'uploaded_file_name': uploaded_file_name,
        'uploaded_file_type': uploaded_file_type,
        'upload_error': upload_error,
    }
    
    return render(request, 'predictions.html', context)


from django.views.decorators.csrf import csrf_exempt
import joblib

@csrf_exempt
def Train_Model(request, user_name):
    """Train and save LSTM model for all ERROR messages"""
    from django.contrib import messages
    from datetime import datetime

    log_data = request.session.get('uploaded_log_file')
    if not log_data:
        return render(request, 'predictions.html', {
            'user_name': user_name,
            'upload_error': "No log file found in session. Please upload a file first."
        })

    try:
        # Parse log lines
        lines = [line.strip() for line in log_data.splitlines() if line.strip()]
        parsed = [parse_log_line(line) for line in lines if parse_log_line(line)[0] is not None]

        log_df = pd.DataFrame(parsed, columns=["date", "level", "thread", "message"])
        log_df["month"] = log_df["date"].dt.to_period("M")

        error_logs = log_df[log_df["level"] == "ERROR"]
        error_counts = error_logs.groupby(["month", "message"]).size().unstack(fill_value=0)

        # 🔥 Define model save path
        model_dir = os.path.join("PhaseI", "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # ❌ Delete older models
        for file in glob.glob(os.path.join(model_dir, "model_*.h5")):
            os.remove(file)
        for file in glob.glob(os.path.join(model_dir, "*_scaler.gz")):
            os.remove(file)

        # 🎯 Train and Save new models
        for message in error_counts.columns:
            y = error_counts[message].values
            if len(y) < 4:
                continue

            scaler = MinMaxScaler()
            y_scaled = scaler.fit_transform(y.reshape(-1, 1))
            X, y_train = prepare_sequences(y_scaled, steps=3)

            if X.shape[0] == 0:
                continue

            model = create_lstm_model()
            model.fit(X.reshape(X.shape[0], X.shape[1], 1), y_train, epochs=100, batch_size=1, verbose=0)

            model_id = re.sub(r'[^a-zA-Z0-9]', '_', message[:30])
            model_path = os.path.join(model_dir, f"model_{model_id}.h5")
            scaler_path = model_path.replace(".h5", "_scaler.gz")

            model.save(model_path)
            joblib.dump(scaler, scaler_path)
        request.session['last_trained'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        messages.success(request, "Model trained and saved successfully!")
        return redirect('Prediction_Task', user_name=user_name, calling_request='website')

    except Exception as e:
        return render(request, 'predictions.html', {
            'user_name': user_name,
            'upload_error': f"Training failed: {str(e)}"
        })








from django.shortcuts import render, redirect
from .models import Register,User_Register,Forgot_Password
from django.conf import settings
from django.core.mail import send_mail
from datetime import datetime
from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings
import pandas as pd
from io import StringIO
from collections import defaultdict
from django.http import JsonResponse
import re,io,csv,os,random
from collections import defaultdict
from collections import defaultdict
from datetime import datetime, timedelta
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from keras.losses import MeanSquaredError

Log_File_Path=settings.KEYWORD_FILE_PATH
keywords={}
with open(Log_File_Path, 'a+', newline='', encoding='utf-8') as csv_file:
            csv_file.seek(0)
            data = csv_file.read()
            data=StringIO(data)
            reader=csv.DictReader(data)
            for i in reader:
                
                keywords[i['keyword']]=i['keyword_name']
print(keywords)

# keywords = {
#                  'EPU_Ambient_Temperature': 'EPU Ambient Temperature',
#                 'batterycharge': 'Battery charge',
#                 'batterytemperature':'Battery Temperature',
#                 'ups_temperature':'c'
#                 # Add more keywords as needed
#         }



def Register(request):
    if request.method == 'POST':
        User_Name=request.POST['user_name']
        Email=request.POST['email']
        Password=request.POST['password']
        ###print(User_Name,Email,Password)
        Register_Data=User_Register(Name=User_Name,Email=Email,Password=Password)
        Register_Data.save()
        return render(request,'login.html')
    return render(request,'createAccount.html')

def User_Login(request):
    request.session['csv_data']=""
    if request.method=='POST':
        User_Email=request.POST['email']
        Password=request.POST['password']
        request.session['user_email']=User_Email
        ###print(User_Email,Password)
        User_Details=User_Register.objects.filter(Email=User_Email,Password=Password).values()
        if User_Details:
            return redirect(reverse('Dashboard',kwargs={'user_name':User_Details[0]['Name']}))
        else:
            return render(request,'login.html',{"message":"credentials are wrong! "})
    # return render(request,'Login_Page.html')
    return render(request,'login.html')


# In your views.py file

def Dashboard(request, user_name):
    """
    Displays a dashboard analysis of the log file uploaded by the user.
    The log data is retrieved from the session.
    """
    # Get log data from session, which was uploaded from the prediction page
    log_data = request.session.get('uploaded_log_file')

    # If there's no log data in the session, render the page with a helpful message
    if not log_data:
        return render(request, 'landing.html', {
            "user_name": user_name,
            "message": "No log file has been uploaded. Please go to the 'Prediction' page to upload a file first."
        })

    # Initialize data structures, similar to the original function
    substring_keywords_info = defaultdict(lambda: {})
    substring_devided_data = defaultdict(lambda: {'INFO': [], 'WARNING': [], 'ERROR': []})
    words_last_reading = {}
    Cretical_Data = defaultdict(list)
    sub_string_data = defaultdict(list)

    try:
        # The uploaded data is a raw string. We parse it line by line.
        lines = [line.strip() for line in log_data.splitlines() if line.strip()]
        
        for line in lines:
            dt, level, thread, message = parse_log_line(line)

            # Skip any lines that could not be parsed successfully
            if not all((dt, level, thread, message)):
                continue

            # Use the 'thread' from the log as the main grouping category, analogous to 'Search_string'
            string_value = thread
            event_template = message

            # Populate the data structures for the template
            sub_string_data[string_value].append(event_template)
            
            # Use the parsed level for more reliable categorization
            if level in ['INFO', 'WARNING', 'ERROR']:
                substring_devided_data[string_value][level].append(event_template)

            if level == 'ERROR':
                Cretical_Data[string_value].append(event_template)

            # This part searches for specific keywords within the log message
            # It relies on the global 'keywords' dictionary defined in your views
            for keyword, search_string in keywords.items():
                pattern = re.compile(rf'{search_string} \(.*?\): (\d+)')
                match = pattern.search(event_template)
                
                if match:
                    value = int(match.group(1))
                    words_last_reading[keyword] = value
                    substring_keywords_info[string_value][keyword] = value

        # Pass the processed data to the template
        return render(request, 'landing.html', {
            "user_name": user_name,
            "substring_devided_data": dict(substring_devided_data),
            "data": dict(sub_string_data),
            "alert_value": words_last_reading,
            "Cretical_Data": dict(Cretical_Data)
        })

    except Exception as e:
        # In case of an error during processing, display a message
        print(f"Error processing uploaded log data in Dashboard: {e}")
        return render(request, 'landing.html', {
            "user_name": user_name,
            "message": f"An error occurred while processing the uploaded log file: {e}"
        })

#Not using this function this function is used for fetching csv file and processing
def Get_Strings_Logs(request,user_name):
    data={}
    list_values=dict()
    keywords = {
            'batterytemperature':'Battery Temperature',
        }
    if request.method == 'POST':
            csv_file=request.FILES['csv_file']
            csv_data = csv_file.read().decode('utf-8')
    
    # Store the CSV data in the session
            request.session['csv_data'] = csv_data
            csv_file.seek(0)
            csv_data=csv_file.read().decode('utf-8')
            csv_file = StringIO(csv_data)
            reader = csv.DictReader(csv_file)
            
            for row in reader:
                string_value = row['Search_string']
                event_template = row['Event_Templete']
                if string_value not in data:
                    data[string_value] = []
                    
                data[string_value].append(event_template)
                for keyword, search_string in keywords.items():
                    # Search for the keyword pattern
                    pattern = re.compile(rf'{search_string} \(.*?\): (\d+)')
                    match = pattern.search(event_template)
                        
                    if match:
                        value = int(match.group(1))
                        list_values[keyword]=value
           
            output=dict(data)
            ###print("the output is",output)
            return render(request,'landing.html',{"user_name":user_name,"data":output,"alert_value":list_values})     
    return render(request,'landing.html',{"user_name":user_name})


def for_checking(request):
    csv_file_path=settings.CSV_FILE_PATH
    values=defaultdict(list)
    Time_values=defaultdict(list)
    list_values=dict()
    log_values = defaultdict(list)

    sub_string_data = {}
    substring_devided_count = defaultdict(lambda: {'INFO': 0, 'WARNING': 0, 'ERROR': 0}) 
   
    # substring_devided_data = defaultdict(lambda: {'INFO': [], 'WARNING': [], 'ERROR': []}) 
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
                csv_data = file.read()
            # Convert CSV string to a file-like object
        csv_file_like = StringIO(csv_data) 
            # Initialize a CSV DictReader to read the data
        reader = csv.DictReader(csv_file_like)
        # Define your keywords and their default words
        
            # Retrieve CSV data from session
        calling_request="website"
        reader=list(reader)
        for row in range(len(reader)):
            event_template=reader[row]['Event_Template']
            string_value = reader[row]['Search_string']
            Date_Time_values=reader[row]['Datetime']
            ###print(event_template)
            for keyword, search_string in keywords.items():
                # Search for the keyword pattern
                # #print(search_string)
                pattern = re.compile(rf'{search_string} \(.*?\): (\d+)')
                match = pattern.search(event_template)
                if calling_request=="website":      
                    if match:
                        # #print("calling this")
                        value = int(match.group(1))
                        # if len(values[keyword])<=5:
                        values[keyword].append(value)
                        list_values[keyword]=value
                        Time_values[keyword].append(Date_Time_values)
                       
                        start_index = max(0, row - 4)
                        end_index = min(len(reader), row + 5)
                        context_lines = [reader[i]['Event_Template'] for i in range(start_index, end_index)]

                        log_values[keyword].append({value: context_lines})
            # string_value = row['Search_string']
            # event_template = row['Event_Template']
            if string_value not in sub_string_data:
                sub_string_data[string_value] = []
                        
            sub_string_data[string_value].append(event_template)
                    # Determine log level from event_template
            if 'INFO' in event_template:
                log_level = 'INFO'
            elif 'WARNING' in event_template:
                log_level = 'WARNING'
            elif 'ERROR' in event_template:
                log_level = 'ERROR'
            else:
                continue  # Skip if no recognized log level is found
            
                    # Update counts
            substring_devided_count[string_value][log_level] += 1
        print(substring_devided_count)
        # print("this are the values",log_values["Battery charge"][0])
        # print("this are the values",log_values["Battery charge"][1])
        # print("this are the values",log_values["Battery charge"][2])
        # print("this are the imte values",Time_values)
    except Exception as e:
        print("error coming",e)
    return HttpResponse("okay")

def Visualization_Task(request,user_name,calling_request):
    # print("calling visualizaion function",calling_request)
    csv_file_path=settings.CSV_FILE_PATH
    values=defaultdict(list)
    Time_values=defaultdict(list)
    list_values=dict()
    sub_string_data = {}
    log_values=defaultdict(list)
    substring_devided_count = defaultdict(lambda: {'INFO': 0, 'WARNING': 0, 'ERROR': 0}) 
    today = datetime.now().date()
    start_date=end_date=today
   
    # substring_devided_data = defaultdict(lambda: {'INFO': [], 'WARNING': [], 'ERROR': []}) 
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
                csv_data = file.read()
            # Convert CSV string to a file-like object
        csv_file_like = StringIO(csv_data) 
            # Initialize a CSV DictReader to read the data
        reader = csv.DictReader(csv_file_like)
        # Define your keywords and their default words
        
            # Retrieve CSV data from session
            
        
        reader=list(reader)
        for row in range(len(reader)):
            event_template=reader[row]['Event_Template']
            string_value = reader[row]['Search_string']
            Date_Time_values=reader[row]['DateTime']
            log_datetime = datetime.strptime(Date_Time_values, '%Y-%m-%d %H:%M:%S').date()
            ###print(event_template)
            if start_date <= log_datetime <= end_date:
                for keyword, search_string in keywords.items():
                    # Search for the keyword pattern
                    # #print(search_string)
                    pattern = re.compile(rf'{search_string} \(.*?\): (\d+)')
                    match = pattern.search(event_template)
                        
                    if match:
                            # #print("calling this")
                            value = int(match.group(1))
                            # if len(values[keyword])<=5:
                            values[keyword].append(value)
                            list_values[keyword]=value
                            Time_values[keyword].append(Date_Time_values)
                        
                            start_index = max(0, row - 4)
                            end_index = min(len(reader), row + 5)
                            context_lines = [reader[i]['Event_Template'] for i in range(start_index, end_index)]

                            log_values[keyword].append({value: context_lines})
                # string_value = row['Search_string']
                # event_template = row['Event_Template']
                if string_value not in sub_string_data:
                    sub_string_data[string_value] = []
                            
                sub_string_data[string_value].append(event_template)
                        # Determine log level from event_template
                if 'INFO' in event_template:
                    log_level = 'INFO'
                elif 'WARNING' in event_template:
                    log_level = 'WARNING'
                elif 'ERROR' in event_template:
                    log_level = 'ERROR'
                else:
                    continue  # Skip if no recognized log level is found
                
                        # Update counts
                substring_devided_count[string_value][log_level] += 1
            
            # if request.method=='POST':
            #     #print("calling inside")
            #     Search_data=dict()
            #     devided_data=dict()
            #     Search_String=request.POST.get('search_string')
            #     print("search value",Search_String)
            #         # Search_String="TrdRobot"
            #     Search_data[Search_String]=[]
            #     devided_data[Search_String]={'INFO':[],'WARNING':[],'ERROR':[]}
            #     with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
            #         csv_data = file.read()
                
            #     # Convert CSV string to a file-like object
            #     csv_file_like = StringIO(csv_data)
                
            #     # Initialize a CSV DictReader to read the data
            #     reader = csv.DictReader(csv_file_like)
                
            #     for row in reader:
            #         #####print(row)
            #         event_template=row['Event_Template']
            #         if Search_String.lower()==row['Search_string'].lower():
            #             #####print("execute")
            #             Search_data[Search_String].append(row['Event_Template'])
            #             #####print("uo to this",Search_data)
            #             ######print(dict("searching data1",Search_data))
            #             if 'INFO' in event_template:
            #                 log_level = 'INFO'
            #             elif 'WARNING' in event_template:
            #                 log_level = 'WARNING'
            #             elif 'ERROR' in event_template:
            #                 log_level = 'ERROR'
            #             else:
            #                 continue
            #             ####print("this is not ok",log_level)
            #             devided_data[Search_String][log_level].append(event_template)  
            #     devided_data=dict(devided_data)
            #     print("this is serach data",devided_data)
            #     return render(request,'result.html',{"user_name":user_name,"substring_devided_data":devided_data,"Count_data":dict(substring_devided_count),
        #                                          "default_words":dict(values),"last_values":list_values,"Search_String":dict(Search_data),"Time_Values":Time_values,"log_data":log_values})
    except Exception as e:
        ######print("Error is comint at visulization task function",e)
        return render(request,'result.html',{"user_name":user_name,"Count_data":"","default_words":"","last_values":"","Time_Values":"","log_data":""})
    return render(request,'result.html',{"user_name":user_name,"Count_data":dict(substring_devided_count),"default_words":dict(values),"last_values":list_values,"Time_Values":Time_values,"log_data":log_values})

def Current_Meter_values():
    #print("calling meter values this function")
    list_values=dict()
    ##print(keywords)
    substring_keywords_info = defaultdict(lambda: {}) 
    csv_file_path=settings.CSV_FILE_PATH
        
    # if not request.session['csv_data']:
    #     return render(request,'result.html',{"message":"Please upload csv file in AddData.",'user_name':user_name,})
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
            csv_data = file.read()
            
            # Convert CSV string to a file-like object
        csv_file_like = StringIO(csv_data)
            
            # Initialize a CSV DictReader to read the data
        reader = csv.DictReader(csv_file_like)
        
        for row in reader:
            string_value=row['Search_string']
            event_template = row['Event_Template']
            ##print(event_template)
            for keyword, search_string in keywords.items():
                    # Search for the keyword pattern
                    pattern = re.compile(rf'{search_string} \(.*?\): (\d+)')
                    match = pattern.search(event_template)
                        
                    if match:
                        value = int(match.group(1))
                        list_values[keyword]=value
                        substring_keywords_info[string_value][keyword]=value
        ###print("list values are",list_values)
        substring_keywords_info=dict(substring_keywords_info)
        #print(substring_keywords_info)
        data = {"response": substring_keywords_info}
        return list_values
        
    except Exception as e:
        #print("Error is coming at Healthy funtion",e)
        data = {"response": "Error is coming"}
        return list_values

def Visualization_Loop(request):
    changed_date=request.GET.get('param1')
    print("changed date value is",changed_date)
    time_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')
    csv_file_path = settings.CSV_FILE_PATH
    values=defaultdict(list)
    list_values=dict()
    Date_values=defaultdict(list)
    Time_values=defaultdict(list)
    log_values = defaultdict(list)
    
    today = datetime.now().date()
    start_date=end_date=today
    sub_string_data = {}
    substring_devided_count = defaultdict(lambda: {'INFO': 0, 'WARNING': 0, 'ERROR': 0}) 
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
            csv_data = file.read()
        
        csv_file_like = StringIO(csv_data)
        reader = csv.DictReader(csv_file_like)

        reader=list(reader)
        if changed_date == "Last_Day":
                print("last day executed")
                end_date = today - timedelta(days=1)
                start_date = end_date  # Yesterday

        elif changed_date == "Today":
                print("Today executed")
                start_date = today
                end_date = today  # Today

        elif changed_date == "Last_Week":
                print("Last_Week executed")
                end_date = today
                start_date = today - timedelta(days=7)

        elif changed_date == "Last_Month":
                print("Last_Month executed")
                end_date = today
                start_date = today - timedelta(days=30)
        for row in range(len(reader)):
            
            event_template=reader[row]['Event_Template']
            string_value = reader[row]['Search_string']
            Date_Time_values=reader[row]['DateTime']
            log_datetime = datetime.strptime(Date_Time_values, '%Y-%m-%d %H:%M:%S').date()
            if start_date <= log_datetime <= end_date:
                for keyword, search_string in keywords.items():
                    # Search for the keyword pattern
                    # #print(search_string)
                    pattern = re.compile(rf'{search_string} \(.*?\): (\d+)')
                    match = pattern.search(event_template)      
                    if match:
                        Only_Time=time_pattern.search(Date_Time_values)
                        value = int(match.group(1))
                        values[keyword].append(value)
                        list_values[keyword]=value   
                        Time_values[keyword].append(Only_Time.group(0)) 
                        Date_values[keyword].append(Date_Time_values) 
                        start_index = max(0, row - 4)
                        end_index = min(len(reader), row + 5)
                        context_lines = [reader[i]['Event_Template'] for i in range(start_index, end_index)]

                        log_values[keyword].append({value: context_lines})   
                # log_values=""
                if string_value not in sub_string_data:
                    sub_string_data[string_value] = []
                            
                sub_string_data[string_value].append(event_template)
                        # Determine log level from event_template
                if 'INFO' in event_template:
                    log_level = 'INFO'
                elif 'WARNING' in event_template:
                    log_level = 'WARNING'
                elif 'ERROR' in event_template:
                    log_level = 'ERROR'
                else:
                    continue  # Skip if no recognized log level is found
                
                        # Update counts
                substring_devided_count[string_value][log_level] += 1
            
            data=Current_Meter_values()
            
            data={
            "Count_data":dict(substring_devided_count),"default_words":dict(values),"last_values":data,
            "Time_Values":Date_values,"log_data":log_values,"x_scale_values":Time_values
            }
        return JsonResponse(data)
        
    except Exception as e:
        ######print("Error is comint at visulization task function",e)
        data={
        "Count_data":"","default_words":"","last_values":"","Time_Values":"","log_data":""
        }
        return JsonResponse(data)



    

# Accourding to pavan requirement I am not using this function. I am separating data and sending information directly
def Soring_Word_Wise(request):
    request_data=[]
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        param1 = request.GET.get('param1', None)
        param2 = request.GET.get('param2', None)
        print(param1,param2)
        csv_file_path=settings.CSV_FILE_PATH
        
    # if not request.session['csv_data']:
    #     return render(request,'result.html',{"message":"Please upload csv file in AddData.",'user_name':user_name,})
        try:
            with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
                csv_data = file.read()
            
            # Convert CSV string to a file-like object
            csv_file_like = StringIO(csv_data)
            
            # Initialize a CSV DictReader to read the data
            reader = csv.DictReader(csv_file_like)
            if param2=="All_Data":
                for row in reader:
                    print(row)
                    sub_string=row['Search_string']
                    event_template=row['Event_Template']
                ###print("event template is",event_template)
                    if sub_string==param1:
                        request_data.append(event_template)
            else:
                for row in reader:
                    sub_string=row['Search_string']
                    event_template=row['Event_Template']
                    ###print("event template is",event_template)
                    if sub_string==param1 and param2 in event_template:
                        request_data.append(event_template)
            ###print("requested data is",request_data)
            response_data = {
                "Requested_Data":request_data
            }
            
            return JsonResponse(response_data)
        except Exception as e:
            ###print("Error is coming at soring_word_wise function",e)
            response_data = {
                "Requested_Data":"Error is coming"
            }
            
            return JsonResponse(response_data)

    return JsonResponse({'error': 'Invalid request'}, status=400)
           


def Sorting_Data(request, user_name):
    ###print("calling sorting data function")
    values = defaultdict(list)
    today = datetime.now().date()
    Time_values=defaultdict(list)
    list_values = {}
    sub_string_data = {}
    substring_devided_count = defaultdict(lambda: {'INFO': 0, 'WARNING': 0, 'ERROR': 0})
    csv_file_path = settings.CSV_FILE_PATH
    if request.method == 'POST':
        param1 = request.POST.get('option_name')

        try:
            with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
                csv_data = file.read()

            csv_file_like = StringIO(csv_data)
            reader = csv.DictReader(csv_file_like)

            print("this is value", param1)
            if param1 == "Last_Day":
                end_date = today - timedelta(days=1)
                start_date = end_date  # Yesterday

            elif param1 == "Today":
                start_date = today
                end_date = today  # Today

            elif param1 == "Last_Week":
                end_date = today
                start_date = today - timedelta(days=7)

            elif param1 == "Last_Month":
                end_date = today
                start_date = today - timedelta(days=30)

            # elif param1 == "All_Data":
            #     return redirect(reverse('Visualization_Task', kwargs={'user_name': user_name,"calling_request":""}))

            print("last day", end_date, start_date)
            for row in reader:
                date_value = row['DateTime']
                log_datetime = datetime.strptime(date_value, '%Y-%m-%d %H:%M:%S').date()
              
                if start_date <= log_datetime <= end_date:
                    ###print("Processing date:", log_datetime)
                    event_template = row['Event_Template']
                    Date_Time_values=row['DateTime']
                    ###print("Event template:", event_template)
                    
                    for keyword, search_string in keywords.items():
                        ###print("Searching for:", search_string)
                        pattern = re.compile(rf'{search_string} \(.*?\): (\d+)')
                        match = pattern.search(event_template)

                        if match:
                            value = int(match.group(1))
                            values[keyword].append(value)
                            list_values[keyword] = value
                            Time_values[keyword].append(Date_Time_values)
                            

                    
                    string_value = row['Search_string']

                    if string_value not in sub_string_data:
                        sub_string_data[string_value] = []
                    sub_string_data[string_value].append(event_template)

                    # Determine log level
                    if 'INFO' in event_template:
                        log_level = 'INFO'
                    elif 'WARNING' in event_template:
                        log_level = 'WARNING'
                    elif 'ERROR' in event_template:
                        log_level = 'ERROR'
                    else:
                        ###print("Unrecognized log level in event_template")
                        continue  # Skip if no recognized log level is found

                   
                    
                    # Update counts
                    substring_devided_count[string_value][log_level] += 1
                    

            ###print("The values are:", dict(values))
            ###print("Last values are:", list_values)

          
            return render(request, 'result.html', {
                "user_name": user_name,
                "Count_data": dict(substring_devided_count),
                "default_words": dict(values),
                "last_values": dict(list_values),
                "option_value": param1,
                "Time_Values":Time_values,
                
            })

        except Exception as e:
            print(f"Error in Sorting_Data function: {e}")
            return render(request, 'result.html', {
                "user_name": user_name,
                "Count_data": "",
                "default_words": "",
                "last_values": "",
                "option_value": param1,
                "error_message": str(e),
                "Time_Values":"" 
            })


def Show_Info(request, user_name):
    """Filters and returns INFO level logs from the session."""
    try:
        print("✅ Checking session for 'uploaded_log_file'")
        print("Session Keys:", request.session.keys())

        info_logs = get_logs_from_session(request, 'INFO')
        print("✅ Info logs fetched successfully.")
        return JsonResponse({'response': info_logs})
    except Exception as e:
        print("❌ Exception in Show_Info view:", e)
        return JsonResponse({'response': [f"An error occurred: {e}"], 'error': True})




def Warning_Loop(request):
    today = datetime.now().date()
    start_date = today
    end_date = today
    csv_file_path=settings.CSV_FILE_PATH
    Warning_Data=[]
    try:
        with open(csv_file_path,'r',newline='',encoding='utf-8') as file:
            csv_data=file.read()
        csv_flie_like=StringIO(csv_data)
        reader=csv.DictReader(csv_flie_like)
        for row in reader:
            # print(row)
            event_template=row['Event_Template']
            date_value = row['DateTime']
            log_datetime = datetime.strptime(date_value, '%Y-%m-%d %H:%M:%S').date()
            if start_date <= log_datetime <= end_date:
                if "WARNING" in event_template:
                    Warning_Data.append(event_template)
        # print("this is warning data",Warning_Data)
        data={
            "response":Warning_Data
        }
        return JsonResponse(data)
    except Exception as e:
        #print("error coming at show warning funciton",e)
        data={
            "response":""
            }
        return JsonResponse(data)
def Error_Loop(request):
    today = datetime.now().date()
    start_date = today
    end_date = today
    csv_file_path=settings.CSV_FILE_PATH
    Error_Data=[]
    try:
        with open(csv_file_path,'r',newline='',encoding='utf-8') as file:
            csv_data=file.read()
        csv_file_like=StringIO(csv_data)
        reader=csv.DictReader(csv_file_like)
        for row in reader:
            event_template=row['Event_Template']
            date_value = row['DateTime']
            log_datetime = datetime.strptime(date_value, '%Y-%m-%d %H:%M:%S').date()
            if start_date <= log_datetime <= end_date:
                if "ERROR" in event_template:
                    Error_Data.append(event_template)
        data={
            "response":Error_Data
        }
        return JsonResponse(data)
    except Exception as e:
        #print('exception occuring in error loop',e)
        data={
            "response":""
        }
        return JsonResponse(data)
    

def Show_Info(request, user_name):
        
        """Filters and returns WARN/WARNING level logs from the session."""
        try:
            info_logs = get_logs_from_session(request, 'INFO')
            return JsonResponse({'response': info_logs})
        except Exception as e:
            return JsonResponse({'response': [f"An error occurred: {e}"], 'error': True})
    # """Filters and returns INFO level logs from the session."""
    # try:
    #     print("✅ Checking session for 'uploaded_log_file'")
    #     print("Session Keys:", request.session.keys())

    #     info_logs = get_logs_from_session(request, 'INFO')
    #     print("✅ Info logs fetched successfully.")
    #     return JsonResponse({'response': info_logs})
    # except Exception as e:
    #     print("❌ Exception in Show_Info view:", e)
    #     return JsonResponse({'response': [f"An error occurred: {e}"], 'error': True})





def Show_Warning(request, user_name):
    """Filters and returns WARN/WARNING level logs from the session."""
    try:
        warn_logs = get_logs_from_session(request, 'WARN')
        return JsonResponse({'response': warn_logs})
    except Exception as e:
        return JsonResponse({'response': [f"An error occurred: {e}"], 'error': True})

def Show_Errors(request, user_name):
    """Filters and returns ERROR level logs from the session."""
    try:
        error_logs = get_logs_from_session(request, 'ERROR')
        return JsonResponse({'response': error_logs})
    except Exception as e:
        return JsonResponse({'response': [f"An error occurred: {e}"], 'error': True})


def Add_New_Keyword(request,user_name):
    
    global keywords
    Log_File_Path=settings.KEYWORD_FILE_PATH
    ##print(Log_File_Path)
    if request.method=="POST":
        keyword_name=request.POST.get('keyword_name')
        action=request.POST.get('action')
    try:
        file_exists = os.path.isfile(Log_File_Path)
        with open(Log_File_Path, 'a+', newline='', encoding='utf-8') as csv_file:
            csv_file.seek(0)
            data = csv_file.read()
            csv_writer = csv.writer(csv_file)

            # Write the header only if the file is empty or doesn't exist
            if action=="Add":
                if not file_exists or os.stat(Log_File_Path).st_size == 0:
                    csv_writer.writerow(['keyword','keyword_name'])
                if keyword_name not in data:
                    csv_writer.writerow([keyword_name,keyword_name])
                else:
                    print("already present")
            elif action=="Delete":
                updated_data = []
                with open(Log_File_Path, mode='r', newline='', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    header = next(csv_reader, None)  # Read the header

                     # Ensure we have the correct header
                    if header == ['keyword', 'keyword_name']:
                        updated_data.append(header)  # Keep the header

                        for row in csv_reader:
                            if row[0] != keyword_name:  # Check if the keyword is not the one to delete
                                updated_data.append(row)

                    # Write the updated data back to the CSV file
                with open(Log_File_Path, mode='w+', newline='', encoding='utf-8') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerows(updated_data)
                    keywords.pop(keyword_name)
    except Exception as e:
        print("error at add_new_data function",e)      
    with open(Log_File_Path, 'a+', newline='', encoding='utf-8') as csv_file:
            #print("I am calling for both")
            csv_file.seek(0)
            data = csv_file.read()
            data=StringIO(data)
            reader=csv.DictReader(data)
            for i in reader:
                keywords[i['keyword']]=i['keyword_name']
        ##print("keywords are",keywords)
            
    #print(keywords)
    return redirect('Healthy',user_name=user_name)
def Healthy(request,user_name):
    list_values=dict()
    ##print(keywords)
   
    csv_file_path=settings.CSV_FILE_PATH
    substring_keywords_info = defaultdict(lambda: {}) 
    # if not request.session['csv_data']:
    #     return render(request,'result.html',{"message":"Please upload csv file in AddData.",'user_name':user_name,})
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
            csv_data = file.read()
            
            # Convert CSV string to a file-like object
        csv_file_like = StringIO(csv_data)
            
            # Initialize a CSV DictReader to read the data
        reader = csv.DictReader(csv_file_like)
        reader=list(reader)
        for row in reversed(reader):
            string_value=row['Search_string']
            event_template = row['Event_Template']
            ##print(event_template)
            for keyword, search_string in keywords.items():
                    # Search for the keyword pattern
                    pattern = re.compile(rf'{search_string} \(.*?\): (\d+)')
                    match = pattern.search(event_template)
                        
                    if match:
                        value = int(match.group(1))
                        list_values[keyword]=value
                        substring_keywords_info[string_value][keyword]=value
        substring_keywords_info=dict(substring_keywords_info)    
        ###print("list values are",list_values)
        return render(request,'Healthy.html',{"Keyword_Values":substring_keywords_info,"user_name":user_name})
    except Exception as e:
        ###print("Error is coming at Healthy funtion")
        return render(request,'Healthy.html',{"Keyword_Values":"","user_name":user_name})


def Meter_values(request):
    #print("calling meter values this function")
    list_values=dict()
    ##print(keywords)
    substring_keywords_info = defaultdict(lambda: {}) 
    csv_file_path=settings.CSV_FILE_PATH
        
    # if not request.session['csv_data']:
    #     return render(request,'result.html',{"message":"Please upload csv file in AddData.",'user_name':user_name,})
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
            csv_data = file.read()
            
            # Convert CSV string to a file-like object
        csv_file_like = StringIO(csv_data)
            
            # Initialize a CSV DictReader to read the data
        reader = csv.DictReader(csv_file_like)
        
        for row in reader:
            string_value=row['Search_string']
            event_template = row['Event_Template']
            ##print(event_template)
            for keyword, search_string in keywords.items():
                    # Search for the keyword pattern
                    pattern = re.compile(rf'{search_string} \(.*?\): (\d+)')
                    match = pattern.search(event_template)
                        
                    if match:
                        value = int(match.group(1))
                        list_values[keyword]=value
                        substring_keywords_info[string_value][keyword]=value
        ###print("list values are",list_values)
        substring_keywords_info=dict(substring_keywords_info)
        #print(substring_keywords_info)
        data = {"response": substring_keywords_info}
        return JsonResponse(data)
        
    except Exception as e:
        #print("Error is coming at Healthy funtion",e)
        data = {"response": "Error is coming"}
        return JsonResponse(data)
        
    
def User_Forgot_Password(request):
    request.session['user_email']=""
    Date_Time_Values=datetime.now()
    formatted_datetime = Date_Time_Values.strftime('%Y-%m-%d %H:%M:%S.%f')
    if request.method=='POST':
        email=request.POST['email'] 
        request.session['user_email']=email
        ###print("email is",email)
        User_Details=User_Register.objects.filter(Email=email).values()
        
        if User_Details:
            OtpNumber=random.randint(100000,999999)
            subject = 'Email with Template'
            from_email = settings.EMAIL_HOST_USER
            recipient_list = [User_Details[0]['Email'],]
            message = "Your request to change your password has been accepted.\nYour 6-digit OTP is: " + str(OtpNumber)
            context = {'name': 'John Doe'}
            send_mail(subject, message, from_email, recipient_list, )
            try:
                Forgot_Password_Details=Forgot_Password.objects.get(Email=email)
                if Forgot_Password_Details:
                    Forgot_Password_Details.OTP=str(OtpNumber)
                    ###print("upto this")
                    Forgot_Password_Details.save()
            except Exception as e:
                Otp_Details=Forgot_Password(Name=User_Details[0]['Name'],Email=User_Details[0]['Email'],DateTime=formatted_datetime,OTP=str(OtpNumber))
                Otp_Details.save()
                ###print("getting error in userforgot_password",{e})
            return render(request,'forgotPassword.html',{"message":"OTP sent to your mail","username":User_Details[0]['Name']})
            pass
        else:
            return render(request,'forgotPassword.html',{"message":"Not a valid email, Please check!"})
    return render(request,'forgotPassword.html')

def User_Verfiy_Otp(request):
    ###print("calling user verify otp")
    if request.method == 'POST':
        otp=request.POST['otp']
        new_password=request.POST.get('new_password').strip()
        confirm_password=request.POST.get('confirm_password').strip()
        ###print(new_password,confirm_password,len(confirm_password),len(confirm_password))
        if new_password != confirm_password:
            return render(request,'forgotPassword.html',{"message":"new_password and confirm password should be same"})
        ###print(otp,new_password,confirm_password)
        User_Change_Password=User_Register.objects.filter(Email=request.session['user_email'])
        Otp_Verify_Details=Forgot_Password.objects.filter(Email=request.session['user_email'],OTP=otp).values()
        ###print(Otp_Verify_Details)
        if Otp_Verify_Details and User_Change_Password:
            User_Register.objects.filter(Email=request.session['user_email']).update(Password=new_password)
            Forgot_Password.objects.filter(Email=request.session['user_email']).update(OTP="")
            return render(request,'forgotPassword.html',{"message":"Password updated successfully"})
        else:
            return render(request,'forgotPassword.html',{"message":"Wrong OTP!"})

    return render(request,'forgotPassword.html')
def User_Logout(request):
    if request.method=='POST':
        return render(request,'login.html')
def Resend_Otp(request):
    if not request.session['user_email']:
        return render(request,'forgotPassword.html',{"message":"Please enter email."})
    Date_Time_Values=datetime.now()
    formatted_datetime = Date_Time_Values.strftime('%Y-%m-%d %H:%M:%S.%f')
    email=request.session['user_email']
    ###print("email is",email)
    User_Details=User_Register.objects.filter(Email=email).values()
        
    if User_Details:
        OtpNumber=random.randint(100000,999999)
        subject = 'Email with Template'
        from_email = settings.EMAIL_HOST_USER
        recipient_list = [User_Details[0]['Email'],]
        message=" your request to change password is accepted \n Your 6 digit otp is"+str(OtpNumber)
        context = {'name': 'John Doe'}
        send_mail(subject, message, from_email, recipient_list, )
        try:
            Forgot_Password_Details=Forgot_Password.objects.get(Email=email)
            if Forgot_Password_Details:
                Forgot_Password_Details.OTP=str(OtpNumber)
                ###print("upto this")
                Forgot_Password_Details.save()
        except Exception as e:
            Otp_Details=Forgot_Password(Name=User_Details[0]['Name'],Email=User_Details[0]['Email'],DateTime=formatted_datetime,OTP=str(OtpNumber))
            Otp_Details.save()
            ###print("getting error in userforgot_password",{e})
        return render(request,'forgotPassword.html',{"message":"OTP sent to your mail","username":User_Details[0]['Name']})

def All_Data(request, user_name):
    """Returns all logs from the session."""
    try:
        all_logs = get_logs_from_session(request, None) # Pass None to get all logs
        return JsonResponse({'response': all_logs})
    except Exception as e:
        return JsonResponse({'response': [f"An error occurred: {e}"], 'error': True})

def All_Data_Loop(request):
    today = datetime.now().date()
    start_date = today
    end_date = today
    csv_file_path=settings.CSV_FILE_PATH
    today_data=[]
    try:
        with open(csv_file_path,'r',newline='',encoding='utf-8') as file:
            csv_file=file.read()
        csv_file_like=StringIO(csv_file)
        reader=csv.DictReader(csv_file_like)
        # all_data=list(reader)
        for row in reader:
                date_value = row['DateTime']
                log_datetime = datetime.strptime(date_value, '%Y-%m-%d %H:%M:%S').date()
                if start_date <= log_datetime <= end_date:
                    #print(row['Event_Template'])
                    today_data.append(row['Event_Template'])

        data={
            'response':today_data
        }
        return JsonResponse(data)
    except Exception as e:
        #print("exception is occuring in all_data",e)
        data={
            'response':""
        }
        return JsonResponse(data)



def Prediction_Task(request,user_name, calling_request):

    # New logic: Only handle file upload and show file name/type here
    uploaded_file_name = None
    uploaded_file_type = None
    if request.method == 'POST' and 'log_file' in request.FILES:
        log_file = request.FILES['log_file']
        uploaded_file_name = log_file.name
        uploaded_file_type = log_file.content_type
        # Save file to session or temp location for later use in result.html
        request.session['uploaded_log_file'] = log_file.read().decode('utf-8')

    context = {
        'user_name': user_name,
        'uploaded_file_name': uploaded_file_name,
        'uploaded_file_type': uploaded_file_type,
    }
    return render(request, 'predictions.html', context)


