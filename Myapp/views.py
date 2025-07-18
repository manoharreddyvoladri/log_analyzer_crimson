from django.views.decorators.csrf import ensure_csrf_cookie
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie

import pandas as pd
import numpy as np
import re
import os
import glob
import joblib
import warnings
import traceback
import itertools
import matplotlib.colors as mcolors
from datetime import datetime

# ML/DL Imports
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Suppress TensorFlow/Keras warnings for cleaner output
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

# --- Configuration & Global Helpers ---

# Define the directory to save models and scalers
MODEL_DIR = os.path.join("PhaseI", "models")
os.makedirs(MODEL_DIR, exist_ok=True) # Ensure directory exists

# Define paths for the new combined model and scaler
COMBINED_MODEL_PATH = os.path.join(MODEL_DIR, "combined_log_model.h5")
COMBINED_SCALER_PATH = os.path.join(MODEL_DIR, "combined_scaler.save")


def parse_log_line(line):
    """
    Parses a single log line using a flexible regex to find timestamp, level, thread, and message.
    Returns a tuple (datetime, level, thread, message).
    """
    # Regex to capture standard log formats
    match = re.match(r"^([\d\-]+\s[\d:,]+)\s*-\s*(\w+)\s*\[(.*?)\]\s*-\s*(.*)", line)
    if match:
        timestamp_str, level, thread, message = match.groups()
        try:
            # Attempt to parse timestamp with milliseconds
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
        except ValueError:
            # Fallback to parsing without milliseconds
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        return dt, level.strip(), thread.strip(), message.strip()
    return None, None, None, None

def get_log_df_from_session(request):
    """
    Retrieves log data from session, parses it, and returns a Pandas DataFrame.
    Returns None if no data is available.
    """
    log_content = request.session.get('uploaded_log_file')
    if not log_content:
        return None

    lines = log_content.strip().splitlines()
    logs = [parse_log_line(line) for line in lines]
    
    valid_logs = [log for log in logs if all(log)]
    if not valid_logs:
        return None

    df = pd.DataFrame(valid_logs, columns=["Timestamp", "Level", "Thread", "Message"])
    df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date
    df["Month"] = pd.to_datetime(df["Timestamp"]).dt.to_period("M")
    # This 'LogKey' is crucial for the combined model approach
    df["LogKey"] = df["Level"] + " - " + df["Message"]
    return df


def create_sequences(data, window=3):
    """
    Creates sequences for LSTM model training.
    """
    X, y = [], []
    if len(data) <= window:
        return np.array(X), np.array(y)
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)


# --- Core ML Views ---

@csrf_exempt
def Train_Model(request, user_name):
    """
    Trains the new combined LSTM model based on the logic from st3.py.
    This replaces the old method of training one model per error.
    """
    df = get_log_df_from_session(request)
    if df is None:
        messages.error(request, "Log file not found in session. Please upload a file first.")
        return redirect('Prediction_Task', user_name=user_name, calling_request='website')

    try:
        # Group by month and LogKey to get counts, similar to st3.py
        monthly_counts = df.groupby(["Month", "LogKey"]).size().unstack(fill_value=0).sort_index()

        if monthly_counts.empty or len(monthly_counts) < 4:
            messages.warning(request, "Not enough monthly data to train a reliable model (requires at least 4 months of logs).")
            return redirect('Prediction_Task', user_name=user_name, calling_request='website')

        # Scale the data using StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(monthly_counts)
        
        # Create sequences for training
        X, y = create_sequences(scaled_data, window=3)

        if X.shape[0] == 0:
            messages.warning(request, "Could not create training sequences from the log data.")
            return redirect('Prediction_Task', user_name=user_name, calling_request='website')
        
        # Reshape data for LSTM [samples, timesteps, features]
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

        # Define the LSTM model architecture from st3.py
        tf.keras.backend.clear_session()
        model = Sequential([
            Input(shape=(X.shape[1], X.shape[2])),
            LSTM(64, return_sequences=True),
            Dropout(0.4),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(y.shape[1]) # Output layer matches number of unique LogKeys
        ])
        model.compile(optimizer="adam", loss="mse")

        # Train the model
        model.fit(X, y, epochs=100, batch_size=8, verbose=0)

        # Save the trained model and the scaler
        model.save(COMBINED_MODEL_PATH)
        joblib.dump(scaler, COMBINED_SCALER_PATH)
        
        request.session['last_trained'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        messages.success(request, "New combined model has been trained and saved successfully!")

    except Exception as e:
        messages.error(request, f"An error occurred during model training: {str(e)}")
        traceback.print_exc()

    return redirect('Prediction_Task', user_name=user_name, calling_request='website')



@ensure_csrf_cookie
def Result_Forecast(request, user_name):
    """
    Handles forecasting for a specific message using the main combined model.
    """
    context = {'user_name': user_name}

    if request.method == 'POST':
        input_msg = request.POST.get('input_msg', '').strip()
        context['input_msg'] = input_msg

        if not input_msg:
            context['error_msg'] = 'Please enter a log message to forecast.'
            return render(request, 'result.html', context)
        
        # Check if the main model exists
        if not os.path.exists(COMBINED_MODEL_PATH) or not os.path.exists(COMBINED_SCALER_PATH):
            context['error_msg'] = 'The main prediction model has not been trained yet. Please train it on the Predictions page first.'
            return render(request, 'result.html', context)

        df = get_log_df_from_session(request)
        if df is None:
            context['error_msg'] = 'Log data not found. Please re-upload your file.'
            return render(request, 'result.html', context)

        try:
            # Load model and scaler
            model = load_model(COMBINED_MODEL_PATH, compile=False)
            scaler = joblib.load(COMBINED_SCALER_PATH)

            # Prepare data exactly as it was for training
            monthly_counts = df.groupby(["Month", "LogKey"]).size().unstack(fill_value=0).sort_index()
            
            # Find columns that match the user's search term
            matched_cols = [col for col in monthly_counts.columns if input_msg.lower() in col.lower()]
            
            if not matched_cols:
                context['error_msg'] = f"No log patterns found matching '{input_msg}'."
                return render(request, 'result.html', context)
            
            # For simplicity, we'll forecast for the first match.
            # A more advanced version could show all matches.
            target_col = matched_cols[0]
            context['level_guess'] = target_col.split(' - ')[0]

            # Scale the historical data
            scaled_data = scaler.transform(monthly_counts)

            # --- Forecasting ---
            def forecast_next(data, steps=7):
                output = []
                current = data[-3:].copy()
                for _ in range(steps):
                    x_input = current.reshape((1, current.shape[0], current.shape[1]))
                    yhat = model.predict(x_input, verbose=0)[0]
                    output.append(yhat)
                    current = np.vstack([current[1:], [yhat]])
                return np.array(output)

            forecast_scaled = forecast_next(scaled_data, steps=7)
            forecast_unscaled = scaler.inverse_transform(forecast_scaled)

            # Create a forecast DataFrame
            last_month = monthly_counts.index[-1].to_timestamp()
            future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=7, freq='MS').to_period("M")
            forecast_df = pd.DataFrame(forecast_unscaled, columns=monthly_counts.columns, index=future_months)
            
            # Combine historical and forecast data for the target column
            historical_series = monthly_counts[target_col]
            forecast_series = forecast_df[target_col].clip(0).round() # Clip at 0 and round

            full_series = pd.concat([historical_series, forecast_series])

            context['forecast_table'] = [{"Month": month.to_timestamp(), "Count": int(count)} for month, count in full_series.items()]
            context['forecast_actual'] = {'values': historical_series.tolist()}
            
            # Calculate stats for display
            context['total_historical'] = int(historical_series.sum())
            context['total_predicted'] = int(forecast_series.sum())

            if historical_series.iloc[-1] > 0:
                change = ((forecast_series.iloc[0] - historical_series.iloc[-1]) / historical_series.iloc[-1]) * 100
                context['forecast_percent_change'] = change
                context['forecast_percent_color'] = '#22c55e' if change >= 0 else "#ffffff" # Green for up, Red for down
            
        except Exception as e:
            context['error_msg'] = f"An error occurred during prediction: {e}"
            traceback.print_exc()

    return render(request, 'result.html', context)


def get_combined_error_forecast(request):
    """
    API-style view to get forecast data for all ERROR messages using the single combined model.
    """
    if not os.path.exists(COMBINED_MODEL_PATH) or not os.path.exists(COMBINED_SCALER_PATH):
        return JsonResponse({'error': 'Model not trained. Please go to the Predictions page and train the model.'}, status=404)

    df = get_log_df_from_session(request)
    if df is None:
        return JsonResponse({'error': 'Log data not found in session. Please upload a file.'}, status=404)

    try:
        model = load_model(COMBINED_MODEL_PATH, compile=False)
        scaler = joblib.load(COMBINED_SCALER_PATH)

        monthly_counts = df.groupby(["Month", "LogKey"]).size().unstack(fill_value=0).sort_index()

        # Check if historical data is sufficient
        if len(monthly_counts) < 3:
             return JsonResponse({'error': 'Not enough historical data (at least 3 months required) to make a prediction.'}, status=400)

        scaled_data = scaler.transform(monthly_counts)
        
        # --- Forecasting Logic ---
        def forecast_next(data, steps=7):
            output = []
            current = data[-3:].copy() # Use last 3 known steps to predict next one
            for _ in range(steps):
                x_input = current.reshape((1, current.shape[0], current.shape[1]))
                yhat = model.predict(x_input, verbose=0)[0]
                output.append(yhat)
                current = np.vstack([current[1:], [yhat]])
            return np.array(output)

        forecast_scaled = forecast_next(scaled_data, steps=7)
        forecast_unscaled = scaler.inverse_transform(forecast_scaled)

        # Create a forecast DataFrame
        last_month = monthly_counts.index[-1].to_timestamp()
        future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=7, freq='MS').to_period("M")
        forecast_df = pd.DataFrame(forecast_unscaled, columns=monthly_counts.columns, index=future_months)

        # Filter for error logs and prepare data for the frontend
        error_cols = [col for col in monthly_counts.columns if col.startswith("ERROR")]
        
        if not error_cols:
            return JsonResponse({'error': 'No ERROR logs found in the uploaded file to forecast.'}, status=404)
        
        color_cycle = itertools.cycle(mcolors.TABLEAU_COLORS.values())
        combined_error_data = []

        for col in error_cols:
            historical_data = monthly_counts[col].tolist()
            predicted_data = forecast_df[col].clip(0).round().tolist()
            
            # Skip if there's no activity at all
            if sum(historical_data) + sum(predicted_data) == 0:
                continue

            combined_error_data.append({
                'label': col.replace("ERROR - ", ""),
                'color': next(color_cycle),
                'months': [str(m) for m in monthly_counts.index] + [str(m) for m in forecast_df.index],
                'values': [int(v) for v in historical_data] + [int(v) for v in predicted_data]
            })

        return JsonResponse({'combined_error_data': combined_error_data})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': f'An unexpected error occurred: {str(e)}'}, status=500)










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

       
  
# --- Standard Django Views (User Management, Page Rendering, etc.) ---

def Prediction_Task(request, user_name, calling_request):
    """Handle file upload and render the main predictions page."""
    context = {'user_name': user_name}
    
    hour = datetime.now().hour
    if 5 <= hour < 12:
        context['greeting'] = f"Good Morning, {user_name}"
    elif 12 <= hour < 17:
        context['greeting'] = f"Good Afternoon, {user_name}"
    else:
        context['greeting'] = f"Good Evening, {user_name}"

    if request.method == 'POST' and 'log_file' in request.FILES:
        log_file = request.FILES['log_file']
        try:
            if log_file.size > 20 * 1024 * 1024: # 20MB limit
                messages.error(request, "File is too large. Maximum size is 20MB.")
            else:
                file_content = log_file.read().decode('utf-8', errors='ignore')
                if not file_content.strip():
                    messages.warning(request, "The uploaded file appears to be empty.")
                else:
                    request.session['uploaded_log_file'] = file_content
                    request.session['uploaded_file_name'] = log_file.name
                    request.session.modified = True
                    messages.success(request, f"Successfully uploaded '{log_file.name}'.")
        except Exception as e:
            messages.error(request, f"Failed to read file: {e}")
    
    # Pass session info to context for display
    context['uploaded_file_name'] = request.session.get('uploaded_file_name')
    context['last_trained'] = request.session.get('last_trained')

    return render(request, 'predictions.html', context)


def get_logs_from_session(request, level=None):
    """
    Helper function to get and filter logs from the session for display on the dashboard.
    `level` can be 'INFO', 'WARN', 'ERROR', or None (for all).
    """
    log_content = request.session.get('uploaded_log_file')
    if not log_content:
        return ["Log file not found. Please upload a file on the 'Predictions' page."]

    lines = log_content.strip().split('\n')
    if not level:
        return lines

    keywords = {
        'INFO': ['INFO'],
        'WARN': ['WARN', 'WARNING'],
        'ERROR': ['ERROR']
    }
    search_terms = keywords.get(level, [])
    
    filtered_lines = [line for line in lines if any(term in line.upper() for term in search_terms)]
    
    return filtered_lines if filtered_lines else [f"No logs found for level: {level}"]







from django.views.decorators.csrf import csrf_exempt
import joblib
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
    request.session['csv_data'] = ""
    if request.method == 'POST':
        User_Email = request.POST['email']
        Password = request.POST['password']
        request.session['user_email'] = User_Email

        # Fetch user details
        User_Details = User_Register.objects.filter(Email=User_Email, Password=Password).values()
        
        if User_Details:
            username = User_Details[0]['Name']  # ✅ get the username correctly
            return redirect('Prediction_Task', user_name=username, calling_request='forecast')  # ✅ use correct variable
        else:
            return render(request, 'login.html', {"message": "Credentials are wrong!"})

    return render(request, 'login.html')



from django.shortcuts import redirect
from django.contrib.auth import logout

def user_logout(request):
    logout(request)
    request.session.flush()
    return redirect('User_Login')

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
    csv_file_path=settings.CSV_FILE_PATH
    values=defaultdict(list)
    Time_values=defaultdict(list)
    list_values=dict()
    sub_string_data = {}
    log_values=defaultdict(list)
    substring_devided_count = defaultdict(lambda: {'INFO': 0, 'WARNING': 0, 'ERROR': 0}) 
    today = datetime.now().date()
    start_date=end_date=today
   
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