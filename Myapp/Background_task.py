from apscheduler.schedulers.background import BackgroundScheduler
import re
import io
import csv
import os
from datetime import datetime, timedelta
from django.http import HttpResponse
from django.conf import settings

# File paths
Datefile = settings.DATETIME_FILE_PATH
log_file_path = settings.LOG_FILE_PATH
file_path = settings.CSV_FILE_PATH

# Read initial logDateTime from the Datefile
try:
    with open(Datefile, mode='r') as file:
        csv_reader = list(csv.reader(file))
        logDateTime = csv_reader[1][0]  # Adjust index as needed
except:
    logDateTime="11/23/23 03:02:07"

original_datetime = datetime.strptime(logDateTime, "%m/%d/%y %H:%M:%S")

def log_to_csv():
    global original_datetime
    
    # Define substring lists to check for specific log events
    substring_list = ["logPsm4Paramters", "TrdRobot"]
    substring_list_2 = [
        "EPU PSU Temperature (Celsius)",
        "EPU Ambient Temperature (Celsius)",
        "EPU on-chip Temperature (Celsius)",
        "Battery Time Remaining (minutes)",
        "Battery charge (percentage)",
        "Battery Temperature (Celsius)",
        "UPS Temperature (Celsius)"
    ]

    # Function to parse each log line
    def parse_log_line(log_line):
        global original_datetime

        # Regex pattern to match the log format
        regex = r"(?P<timestamp>\d{3}:\d{2}:\d{2}\.\d{3}) (?P<log_level>INFO|WARNING|ERROR) (?P<event>.*)"
        match = re.match(regex, log_line)

        if match:
            Time = match.group('timestamp')
            log_level = match.group('log_level')  # Capture log level

            # Convert Time string to timedelta
            hours, minutes, seconds_with_milliseconds = map(float, Time.split(':'))
            duration = timedelta(hours=hours, minutes=minutes, seconds=int(seconds_with_milliseconds))

            # Combine original datetime with duration
            combined_datetime = original_datetime + duration

            # Handle overflow: If combined_datetime exceeds 24 hours, adjust it
            if combined_datetime.hour >= 24:
                combined_datetime += timedelta(days=combined_datetime.hour // 24)
                combined_datetime = combined_datetime.replace(hour=combined_datetime.hour % 24)

            original_datetime = combined_datetime
            try:
                with open(Datefile, 'w') as file:
        # Write the header and a newline
                    file.write('logDateTime\n')
        # Write the original_datetime as a formatted string with a newline
                    file.write(original_datetime.strftime("%m/%d/%y %H:%M:%S") + '\n')
            except Exception as e:
                print("Error writing to datetime CSV file:", str(e))
            # Create the formatted log line
            cleaned_line = match.group('event').strip()
            formatted_line = f"{combined_datetime.strftime('%Y-%m-%d %H:%M:%S')} {log_level} {cleaned_line}"

            # Find matching substrings
            substring = next((sub for sub in substring_list if sub in formatted_line), '')
            substring_list_2_template = next((formatted_line for sub in substring_list_2 if sub in formatted_line), '')

            if substring and substring_list_2_template:
                return combined_datetime.strftime('%Y-%m-%d %H:%M:%S'), formatted_line, substring
            return None, None, None

        else:
            return None, None, None

    # Create or overwrite the CSV file
    try:
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the header if the file is new
            if not file_exists or os.stat(file_path).st_size == 0:
                csv_writer.writerow(['DateTime', 'Event_Template', 'Search_string'])

            # Read log file, parse it, and write to CSV
            with open(log_file_path, 'r') as log_file:
                for line in log_file:
                    datetime_str, event_template, substring = parse_log_line(line)
                    if datetime_str:
                        csv_writer.writerow([datetime_str, event_template, substring])

        # Optionally truncate the log file to avoid duplicates
        with open(log_file_path, 'w') as log_file:
            log_file.truncate(0)

    except FileNotFoundError:
        print(f"Log file not found: {log_file_path}")
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")

# To run the log_to_csv function, you can schedule it or call it directly
# For example, you could call it in a view or during your application's startup.
log_to_csv()
