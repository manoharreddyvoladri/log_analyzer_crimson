from kafka import KafkaConsumer
import pickle
import pandas as pd
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#from views import config_strings

# Step 1: Load the model and encoders from the .pkl file
with open('/home/arun/kafka/onepickle/model_and_encoders.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    tfidf_vectorizer = data['tfidf_vectorizer']
    label_encoder = data['label_encoder']

# Step 2: Define the list of target strings to search for
target_strings = ['Battery charge (percentage)', 'Battery Temperature', 'UPS Temperature','EPU Ambient Temperature','EPU on-chip Temperature']
#target_strings = config_strings()

# Step 3: Set up Kafka consumer to consume messages from the Kafka topic
topic = 'logfile'  # Kafka topic name
bootstrap_servers = ['localhost:9092']  # Update with your Kafka broker addresses

consumer = KafkaConsumer(
    topic,
    bootstrap_servers=bootstrap_servers,
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='log-consumer-group',
    value_deserializer=lambda x: x.decode('utf-8')
)

# Step 4: Set up logging to store predictions in a .log file (without timestamp)
log_file = '/home/arun/kafka/Logfile_Operations/logs/logdata.log'  # Specify the path where you want to store the log file
#log_file = '/home/arun/kafka/onepickle/logdata_output.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(message)s',  # Exclude timestamp from the log format
)

print("Listening to Kafka topic...")

# Step 5: Infinite loop to continuously consume messages and log predictions
while True:
    try:
        # Step 6: Consume and filter messages in real-time
        for message in consumer:
            log_message = message.value.strip()
            cleaned_log_message = log_message.replace('\u00a0', ' ')  # Clean up message

            # Step 7: Check if the log message contains any of the target strings
            if any(target in cleaned_log_message for target in target_strings):
                # Wrap log_message in a DataFrame for tfidf_vectorizer transformation
                new_data = pd.DataFrame({'LogType': [cleaned_log_message]})
                # Step 8: Use the loaded TfidfVectorizer to transform new data
                new_data_encoded = tfidf_vectorizer.transform(new_data['LogType']).toarray()

                # Step 9: Make a prediction for the log message
                y_pred_encoded = model.predict(new_data_encoded)
                y_pred = label_encoder.inverse_transform(y_pred_encoded)

                # Step 11: Print and log the prediction
                log_entry = f"{log_message} -> Predicted status: {y_pred[0]}"
                print(log_entry)  # Print to console (optional)

                # Write to log file without timestamp
                logging.info(log_entry)
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        # Optionally add sleep or error handling logic before retrying

