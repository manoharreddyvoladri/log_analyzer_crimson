from kafka import KafkaProducer
import time
import os

# Initialize the producer to send raw string data
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: v.encode('utf-8')  # Serialize as plain UTF-8 string
)

# Path to your log file (update this path)
log_file_path = '/home/arun/kafka/onepickle/DSM_Test.log'  # Update this to your actual log file path

# Function to read the log file and send each line to Kafka
def produce_log_entries():
    if not os.path.isfile(log_file_path):
        print(f"Log file {log_file_path} not found.")
        return

    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            # Strip the line of any trailing newlines or spaces
            log_message = line.strip()

            # Send the plain log message as a raw string
            producer.send('logfile', value=log_message)
            print(f'Sent: {log_message}')

            # Optional: Control the sending rate
            time.sleep(1)

    producer.flush()  # Ensure all messages are sent
    producer.close()  # Close the producer

if __name__ == "__main__":
    produce_log_entries()


