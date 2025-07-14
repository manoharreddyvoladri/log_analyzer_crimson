from apscheduler.schedulers.background import BackgroundScheduler
from .Background_task import log_to_csv

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(log_to_csv, 'interval', seconds=10)
    scheduler.start()