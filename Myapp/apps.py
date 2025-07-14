from django.apps import AppConfig
from .scheduler import start_scheduler

class MyappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Myapp'

    def ready(self):
        start_scheduler()
