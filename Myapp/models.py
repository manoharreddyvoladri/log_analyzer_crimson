from django.db import models

# Create your models here.
class Register(models.Model):
    Name=models.TextField()
    Email=models.EmailField()
    Password=models.TextField()

class User_Register(models.Model):
    Name=models.TextField()
    Email=models.EmailField()
    Password=models.TextField()

class Forgot_Password(models.Model):
    Name=models.TextField()
    Email=models.EmailField()
    DateTime=models.DateTimeField()
    OTP=models.TextField()
# models.py
class TrainedModel(models.Model):
    user = models.ForeignKey(User_Register, on_delete=models.CASCADE)
    message = models.TextField()
    model_path = models.FilePathField(path='PhaseI/models')
    created_at = models.DateTimeField(auto_now_add=True)
