from django.db import models

# Create your models here.

class ModelF(models.Model):
    file_name = models.CharField(max_length=250)
    csv_file = models.FileField(upload_to='csvs/',unique=True)
    added_dt = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ["file_name","csv_file"]

    def __str__(self):
        return self.file_name



