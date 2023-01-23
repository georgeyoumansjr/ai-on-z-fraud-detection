from django import forms
from .models import ModelF

class AddCsvFile(forms.ModelForm):

    class Meta:
        model = ModelF
        fields = ['file_name','csv_file',]