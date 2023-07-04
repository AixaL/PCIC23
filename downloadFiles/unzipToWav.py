#Import data packages
import os
import sys
import glob
# importing the zipfile module
from zipfile import ZipFile

current_directory = os.getcwd()
print("Directorio actual:", current_directory)
audio_files = [os.path.join(current_directory, file) for file in os.listdir(current_directory) if file.endswith(".zip")]
print(audio_files)