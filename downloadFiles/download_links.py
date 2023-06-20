from moviepy.editor import *
import os
import scipy.io.wavfile as wav
import librosa
import opensmile
import scipy.stats as stats
import subprocess

#Import data packages
import os
import sys
import glob
import pandas as pd

#Import audio packages
import librosa.display
from scipy.io import wavfile
import scipy.io.wavfile
import sys

#Import plotting packages
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import seaborn as sns

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


ruta_archivo_links = ".\Links190623.txt"
datos_link = pd.read_csv(ruta_archivo_links, sep="    ",header=None,names=["name", "link"])

print(datos_link.head())

for index, row in datos_link.iterrows():
    file_name = row['name']
    cdn_link = row['link']
    link_wget= "wget -O "+ file_name +" "+ cdn_link
    print('"'+link_wget+'"')
    runcmd('"'+link_wget+'"' , verbose = True)
