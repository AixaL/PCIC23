import os
import subprocess
import sys
import glob
import pandas as pd
# importing the zipfile module
from zipfile import ZipFile
from moviepy.editor import *

current_directory = os.getcwd()
print("Directorio actual:", current_directory)


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

def convert_to_wav(audio_dir):
    print(audio_dir)
    contenido = os.listdir(audio_dir)
    videos = []
    videoWav = ''
    try:
        for fichero in contenido:
            if os.path.isfile(os.path.join(audio_dir, fichero)) and fichero.endswith('.mp4'):
                print(fichero)
                videos.append(fichero)

        for video in videos:
            print(video)
            audioclip = AudioFileClip(audio_dir + video)
            videoWav= video.replace('mp4' , 'wav')
            audioclip.write_audiofile(audio_dir + videoWav) 
            runcmd('rm '+ audio_dir + '/' +video , verbose = True)

    except:
        print(video)
        runcmd('rm '+ audio_dir + '/' +video , verbose = True)
        runcmd('rm '+ audio_dir + '/' +videoWav , verbose = True)
        convert_to_wav('CCv2_part_49_audios/')


carpeta = 'CCv2_part_49_audios'
convert_to_wav(carpeta+'/')