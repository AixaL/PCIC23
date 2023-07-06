#Import data packages
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
    for fichero in contenido:
        if os.path.isfile(os.path.join(audio_dir, fichero)) and fichero.endswith('.mp4'):
            print(fichero)
            videos.append(fichero)

    for video in videos:
        print(audio_dir)
        audioclip = AudioFileClip(audio_dir + video)
        videoWav= video.replace('mp4' , 'wav')
        audioclip.write_audiofile(audio_dir + videoWav) 
        runcmd('rm '+ audio_dir + '/' +video , verbose = True)

def unzip():
    audio_files = [os.path.join(current_directory, file) for file in os.listdir(current_directory) if file.endswith(".zip")]
    print(audio_files)
    for file in audio_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_zip = os.path.splitext(os.path.basename(file))[1]
        print(file_name)
        carpeta = file_name + '_audios'
        if (str(file_name) != 'CCv2_part_14') and (str(file_name) != 'CCv2_annotations'):
            with ZipFile(file, 'r') as zObject:
                zObject.extractall(path=carpeta)
                convert_to_wav(carpeta+'/')
            runcmd('rm ' + file_name + file_zip )
        else:
            print('Carpeta No')

ruta_archivo_links = "Links190623.txt"
datos_link = pd.read_csv(ruta_archivo_links, sep=" ",header=None,names=["name", "link"])

print(datos_link.head())

for index, row in datos_link.iterrows():
    file_name = row['name']
    cdn_link = row['link']
    link_wget= 'wget -O '+ file_name +' "'+ cdn_link+'"'
    # link_wget= 'sudo wget "'+ cdn_link+'"'
    print(''+link_wget+'')
    runcmd(''+link_wget+'' , verbose = True)
    unzip()


