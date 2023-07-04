#Import data packages
import os
import subprocess
import sys
import glob
# importing the zipfile module
from zipfile import ZipFile
from moviepy.editor import *

current_directory = os.getcwd()
print("Directorio actual:", current_directory)
audio_files = [os.path.join(current_directory, file) for file in os.listdir(current_directory) if file.endswith(".zip")]
print(audio_files)

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

for file in audio_files:
    file_name = os.path.splitext(os.path.basename(file))[0]
    print(file_name)
    carpeta = file_name + '_audios'
    if (str(file_name) != 'CCv2_part_14') and (str(file_name) != 'CCv2_annotations'):
        print('entro')
    # loading the temp.zip and creating a zip object
        with ZipFile(file, 'r') as zObject:
            zObject.extractall(path=carpeta)
            convert_to_wav(carpeta+'/')
    else:
        print('Carpeta No')
