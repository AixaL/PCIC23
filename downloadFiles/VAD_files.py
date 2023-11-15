import torch
torch.set_num_threads(1)
import os
import math
import subprocess
from moviepy.editor import *
# from IPython.display import Audio
from pprint import pprint

current_directory = os.getcwd()
SAMPLING_RATE = 16000

USE_ONNX = False 
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=USE_ONNX)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


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


def cleanAudio(rutaArchivo, directorio):
    wav = read_audio(rutaArchivo, sampling_rate=SAMPLING_RATE)
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE,threshold=0.8, return_seconds=False,window_size_samples=1024)
    pprint(speech_timestamps)
    print(len(speech_timestamps))

    chuncks = len(speech_timestamps)
    
    nombre_archivo = os.path.basename(rutaArchivo)
    nombre_archivo_sin_extension = os.path.splitext(nombre_archivo)[0]

    print(nombre_archivo)

    if os.path.exists(directorio):
        # self.runcmd('mkdir ' + directorio )
        os.makedirs(directorio, exist_ok=True)
        if chuncks > 0:
            save_audio(directorio+'/clean_'+nombre_archivo, collect_chunks(speech_timestamps[0:chuncks], wav), sampling_rate=SAMPLING_RATE)
    else:
        os.makedirs(directorio, exist_ok=True)
        if chuncks > 0:
            save_audio(directorio+'/clean_'+nombre_archivo,
                collect_chunks(speech_timestamps[0:chuncks], wav), sampling_rate=SAMPLING_RATE)
    
    # # merge all speech chunks to one audio
    # save_audio('/clean_'+directorio+'/clean_'+nombre_archivo,
    #            collect_chunks(speech_timestamps[0:chuncks], wav), sampling_rate=SAMPLING_RATE) 
    print('rm '+ rutaArchivo)
    
    runcmd('rm '+ rutaArchivo , verbose = True)

def cut_wav(audio, tiempo):
    ruta= audio
    print(audio)
    # Cargar el archivo de audio
    audio = AudioFileClip(audio)
    duracion = audio.duration

    print(duracion)
    print(math.floor(duracion/30))
    
    for i in range(math.floor(duracion/tiempo)):

        # Definir los puntos de inicio y finalización
        start_time = i*tiempo  # Tiempo de inicio en segundos

        # Realizar el corte
        cut_audio = audio.subclip(start_time)

        # Ajustar la duración del audio al valor deseado
        dur_audio = cut_audio.set_duration(tiempo)

        # Aplicar normalización al audio
        normalized_audio = dur_audio.audio_normalize

        nombre_archivo = os.path.basename(ruta)
        nombre_archivo_sin_extension = os.path.splitext(nombre_archivo)[0]

        print(nombre_archivo)

        # Exportar el audio cortado
        dur_audio.write_audiofile("./cut"+str(tiempo)+'/'+str(i)+'_'+ nombre_archivo)


# cleanAudio('D:\\Dtataset 2\\12\\0770_portuguese_nonscripted_1.wav') 

# cut_wav('D:\\Dtataset 2\\12\\0770_portuguese_nonscripted_1.wav',20) 

current_directory = os.getcwd()
with os.scandir(current_directory) as ficheros:
    subdirectorios = [fichero.name for fichero in ficheros if fichero.is_dir()]

for directorio in subdirectorios:
    if 'CCv2' in directorio and '_audios' in directorio:
        print(directorio)
        audio_files = [os.path.join(directorio, file) for file in os.listdir(directorio) if file.endswith(".wav")]
        print(audio_files)
        for file in audio_files:
            print(file)
            # cleanAudio(directorio+'/'+file)
            cleanAudio(file, directorio)

# Obtener todas las carpetas
#     por cada carpeta obtener el wav y quitarle los silencios
#     guardarlo con el mismo nombre. borrar el sucio

