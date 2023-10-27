import torch
torch.set_num_threads(1)
import os
import math
import subprocess
from moviepy.editor import *
from pprint import pprint
import librosa
import torchaudio
import torchaudio.transforms as transforms
import IPython as ip
from moviepy.editor import AudioFileClip

class utils_wav:
    def __init__(self, carpeta):
        self.name = carpeta
        self.audio_files = []
        self.audio_dir = 'D:/Dtataset 2/' + carpeta +'/'

        self.current_directory = os.getcwd()
        self.SAMPLING_RATE = 16000

        USE_ONNX = False 
        # model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
        #                             model='silero_vad',
        #                             force_reload=True,
        #                             onnx=USE_ONNX)
        # self.model_clean_audio = model

        # (self.get_speech_timestamps,
        # self.save_audio,
        # self.read_audio,
        # self.VADIterator,
        # self.collect_chunks) = utils


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


    def cleanAudio(self, rutaArchivo, directorio,SAMPLING_RATE=16000):
        wav = self.read_audio(rutaArchivo, sampling_rate=SAMPLING_RATE)
        # get speech timestamps from full audio file
        speech_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=SAMPLING_RATE,threshold=0.8, return_seconds=False,window_size_samples=1024)
        pprint(speech_timestamps)
        print(len(speech_timestamps))

        chuncks = len(speech_timestamps)
        
        nombre_archivo = os.path.basename(rutaArchivo)
        nombre_archivo_sin_extension = os.path.splitext(nombre_archivo)[0]

        print(nombre_archivo)
        
        # merge all speech chunks to one audio
        self.save_audio('/'+directorio+'/clean_'+nombre_archivo,
                self.collect_chunks(speech_timestamps[0:chuncks], wav), sampling_rate=SAMPLING_RATE) 
        

    def cut_wav(self, audio='D:/Dtataset 2/1/cut30/0000_portuguese_nonscripted_1.wav'):
        print('AUDIO', audio)

        ruta= audio
        # Realiza el corte
        start_time = 10
        audio = AudioFileClip(ruta).subclip(start_time)
        duracion = 30
        audio = audio.set_duration(duracion)

        # Exporta el audio cortado como un archivo WAV temporal
        archivo_temporal = "audio_temporal.wav"
        audio.write_audiofile(archivo_temporal, codec="pcm_s16le")

        # Carga el archivo WAV temporal como un tensor
        waveform, sample_rate = torchaudio.load(archivo_temporal)

        # Elimina el archivo WAV temporal si es necesario
        os.remove(archivo_temporal)
        return waveform , sample_rate

            # Exportar el audio cortado
            # dur_audio.write_audiofile("./cut"+str(tiempo)+'/'+str(i)+'_'+ nombre_archivo)


    def extract_mfcc(self,file,n_mfcc, tipo=1):
        if tipo == 1:
            X, sample_rate = librosa.load(file, sr = self.SAMPLING_RATE)
            mfccs = librosa.feature.mfcc(y=X,sr = sample_rate, n_mfcc = n_mfcc)

            return mfccs
        
        else:
            waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
            transform = transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=13,
                melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
            )
            mfcc = transform(waveform)

            return mfcc
