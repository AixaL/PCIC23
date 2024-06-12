import torch as th
from torch import nn

import numpy as np
import matplotlib.pyplot as plt

import random
from itertools import islice as take

#Procesamiento de audio
import librosa
import scipy.io.wavfile as wav
import librosa.display
import IPython as ip
from moviepy.editor import *

# redes neuronales
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from pooling import StatsPooling, AttnPooling
from torch.optim import Adam
from torch import optim

# from torch.utils.tensorboard import SummaryWriter

# redes audio
import torchaudio
import torchaudio.transforms as T

# barras de progreso
from tqdm.auto import trange
import wandb

#Counter
import collections

# Files
from os.path import join
import tempfile

import pandas as pd
import copy
from librosa.util import fix_length

# Redes
# from ModelsLoss import QuartzNet_Cross1, QuartzNet_Cross2, LSTMDvector, Xvector
from Models_Soft2 import QuartzNet_Cross1, QuartzNet_Cross2, LSTMDvector_Gender,LSTMDvector_Age, Xvector
from ModelSeq import QuartzNet_single_gender, QuartzNet_single_age, Ensamble 


#-------------------------------------------------------------------

#Funciones 

def registra_info_tboard(writer, epoca, hist):
  for (m,v) in hist.items():
    writer.add_scalar(m, v[epoca], epoca)

def guarda_ckpt(ckptpath, modelo, epoca, opt):
  estado_modelo = {'epoch': epoca,
                   'model_state_dict': modelo.state_dict(),
                   'optimizer_state_dict': opt.state_dict()}
  th.save(estado_modelo, ckptpath)


def set_seed(seed=0):
    """Initializes pseudo-random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def exactitud(y_hat, y):
  cmp = y_hat.argmax(dim=-1) == y
  aciertos = th.count_nonzero(cmp)
  return aciertos / cmp.shape[0]

# reproducibilidad
set_seed()

def identity(x):
    return x

def label2index_age(label):
    return CLASS_IDX[label]

def label2index_gender(label):
    return CLASS_IDX2[label]

#-------------------------------------------------------------------------------

# 0 - Gender
# 1 - Age class
# 2 - Age Regression
# 3 - All Multitask

# parámetros de audio
# SAMPLE_RATE = 8000
SAMPLE_RATE = 16000
# tamaño de la ventana
n_fft = 1024
# tamaño del salto
hop_length = n_fft // 2

#Datasets dataloaders

#-------------------------------------- TIMIT ------------------------------
class TIMIT(Dataset):

    def __init__(self, df, waveform_tsfm=identity, label_tsfm=identity, cut=False, cut_sec=1, task1=0, task2=1):
        self.waveform_tsfm = waveform_tsfm
        self.label_tsfm = label_tsfm
        self.df = df
        self.cut = cut
        self.cut_sec = cut_sec

        self.samples = cut_sec * SAMPLE_RATE

        self.task1 = task1
        self.task2 = task2

    def __getitem__(self, i):
        # print(i)
        dato = self.df.iloc[i]
        path = dato['path_from_data_dir']
        edad = dato['age_group']
        edad_num = dato['age']
        genero = dato['gender']

        directorio_actual = os.getcwd()
        directorio_actual +='/temp'

        audio = AudioFileClip(path)
        duracion = audio.duration

        if duracion >= self.cut_sec and self.cut:
            # CORTAR EL AUDIO
            # if self.cut:
            start_time = 0  # Tiempo de inicio en segundos

            # Realizar el corte
            cut_audio = audio.subclip(start_time)
            
            # Ajustar la duración del audio al valor deseado
            dur_audio = cut_audio.set_duration(self.cut_sec)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

                # Exportar el audio cortado al archivo temporal
                dur_audio.write_audiofile(temp_path,verbose=False, logger=None)

            waveform, sample_rate = librosa.load(temp_path, sr=SAMPLE_RATE)
            os.remove(temp_path)
        else:
             
            waveform, sample_rate = librosa.load(path, sr = SAMPLE_RATE)
            padded_audio = fix_length(waveform, size=self.cut_sec*SAMPLE_RATE)
            # waveform1 = torch.from_numpy(waveform)
            # print(np.shape(padded_audio))

            waveform = padded_audio

        x = self.waveform_tsfm(waveform)


        # if self.task1 == 0 and self.task2==1:
        #     return x, genero, edad
        # elif self.task1 == 0 and self.task2 == 2:
        #     return x, genero, edad_num
        # else:
        return x, edad , genero , edad_num
    
    def __len__(self):
        return (len(self.df))
    
# ----------------------------------------------- CASUAL CONVERSATIONS V2 -------------------------------------------------
    
class CCV2(Dataset):

    def __init__(self, df, waveform_tsfm=identity, label_tsfm=identity, cut=False, cut_sec=1, task1=0, task2=1):
        self.waveform_tsfm = waveform_tsfm
        self.label_tsfm = label_tsfm
        self.df = df
        self.cut = cut
        self.cut_sec = cut_sec
        self.samples = cut_sec * SAMPLE_RATE


        self.task1 = task1
        self.task2 = task2

    def __getitem__(self, i):
        # print(i)
        dato = self.df.iloc[i]
        path = dato['file_path']
        edad = dato['age_group']
        edad_num = dato['age']
        genero = dato['gender']

        directorio_actual = os.getcwd()
        directorio_actual +='/temp'

        audio = AudioFileClip(path)
        duracion = audio.duration

        if duracion >= self.cut_sec and self.cut:
            # CORTAR EL AUDIO
            # if self.cut:
            start_time = 0  # Tiempo de inicio en segundos

            # Realizar el corte
            cut_audio = audio.subclip(start_time)
            
            # Ajustar la duración del audio al valor deseado
            dur_audio = cut_audio.set_duration(self.cut_sec)
        
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

                # Exportar el audio cortado al archivo temporal
                dur_audio.write_audiofile(temp_path,verbose=False, logger=None)

            waveform, sample_rate = librosa.load(temp_path, sr=SAMPLE_RATE)
            os.remove(temp_path)
        else:
            waveform, sample_rate = librosa.load(path, sr = SAMPLE_RATE)

        x = self.waveform_tsfm(waveform)

        # if self.task1 == 0 and self.task2==1:
        #     return x, genero, edad
        # elif self.task1 == 0 and self.task2 == 2:
        #     return x, genero, edad_num
        # else:
        return x, edad , genero , edad_num
    
    def __len__(self):
        return (len(self.df))
    

# ------------------------------ COMMON VOICE ----------------------------------------------------------
    
class CommonVoice2(Dataset):

    def __init__(self, df, waveform_tsfm=identity, label_tsfm=identity, cut=False, cut_sec=1, task1=0, task2=1):
        self.waveform_tsfm = waveform_tsfm
        self.label_tsfm = label_tsfm
        self.df = df
        self.cut = cut
        self.cut_sec = cut_sec

        self.task1 = task1
        self.task2 = task2
        self.samples = cut_sec * SAMPLE_RATE

    def __getitem__(self, i):
        # print(i)
        dato = self.df.iloc[i]
        path = dato['path']
        edad = dato['Age_Group']
        # edad = dato['Age_3Group_class']
        edad_num = dato['age']
        genero = dato['gender']

        directorio_actual = os.getcwd()
        directorio_actual +='/temp'

        audio = AudioFileClip(path)
        duracion = audio.duration
        
        if duracion >= self.cut_sec and self.cut:
            # CORTAR EL AUDIO
            # if self.cut:
            start_time = 0  # Tiempo de inicio en segundos

            # Realizar el corte
            cut_audio = audio.subclip(start_time)
            
            # Ajustar la duración del audio al valor deseado
            dur_audio = cut_audio.set_duration(self.cut_sec)
            
            # Crear un archivo temporal
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

                # Exportar el audio cortado al archivo temporal
                dur_audio.write_audiofile(temp_path,verbose=False, logger=None)

                # print(temp_path)

                # Cargar la forma de onda del archivo de audio temporal antes de salir del bloque 'with'
            waveform, sample_rate = librosa.load(temp_path, sr=SAMPLE_RATE)
            os.remove(temp_path)
        else:
            waveform, sample_rate = librosa.load(path, sr = SAMPLE_RATE)
            padded_audio = fix_length(waveform, size=self.cut_sec*SAMPLE_RATE)
            # waveform1 = torch.from_numpy(waveform)
            # print(np.shape(padded_audio))

            waveform = padded_audio

        # print(path)
        # waveform, sample_rate, label, *_ = super().__getitem__(i)
        x = self.waveform_tsfm(waveform)

        # print(x.shape)

        # if self.task1 == 0 and self.task2==1:
        #     return x, genero, edad
        # elif self.task1 == 0 and self.task2 == 2:
        #     return x, genero, edad_num
        # else:
        return x, edad , genero , edad_num
    
    def __len__(self):
        return (len(self.df))
    

# ---------------------------- TRANSFORMACIÓN DEL AUDIO -----------------------------------------------

class WaveformPadTruncate(nn.Module):

    def __init__(self, secs=1, sample_rate=SAMPLE_RATE, transform_type=0):
        super().__init__()
        self.samples = secs * sample_rate
        self.transform_type=transform_type
        self.sample_rate=sample_rate

    def forward(self, waveform_librosa):
        samples = len(waveform_librosa)
        wave = torch.tensor(waveform_librosa, dtype=torch.float32)
        waveform = torch.from_numpy(waveform_librosa)

        if samples < self.samples:
          waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
          difference = self.samples - samples
          padding = torch.zeros(1, difference)
          waveform = torch.cat([waveform, padding], 1)
          # print(waveform.shape)
          waveform= waveform
          # waveform= waveform.squeeze()

        elif samples > self.samples:
            start = random.randint(0, samples - self.samples)
            # Devuelve un nuevo tensor que es una versión reducida del tensor de entrada.
            waveform = waveform.narrow(1, start, self.samples) # (dimension, start, length)


        if self.transform_type==1:
          spectrograma = T.MelSpectrogram(n_fft=n_fft, hop_length=512)(waveform)
          spectrograma2 = spectrograma.flatten(start_dim=1)
          spectrograma3  = spectrograma.reshape(-1, 1)
          # print(spectrograma.shape)
          return spectrograma
        
        elif self.transform_type==2:
          
          # waveform = torch.from_numpy(waveform)
          mfcc = T.MFCC(n_mfcc=23,sample_rate=self.sample_rate)(waveform)
          mfcc = librosa.feature.mfcc(y=waveform_librosa, sr=self.sample_rate, hop_length=256, n_mfcc=30)
          return mfcc
        
        elif self.transform_type==3:
           f0, voiced_flag, voiced_probs = librosa.pyin(np.array(waveform),
                                             sr=SAMPLE_RATE,
                                             fmin=librosa.note_to_hz('C2'),
                                             fmax=librosa.note_to_hz('C7'))
           times = librosa.times_like(f0, sr=SAMPLE_RATE)
           return torch.from_numpy(times).float()

        else:
          return waveform
        
# ----------------------------------------------------------------------------------------------------------------------------------------
        
# DATA
        
# CCV2 classes
CLASSES_AGE = (
    'teens', 'twenties', 'thirties', 'fourties', 'fifties',
    'sixties', 'seventies', 'eighties')

# CCV2 classes
CLASSES_AGE_3 = (
    'Young', 'Adult', 'Senior')

NUM_CLASSES_GEN = 2
NUM_CLASES_AGE = len(CLASSES_AGE)
NUM_CLASES_AGE_3 = len(CLASSES_AGE_3)

# df_Timit = pd.read_csv("./data/train_data_all_completo.csv")
# df_CCv2 = pd.read_csv('./audios_CCV2_Train.csv')

df_Timit = pd.read_csv("./data/train_data_all_completo_3Ages.csv")
df_CCv2 = pd.read_csv('./Corpus/audios_CCV2_Train_3Ages.csv')
df_CommonV = pd.read_csv('./data/cv-corpus-17.0-2024-03-15/en/train_8class.csv')

df_ent, df_val = train_test_split(df_CommonV,
                                  test_size=0.2,
                                  shuffle=True)

df_ent = df_ent.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

#----------------------------------------------------------------------------------------------------------------------------------

# transform_type - tipo de transformación del audio
# 0 - Ninguna transformación solo el waveform
# 1 - Espectograma
# 2 - MFCC

BATCH_SIZE = 100

# DataLoaders

ds_ent = CommonVoice2(
    # directorio de datos
    df = df_ent,
    # transformación de la forma de onda
    waveform_tsfm=WaveformPadTruncate(transform_type=2, secs=2),
    # transformación de etiqueta
    label_tsfm=label2index_age,
    # si va a haber corte de audio
    cut=True,
    # corte de cuantos segundos
    cut_sec=2
)

dl_ent= DataLoader(
    ds_ent,
    # tamaño del lote
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

# --------------------------

ds_val = CommonVoice2(
    # directorio de datos
    df = df_val,
    # transformación de la forma de onda
    waveform_tsfm=WaveformPadTruncate(transform_type=2, secs=2),
    # transformación de etiqueta
    label_tsfm=label2index_age,
    # si va a haber corte de audio
    cut=True,
    # corte de cuantos segundos
    cut_sec=2
)

dl_val= DataLoader(
    ds_val,
    # tamaño del lote
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)


# --------------------------------------------- ENTRENAMIENTO ----------------------------------------------------------

# 0 - Gender
# 1 - Age class
# 2 - Age Regression
# 3 - All Multitask

def paso_ent(modelo,fp_edades,fp_genero,fp_reg,metrica_edades,metrica_genero, metrica_edadN, opt, X, y_edad, y_genero, y_reg, task1=1, task2=1):
  
  # se ponen los gradientes asociados a los parámetros a actualizaren en cero
  opt.zero_grad()

  # Le damos un peso a cada tarea
  w_genero = 0.3
  w_edad = 0.7
  w_reg = 0.7


  # se propagan las entradas para obtener las predicciones        
  if task1==0 and task2==0:
    #  y_hat_edad, y_hat_genero = modelo(X, y_edad) 
     y_hat_genero = modelo(X) 
     y_hat_genero = y_hat_genero.squeeze(1)

     # Sacamos las perdidas de cada tarea
     perdida_genero = F.binary_cross_entropy(y_hat_genero, y_genero.float()) #fp_genero

     # Calcular la pérdida total como la suma ponderada de las pérdidas individuales
     perdida_total = perdida_genero.float()
  # se propagan las entradas para obtener las predicciones        
  elif task1==0 and task2==1:
    #  y_hat_edad, y_hat_genero = modelo(X, y_edad) 
     y_hat_edad, y_hat_genero = modelo(X) 

     # sacamos las probabilidades de las clases de edad
     y_prob = F.softmax(y_hat_edad, 1)

     # sacamos las clases
     y_pred = torch.argmax(y_prob, 1).detach().cpu().numpy()


     # Sacamos las perdidas de cada tarea
     perdida_genero = F.binary_cross_entropy(y_hat_genero, y_genero.float()) #fp_genero
     perdida_edad = F.cross_entropy(y_hat_edad, y_edad) #fp_edades

     # Calcular la pérdida total como la suma ponderada de las pérdidas individuales
     perdida_total = (w_genero * perdida_genero.float()) + ( w_edad * perdida_edad.float())

  elif task1==1 and task2==1:
    #  y_hat_edad, y_hat_genero = modelo(X, y_edad) 
     y_hat_edad = modelo(X) 

     # sacamos las probabilidades de las clases de edad
     y_prob = F.softmax(y_hat_edad, 1)

     # sacamos las clases
     y_pred = torch.argmax(y_prob, 1).detach().cpu().numpy()

    #  print(y_hat_edad.shape)
    #  print(y_hat_edad)
    #  print(y_edad)


     # Sacamos las perdidas de cada tarea
     perdida_edad = F.cross_entropy(y_hat_edad, y_edad) #fp_edades

     # Calcular la pérdida total como la suma ponderada de las pérdidas individuales
     perdida_total = perdida_edad.float()

    #  print(perdida_total)

  elif task1==0 and task2==2:
     y_hat_reg, y_hat_genero = modelo(X)


    #  y_hat_genero = y_hat_genero.squeeze(1)
    #  y_hat_reg = y_hat_reg.squeeze(1)


    #  print(y_hat_genero)
     # Sacamos las perdidas de cada tarea
     perdida_genero = F.binary_cross_entropy(y_hat_genero, y_genero.float()) #fp_genero
    #  print(perdida_genero)
     perdida_reg = F.mse_loss(y_hat_reg, y_reg.float()) #fp_reg

    #  print(perdida_reg)

     # Calcular la pérdida total como la suma ponderada de las pérdidas individuales
     perdida_total = (w_genero * perdida_genero.float()) + (w_reg * perdida_reg.float())

    #  print(perdida_total)


  perdida_total = perdida_total.float()

  opt.zero_grad()
  perdida_total.backward() # se obtienen los gradientes
  opt.step() # se actualizan todos los parámetros del modelo

  with th.no_grad():
    perdida_paso = perdida_total.cpu().numpy()

    if task1==0 and task2==1:
       metrica_genero_paso = accuracy_score(y_genero,y_hat_genero.round())
       metrica_edad_paso = accuracy_score(y_edad, y_pred)
       macrof1= f1_score(y_edad.cpu(), y_pred, average='macro')
       
       return perdida_paso,metrica_genero_paso, metrica_edad_paso, macrof1
       
    elif task1==1 and task2==1:
       metrica_edad_paso = accuracy_score(y_edad, y_pred)
       macrof1= f1_score(y_edad.cpu(), y_pred, average='macro')
       
       return perdida_paso, metrica_edad_paso, macrof1
       
    elif task1==0 and task2==0:
       metrica_genero_paso = accuracy_score(y_genero,y_hat_genero.round())

       return perdida_paso, metrica_genero_paso
    elif task1==0 and task2==2:
       metrica_genero_paso = accuracy_score(y_genero,y_hat_genero.round())
       metrica_reg_paso = metrica_edadN(y_hat_reg, y_reg)

       return perdida_paso, metrica_genero_paso, metrica_reg_paso

  




def entrena(modelo,fp_edades,fp_genero, fp_reg, metrica_edades, metrica_genero, metrica_edadN, opt, entdl, valdl, disp, ckptpath, n_epocas=10, tbdir='runs/', task1=0, task2=1):
 
  n_lotes_ent = len(entdl)
  n_lotes_val = len(valdl)

  print(n_lotes_ent)
  
  if task1==0 and task2==1:
     hist = {'perdida_ent':np.zeros(n_epocas),
          'perdida_val': np.zeros(n_epocas),

          'macroF1_ent': np.zeros(n_epocas),
          'macroF1_val': np.zeros(n_epocas),
          
          'Accuracy_Edades_ent': np.zeros(n_epocas),
          'Accuracy_Edades_val': np.zeros(n_epocas),

          'Accuracy_Genero_ent': np.zeros(n_epocas),
          'Accuracy_Genero_val': np.zeros(n_epocas)}
     
  elif task1==0 and task2==0:
     hist = {'perdida_ent':np.zeros(n_epocas),
          'perdida_val': np.zeros(n_epocas),

          'Accuracy_Genero_ent': np.zeros(n_epocas),
          'Accuracy_Genero_val': np.zeros(n_epocas)}
  elif task1==1 and task2==1:
     hist = {'perdida_ent':np.zeros(n_epocas),
          'perdida_val': np.zeros(n_epocas),

          'macroF1_ent': np.zeros(n_epocas),
          'macroF1_val': np.zeros(n_epocas),
          
          'Accuracy_Edades_ent': np.zeros(n_epocas),
          'Accuracy_Edades_val': np.zeros(n_epocas)}
  elif task1==0 and task2==2:
     hist = {'perdida_ent':np.zeros(n_epocas),
          'perdida_val': np.zeros(n_epocas),

          'Accuracy_Genero_ent': np.zeros(n_epocas),
          'Accuracy_Genero_val': np.zeros(n_epocas),

          'MSE_Edad_ent': np.zeros(n_epocas),
          'MSE_Edad_val': np.zeros(n_epocas)}
     

#   tbwriter = SummaryWriter(tbdir)
  perdida_min = th.inf
  mejor_modelo = copy.deepcopy(modelo)


  for e in range(n_epocas):
    # bucle de entrenamiento
    modelo.train()
    for Xlote, ylote_edades, ylote_genero, ylote_reg, *_ in entdl:
      Xlote = Xlote.to(disp)
      ylote_edades = ylote_edades.to(disp)
      ylote_genero = ylote_genero.to(disp)

    #   ylote_reg = ylote_reg.to(disp)

      # perdida_paso, perdida_edad_paso, perdida_genero_paso, perdida_reg_paso, macrof1
      if task1==0 and task2==1:
         perdida_paso, metrica_genero_paso,metrica_edad_paso, macrof1 = paso_ent(modelo, fp_edades, fp_genero, 
                                                                                                    fp_reg, metrica_edades, 
                                                                                                    metrica_genero, metrica_edadN, 
                                                                                                    opt, Xlote, ylote_edades, 
                                                                                                    ylote_genero, ylote_reg,task1=0, task2=task2)
         
      elif task1==0 and task2==2:
         perdida_paso, metrica_genero_paso, metrica_reg_paso = paso_ent(modelo, fp_edades, fp_genero, 
                                                                                                    fp_reg, metrica_edades, 
                                                                                                    metrica_genero, metrica_edadN, 
                                                                                                    opt, Xlote, ylote_edades, 
                                                                                                    ylote_genero, ylote_reg,task1=0, task2=task2)
      elif task1==0 and task2==0:
         perdida_paso, metrica_genero_paso = paso_ent(modelo, fp_edades, fp_genero, 
                                                                                                    fp_reg, metrica_edades, 
                                                                                                    metrica_genero, metrica_edadN, 
                                                                                                    opt, Xlote, ylote_edades, 
                                                                                                    ylote_genero, ylote_reg,task1=0, task2=task2)
      elif task1==1 and task2==1:
         perdida_paso, metrica_edad_paso, macrof1 = paso_ent(modelo, fp_edades, fp_genero, 
                                                                                                    fp_reg, metrica_edades, 
                                                                                                    metrica_genero, metrica_edadN, 
                                                                                                    opt, Xlote, ylote_edades, 
                                                                                                    ylote_genero, ylote_reg,task1=task1, task2=task2)


      if task1==0 and task2==1:
         hist['perdida_ent'][e] += perdida_paso
         hist['macroF1_ent'][e] += macrof1
         hist['Accuracy_Edades_ent'][e] += metrica_edad_paso
         hist['Accuracy_Genero_ent'][e] += metrica_genero_paso

      if task1==1 and task2==1:
         hist['perdida_ent'][e] += perdida_paso
         hist['macroF1_ent'][e] += macrof1
         hist['Accuracy_Edades_ent'][e] += metrica_edad_paso
         print(hist)
         
      elif task1==0 and task2==2:
         hist['perdida_ent'][e] += perdida_paso
         hist['Accuracy_Genero_ent'][e] += metrica_genero_paso
         hist['MSE_Edad_ent'][e] += metrica_reg_paso

      elif task1==0 and task2==0:
         hist['perdida_ent'][e] += perdida_paso
         hist['Accuracy_Genero_ent'][e] += metrica_genero_paso
      

    # ---------------------------------------- VALIDACIÓN -------------------------------------------------------------------
      
    modelo.eval()

    with th.no_grad():
      for Xlote, ylote_edades, ylote_genero, ylote_reg, *_  in valdl:

        Xlote = Xlote.to(disp)
        ylote_edades = ylote_edades.to(disp)
        ylote_genero = ylote_genero.to(disp)
        ylote_reg = ylote_reg.to(disp)

        w_genero = 0.3
        w_edad = 0.7
        w_reg = 0.7

        if task1==0 and task2==1:
           y_hat_edades, y_hat_genero = modelo(Xlote)
        #    y_hat_edades, y_hat_genero = modelo(Xlote, ylote_edades)
           y_hat_genero = y_hat_genero.squeeze().float()
           # sacamos las probabilidades
           y_prob = F.softmax(y_hat_edades, 1)
           # sacamos las clases
           y_pred = torch.argmax(y_prob, 1)

           macrof1= f1_score(ylote_edades.cpu(), y_pred.cpu().numpy(), average='macro')

           perdida_genero = F.binary_cross_entropy(y_hat_genero, ylote_genero.float()) #fp_edades
           perdida_edad = F.cross_entropy(y_hat_edades, ylote_edades) #fp_genero

           # Calcular la pérdida total como la suma ponderada de las pérdidas individuales
           perdida_total = w_genero * perdida_genero.float() + w_edad * perdida_edad.float()

           metrica_genero_val = accuracy_score(ylote_genero,y_hat_genero.round())
           metrica_edad_val = accuracy_score(ylote_edades, y_pred )

           hist['perdida_val'][e] += perdida_total
           hist['macroF1_val'][e] += macrof1
           hist['Accuracy_Edades_val'][e] += metrica_edad_val
           hist['Accuracy_Genero_val'][e] += metrica_genero_val
        elif task1==1 and task2==1:
           y_hat_edades = modelo(Xlote)
          
           # sacamos las probabilidades
           y_prob = F.softmax(y_hat_edades, 1)
           # sacamos las clases
           y_pred = torch.argmax(y_prob, 1)

           macrof1= f1_score(ylote_edades.cpu(), y_pred.cpu().numpy(), average='macro')

           perdida_edad = F.cross_entropy(y_hat_edades, ylote_edades) #fp_genero

           # Calcular la pérdida total como la suma ponderada de las pérdidas individuales
           perdida_total = perdida_edad.float()

           metrica_edad_val = accuracy_score(ylote_edades, y_pred )

           hist['perdida_val'][e] += perdida_total
           hist['macroF1_val'][e] += macrof1
           hist['Accuracy_Edades_val'][e] += metrica_edad_val
           
        elif task1==0 and task2==2:
           y_hat_reg, y_hat_genero = modelo(Xlote)
        #    y_hat_genero, y_hat_reg = modelo(Xlote)

        #    print(y_hat_genero)
           y_hat_genero = y_hat_genero.squeeze().float()
           y_hat_reg = y_hat_reg.squeeze().float()

           perdida_genero = F.binary_cross_entropy(y_hat_genero, ylote_genero.float()) #fp_edades
           perdida_reg = F.mse_loss(y_hat_reg, ylote_reg) #fp_reg

           # Calcular la pérdida total como la suma ponderada de las pérdidas individuales
           perdida_total = (w_genero * perdida_genero.float()) +( w_reg * perdida_reg.float())

           metrica_genero_val = accuracy_score(ylote_genero,y_hat_genero.round())
           metrica_reg_val = metrica_edadN(y_hat_reg, ylote_reg)
           hist['perdida_val'][e] += perdida_total
           hist['Accuracy_Genero_val'][e] += metrica_genero_val
           hist['MSE_Edad_val'][e] += metrica_reg_val

        elif task1==0 and task2==0:
           y_hat_genero = modelo(Xlote)
        
           y_hat_genero = y_hat_genero.squeeze().float()
         
           perdida_genero = F.binary_cross_entropy(y_hat_genero, ylote_genero.float()) #fp_edades
          
           # Calcular la pérdida total como la suma ponderada de las pérdidas individuales
           perdida_total =  perdida_genero.float()

           metrica_genero_val = accuracy_score(ylote_genero,y_hat_genero.round())
          
           hist['perdida_val'][e] += perdida_total
           hist['Accuracy_Genero_val'][e] += metrica_genero_val

    if task1==0 and task2==1:
       hist['macroF1_ent'][e] /=  n_lotes_ent
       hist['macroF1_ent'][e] *= 100

       hist['Accuracy_Edades_ent'][e] /= n_lotes_ent
       hist['Accuracy_Edades_ent'][e] *= 100

       hist['macroF1_val'][e] /=  n_lotes_val
       hist['macroF1_val'][e] *= 100

       hist['Accuracy_Edades_val'][e] /= n_lotes_val
       hist['Accuracy_Edades_val'][e] *= 100

       hist['Accuracy_Genero_ent'][e] /= n_lotes_ent
       hist['Accuracy_Genero_ent'][e] *= 100 

       hist['Accuracy_Genero_val'][e] /= n_lotes_val
       hist['Accuracy_Genero_val'][e] *= 100

    elif task1==1 and task2==1:
       hist['macroF1_ent'][e] /=  n_lotes_ent
       hist['macroF1_ent'][e] *= 100

       hist['Accuracy_Edades_ent'][e] /= n_lotes_ent
       hist['Accuracy_Edades_ent'][e] *= 100

       hist['macroF1_val'][e] /=  n_lotes_val
       hist['macroF1_val'][e] *= 100

       hist['Accuracy_Edades_val'][e] /= n_lotes_val
       hist['Accuracy_Edades_val'][e] *= 100
       
    elif task1==0 and task2==2:
       
       hist['MSE_Edad_ent'][e] /= n_lotes_ent

       hist['MSE_Edad_val'][e] /= n_lotes_val
       hist['Accuracy_Genero_ent'][e] /= n_lotes_ent
       hist['Accuracy_Genero_ent'][e] *= 100 

       hist['Accuracy_Genero_val'][e] /= n_lotes_val
       hist['Accuracy_Genero_val'][e] *= 100

    elif task1==0 and task2==0:
       
       hist['Accuracy_Genero_ent'][e] /= n_lotes_ent
       hist['Accuracy_Genero_ent'][e] *= 100 

       hist['Accuracy_Genero_val'][e] /= n_lotes_val
       hist['Accuracy_Genero_val'][e] *= 100


    

    hist['perdida_ent'][e] /=  n_lotes_ent
    hist['perdida_val'][e] /=  n_lotes_val
    # hist['perdida_val'][e] =  hist['perdida_val'][e]*100
    
    # guardamos checkpoint y copiamos pesos y sesgos del modelo
    # actual si disminuye la metrica a monitorear
    if hist['perdida_val'][e] < perdida_min:
      mejor_modelo.load_state_dict(modelo.state_dict())
      guarda_ckpt(ckptpath, modelo, e, opt)

    # registra_info_tboard(tbwriter, e, hist)
    if task1==0 and task2==1:
       wandb.log({"Val Accuracy Gender": hist["Accuracy_Genero_val"][e],
               "Train Accuracy Gender": hist["Accuracy_Genero_ent"][e],

               "Val F1" : hist["macroF1_val"][e],
               "Train F1" : hist["macroF1_ent"][e],

               "Val Accuracy Ages" : hist["Accuracy_Edades_val"][e],
               "Train Accuracy Ages" : hist["Accuracy_Edades_ent"][e],
        
               "Val Total Loss": hist["perdida_val"][e],
               "Train Total Loss": hist["perdida_ent"][e] })
       

       print(f'\nÉpoca {e}:\n '
          'ENTRENAMIENTO: \n'
          f'macro_F1(E) = {hist["macroF1_ent"][e]:.3f},\n '
          f'Perdida(E) = {hist["perdida_ent"][e]:.3f}, \n'
          f'Accuracy_Edades(E) = {hist["Accuracy_Edades_ent"][e]:.3f},\n '
          f'Accuracy_Genero(E) = {hist["Accuracy_Genero_ent"][e]:.3f},\n'
         
          'VALIDACIÓN: \n'
          f'macro_F1(V) = {hist["macroF1_val"][e]:.3f},\n  '
          f'Perdida(V) = {hist["perdida_val"][e]:.3f},\n  '
          f'Accuracy_Edades(V) = {hist["Accuracy_Edades_val"][e]:.3f}\n '
          f'Accuracy_Genero(V) = {hist["Accuracy_Genero_val"][e]:.3f}\n '

          '---------------------------------------------------------------------')
    elif task1==0 and task2==2:
       wandb.log({"Val Accuracy Gender": hist["Accuracy_Genero_val"][e],
               "Train Accuracy Gender": hist["Accuracy_Genero_ent"][e],
            
               "Val MSE Age" : hist["MSE_Edad_val"][e],
               "Train MSE Age" : hist["MSE_Edad_ent"][e],

               "Val Total Loss": hist["perdida_val"][e],
               "Train Total Loss": hist["perdida_ent"][e] })
       

       print(f'\nÉpoca {e}:\n '
          'ENTRENAMIENTO: \n'
          f'Perdida(E) = {hist["perdida_ent"][e]:.3f}, \n'
          f'Accuracy_Genero(E) = {hist["Accuracy_Genero_ent"][e]:.3f},\n'
          f'MSE_Edad(E) = {hist["MSE_Edad_ent"][e]:.3f},\n '
          'VALIDACIÓN: \n'
          f'Perdida(V) = {hist["perdida_val"][e]:.3f},\n  '
          f'Accuracy_Genero(V) = {hist["Accuracy_Genero_val"][e]:.3f}\n '
          f'MSE_Edad(V) = {hist["MSE_Edad_val"][e]:.3f}\n '
          '---------------------------------------------------------------------')
       
    elif task1==0 and task2==0:
       wandb.log({"Val Accuracy Gender": hist["Accuracy_Genero_val"][e],
               "Train Accuracy Gender": hist["Accuracy_Genero_ent"][e],
            
               "Val Total Loss": hist["perdida_val"][e],
               "Train Total Loss": hist["perdida_ent"][e] })
       

       print(f'\nÉpoca {e}:\n '
          'ENTRENAMIENTO: \n'
          f'Perdida(E) = {hist["perdida_ent"][e]:.3f}, \n'
          f'Accuracy_Genero(E) = {hist["Accuracy_Genero_ent"][e]:.3f},\n'
         
          'VALIDACIÓN: \n'
          f'Perdida(V) = {hist["perdida_val"][e]:.3f},\n  '
          f'Accuracy_Genero(V) = {hist["Accuracy_Genero_val"][e]:.3f}\n '
          '---------------------------------------------------------------------')
    elif task1==1 and task2==1:
       wandb.log({
               "Val F1" : hist["macroF1_val"][e],
               "Train F1" : hist["macroF1_ent"][e],

               "Val Accuracy Ages" : hist["Accuracy_Edades_val"][e],
               "Train Accuracy Ages" : hist["Accuracy_Edades_ent"][e],
        
               "Val Total Loss": hist["perdida_val"][e],
               "Train Total Loss": hist["perdida_ent"][e] })
       

       print(f'\nÉpoca {e}:\n '
          'ENTRENAMIENTO: \n'
          f'macro_F1(E) = {hist["macroF1_ent"][e]:.3f},\n '
          f'Perdida(E) = {hist["perdida_ent"][e]:.3f}, \n'
          f'Accuracy_Edades(E) = {hist["Accuracy_Edades_ent"][e]:.3f},\n '
         
          'VALIDACIÓN: \n'
          f'macro_F1(V) = {hist["macroF1_val"][e]:.3f},\n  '
          f'Perdida(V) = {hist["perdida_val"][e]:.3f},\n  '
          f'Accuracy_Edades(V) = {hist["Accuracy_Edades_val"][e]:.3f}\n '

          '---------------------------------------------------------------------')
  return modelo, mejor_modelo, hist

# -------------------------------- Entrenamiento modelos -----------------------------------------------------


train_batch = next(iter(dl_ent))


TASA_AP = 0.0001
TASA_AP2 ='0.0001'

# DATASET = 'TIMIT'
DATASET = 'CommonVoice'

N_EPOCAS=100

DC = 'cuda:1' if th.cuda.is_available() else 'cpu'
# LOGDIR = './logs/Quartznet'
# LOGDIR = './logs/Dvector'
# LOGDIR = './logs/Dvector/Base'
LOGDIR = './logs/Quartznet/Base'
# LOGDIR = './logs/Dvector/Ensamble'
# LOGDIR = './logs/Xvector'

#-------------------------------

# Tasks
# 0 - Gender
# 1 - Age class
# 2 - Age Regression
# 3 - All Multitask

# red= Xvector((30*63), num_classes=NUM_CLASES_AGE, task1=0, task2=1)
red= QuartzNet_single_age(30, num_classes=NUM_CLASES_AGE, task=1)
# red= QuartzNet_single_age(63, num_classes=NUM_CLASES_AGE, task=1)
# red= LSTMDvector_Gender(126, num_classes=NUM_CLASES_AGE, task=0)
# red= LSTMDvector_Gender(63, num_classes=NUM_CLASES_AGE, task=0)
# red = QuartzNet_Cross2(30,num_classes=NUM_CLASES_AGE_3, task1=0, task2=1)

# modelA = LSTMDvector_Gender(126,num_classes=NUM_CLASES_AGE_3, task=0)
# # modelB = QuartzNet_single_age(30,num_classes=NUM_CLASES_AGE_3, task=1)
# modelB = LSTMDvector_Age(126,num_classes=NUM_CLASES_AGE_3, task=1)

# checkpoint = torch.load('./logs/Dvector/Base/red_LSTMDvector_TIMIT_0.0001_Gender_1.pt')

# modelA.load_state_dict(checkpoint['model_state_dict'])

# red = Ensamble(modelA, modelB)

model = red
perdida_edades = nn.CrossEntropyLoss(weight=None,reduction='mean',label_smoothing=0.01)
perdida_genero = nn.BCELoss(weight=None, reduction='mean')
perdida_edadN = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

metrica_edadN = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

redName = '/red_Quartz_'+DATASET+'_'+TASA_AP2+'_Age_8Class_30MFCC_1.pt'

opt = Adam(model.parameters(), lr=TASA_AP,weight_decay=0.01)
# opt = optim.SGD(model.parameters(), lr=TASA_AP, weight_decay=0.01)

# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="Quartznet",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": TASA_AP,
#     "weight_decay": "0.01",
#     "architecture": "Quartz Age 8Classes Macro F1 30MFCC",
#     "dataset": DATASET,
#     "epochs": N_EPOCAS,
#     "Batch": BATCH_SIZE,
#     "edadWeight":1.0,
#     "genWeight":1.0,
#     "regWeight":1.0,
#     "segundos":1,
#     "mfcc":30,
#     "secs":2,
#     "ages":8,
#     "opt":'ADAM',
#     "red" : redName,
#     "sample_rate":SAMPLE_RATE
#     }
# )


red_Xvect2, mejor_Xvect2, hist_Xvect2 = entrena(model, perdida_edades, perdida_genero, perdida_edadN,
                                   exactitud, exactitud, metrica_edadN,
                                   opt,
                                   dl_ent,
                                   dl_val,
                                   DC,
                                   LOGDIR + redName,
                                   n_epocas=N_EPOCAS,
                                   tbdir = LOGDIR, task1=1, task2=1)