import wandb

import torch as th
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile as wav
th.manual_seed(42)
np.random.seed(42)

DC = 'cuda:1' if th.cuda.is_available() else 'cpu'

# funciones aleatorias
import random
# tomar n elementos de una secuencia
from itertools import islice as take

# audio
import librosa
import librosa.display
import IPython as ip

# redes neuronales
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# redes audio
import torchaudio
import torchaudio.transforms as T
# redes visión
# import torchvision.models as tvm

# redes neuronales
from torch.utils.data import DataLoader
# from torchaudio.datasets import SPEECHCOMMANDS
# inspección de arquitectura
from torchinfo import summary

# barras de progreso
from tqdm.auto import trange

#Counter
import collections

# Files
from os.path import join
from moviepy.editor import *
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchaudio.transforms import MelSpectrogram

# Dataloader
import pandas as pd
from torchinfo import summary


# Lee el archivo de texto utilizando pd.read_csv()
df = pd.read_csv('../../../../../../media/ar/Expansion/CommonVoice2/clean_data_all.csv')

len(df)

# df.head()

# idx, cuentas = np.unique(df['age'], return_counts=True)
# plt.bar(x=idx, height=cuentas)
# plt.xlabel('Categoría')
# plt.ylabel('Número de audios')
# plt.xticks(rotation=90)
# plt.show()


# idx, cuentas = np.unique(df['gender'], return_counts=True)
# plt.bar(x=idx, height=cuentas)
# plt.xlabel('Categoría')
# plt.ylabel('Número de audios')
# plt.xticks(rotation=90)
# plt.show()


# unique_classes = df['age'].unique()
# print(unique_classes)

# Mapear las categorías de edad a valores numéricos
mapeo_edades = {'teens': 18, 'twenties': 20, 'thirties': 35, 'fourties': 45, 'fifties': 55, 'sixties': 65, 'seventies': 75, 'eighties': 80, 'nineties': 90}

# Crear una nueva columna 'age_numerico' usando el mapeo
df['age_numerico'] = df['age'].replace(mapeo_edades)

df['age'] = df['age'].replace({'teens': 0,
                                'twenties': 1,
                                'thirties':2,
                                'fourties':3,
                                'fifties':4,
                                'sixties': 5,
                                'seventies':6,
                                'eighties':7,
                                'nineties': 8})


df = df[df['gender'] != 'other']

# tamaño de la ventana
n_fft = 1024
# tamaño del salto
hop_length = n_fft // 2

df['gender'] = df['gender'].replace({'male': 0,
                                'female': 1})

df.head()
import os
ruta_expansion = os.path.abspath('../../../../../../media/ar/Expansion/')
print(ruta_expansion)

df['path'] = df['path'].replace(to_replace=r'^D:/', value='../../../../../../media/ar/Expansion/', regex=True)
# df['path'] = df['path'].replace(to_replace=r'^\\', value='/', regex=True)

# tamaño del lote
BATCH_SIZE = 40

# parámetros de audio
SECS = 1
SAMPLE_RATE = 16000

# parámetros FFT
N_FFT = 400
HOP_LENGTH = N_FFT // 2

# SpeechCommands classes
CLASSES_AGE = (
    'teens', 'twenties', 'thirties', 'fourties', 'fifties',
    'sixties', 'seventies', 'eighties', 'nineties'
)

CLASSES_GENDER =('male','female')

NUM_CLASSES = len(CLASSES_AGE)
CLASS_IDX = {c: i for i, c in enumerate(CLASSES_AGE)}
print(CLASS_IDX)

NUM_CLASSES2 = len(CLASSES_GENDER)
CLASS_IDX2 = {c: i for i, c in enumerate(CLASSES_GENDER)}
print(CLASS_IDX2)


def set_seed(seed=0):
    """Initializes pseudo-random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# reproducibilidad
set_seed()


NUM_CLASSES_GEN =len(CLASSES_GENDER)
NUM_CLASES_AGE = len(CLASSES_AGE)

def identity(x):
    return x

def label2index_age(label):
    return CLASS_IDX[label]

def label2index_gender(label):
    return CLASS_IDX2[label]


import tempfile


class CommonVoice2(Dataset):

    def __init__(self, df, waveform_tsfm=identity, label_tsfm=identity, cut=False, cut_sec=1):
        self.waveform_tsfm = waveform_tsfm
        self.label_tsfm = label_tsfm
        self.df = df
        self.cut = cut
        self.cut_sec = cut_sec

    def __getitem__(self, i):
        # print(i)
        dato = self.df.iloc[i]
        path = dato['path']
        edad = dato['age']
        edad_num = dato['age_numerico']
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
            
            # nombre_archivo = 'temp_'+i+'.wav'
            # print(nombre_archivo)

            # Crear un archivo temporal
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

                # Exportar el audio cortado al archivo temporal
                dur_audio.write_audiofile(temp_path,verbose=False, logger=None)

                # print(temp_path)

                # Cargar la forma de onda del archivo de audio temporal antes de salir del bloque 'with'
            waveform, sample_rate = librosa.load(temp_path, sr=16000)
            os.remove(temp_path)
        else:
            waveform, sample_rate = librosa.load(path, sr = 16000)

        # print(path)
        # waveform, sample_rate, label, *_ = super().__getitem__(i)
        x = self.waveform_tsfm(waveform)
        # y = self.label_tsfm(label)
        return x, edad , genero , edad_num
        # return x, edad, genero, edad_num, sample_rate
    
    def __len__(self):
        return (len(self.df))


from torchaudio.io import AudioEffector, CodecConfig


class WaveformPadTruncate(nn.Module):

    def __init__(self, secs=1, sample_rate=16000, transform_type=0):
        super().__init__()
        self.samples = secs * sample_rate
        self.transform_type=transform_type
        self.sample_rate=sample_rate

    def forward(self, waveform):
        samples = len(waveform)
        wave = torch.tensor(waveform, dtype=torch.float32)
        waveform = torch.from_numpy(waveform)

        if samples < self.samples:
          waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
          difference = self.samples - samples
          padding = torch.zeros(1, difference)
          waveform = torch.cat([waveform, padding], 1)
          # print(waveform.shape)
          waveform= waveform.squeeze()

        elif samples > self.samples:
            start = random.randint(0, samples - self.samples)
            # Devuelve un nuevo tensor que es una versión reducida del tensor de entrada.
            waveform = waveform.narrow(1, start, self.samples) # (dimension, start, length)


        if self.transform_type==1:
          spectrograma = T.MelSpectrogram(n_fft=n_fft, hop_length=hop_length)(waveform)
          spectrograma2 = spectrograma.flatten(start_dim=1)
          spectrograma3  = spectrograma.reshape(-1, 1)
          # print(spectrograma.shape)
          return spectrograma
        elif self.transform_type==2:
          
          # waveform = torch.from_numpy(waveform)
          mfcc = T.MFCC(n_mfcc=23,sample_rate=16000)(waveform)
          return mfcc
        else:
          return waveform


df = df.drop('client_id', axis=1)
df = df.drop('sentence', axis=1)
df = df.drop('up_votes', axis=1)
df = df.drop('down_votes', axis=1)
df = df.drop('accents', axis=1)
df = df.drop('variant', axis=1)
df = df.drop('locale', axis=1)
df = df.drop('segment', axis=1)


from sklearn.model_selection import train_test_split

df_ent, df_val = train_test_split(df,
                                  test_size=0.2,
                                  shuffle=True)


df_ent = df_ent.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)


# creamos un Dataset
ds_ent = CommonVoice2(
    # directorio de datos
    df = df_ent,
    # transformación de la forma de onda
    waveform_tsfm=WaveformPadTruncate(transform_type=2),
    # transformación de etiqueta
    label_tsfm=label2index_age,
    cut=True,
    cut_sec=1
)


dl_ent= DataLoader(
    ds_ent,
    # tamaño del lote
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)


# creamos un Dataset
ds_val = CommonVoice2(
    # directorio de datos
    df = df_val,
    # transformación de la forma de onda
    waveform_tsfm=WaveformPadTruncate(transform_type=2),
    # transformación de etiqueta
    label_tsfm=label2index_age,
    cut=True,
    cut_sec=1
)


dl_val= DataLoader(
    ds_val,
    # tamaño del lote
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)


it_ent = iter(ds_ent)
# waveform, edad, genero, edad_num, sr = next(it_ent)
waveform, edad , genero, edad_num= next(it_ent)

print(f'wave = {waveform}, Categoría = {edad}')

waveform.shape


df_ent.iloc[1]


path = df_ent.iloc[0]
print(path['path'])
waveform, sample_rate = librosa.load(path["path"], sr=16000)

print(len(waveform))
print(sample_rate)

# Cargar el archivo de audio
audio = AudioFileClip(path['path'])
duracion = audio.duration

# reproducimos
ip.display.Audio(waveform, rate=sample_rate)


# durations = []
# for i in trange(len(df_ent)):
#     path = df_ent.iloc[i]
#     waveform, sample_rate = librosa.load(path["path"],sr = 16000)
#     # waveform, sample_rate, label ,id_spk, utt = ds[i]
#     samples = len(waveform)
#     duracion_segundos = samples/sample_rate
#     durations.append(duracion_segundos)


# counter = collections.Counter(durations)

# print('Duraciones')
# print(counter)

# plt.figure(figsize=(10, 5))
# plt.hist(durations, bins=len(counter.keys())+1)
# plt.bar(counter.keys(), counter.values())
# plt.title('Histograma de duraciones')
# plt.xlabel('muestras')
# plt.ylabel('ejemplos')
# plt.show()


# # Crea el histograma
# plt.hist(durations, color='blue', edgecolor='black')

# # Agrega etiquetas y título
# plt.xlabel('Duración (segundos)')
# plt.ylabel('Frecuencia')
# plt.title('Histograma de Duraciones de Archivos de Audio')

# # Muestra el histograma
# plt.show()



df['age_numerico'].value_counts()


def exactitud(y_hat, y):
  cmp = y_hat.argmax(dim=-1) == y
  aciertos = th.count_nonzero(cmp)
  return aciertos / cmp.shape[0]


def guarda_ckpt(ckptpath, modelo, epoca, opt):
  estado_modelo = {'epoch': epoca,
                   'model_state_dict': modelo.state_dict(),
                   'optimizer_state_dict': opt.state_dict()}
  th.save(estado_modelo, ckptpath)



# from torch.utils.tensorboard import SummaryWriter

# def registra_info_tboard(writer, epoca, hist):
#   for (m,v) in hist.items():
#     writer.add_scalar(m, v[epoca], epoca)


from tqdm import tqdm
from sklearn.metrics import f1_score


def paso_ent(modelo,
             fp_edades,
             fp_genero,
             fp_reg,
             metrica_edades,
             metrica_genero,
             metrica_edadN,
             opt,
             X,
             y_edad,
             y_genero,
             y_reg):
  opt.zero_grad() # se ponen los gradientes asociados a los parámetros
                    # a actualizaren en cero

  y_hat_edad, y_hat_genero, y_hat_reg = modelo(X) # se propagan las entradas para obtener las predicciones

  # y_hat_genero = y_hat_genero.squeeze().float()
  # y_hat_reg = y_hat_reg.squeeze().float()


  # sacamos las probabilidades
  y_prob = F.softmax(y_hat_edad, 1)

  # sacamos las clases
  y_pred = torch.argmax(y_prob, 1).detach().cpu().numpy()

  # print(y_pred)
  # print(y_hat_edad)

  y_pred_genero = torch.round(y_hat_genero)

  # y_pred_detached = y_pred.detach()

  # perdida = F.cross_entropy(y_hat, y) # se calcula la pérdida
  # print('y_hat')
  # print(y_hat_edad.dtype)
  # print(y_hat_reg.dtype)
  # print(y_hat_genero.dtype)
  # print('y_')
  # print(y_genero.dtype)
  # print(y_edad.dtype)
  # print(y_reg.dtype)
  

  perdida_genero = F.binary_cross_entropy(y_hat_genero, y_genero.float()) #fp_edades
  perdida_edad = F.cross_entropy(y_hat_edad, y_edad) #fp_genero
  perdida_reg = F.mse_loss(y_hat_reg, y_reg.float()) #fp_reg

  # print('Perdidas')
  # print(perdida_genero.dtype)
  # print(perdida_edad.dtype)
  # print(perdida_reg.dtype)
  # Puedes ajustar los pesos según sea necesario
  w_genero = 1.0
  w_edad = 1.0
  w_reg = 1.0

  # print('Pesos')
  # print(w_genero.dtype)
  # print(w_edad.dtype)
  # print(w_reg.dtype)

  # Calcular la pérdida total como la suma ponderada de las pérdidas individuales
  perdida_total = w_genero * perdida_genero.float() + w_edad * perdida_edad.float() + w_reg * perdida_reg.float()
  perdida_total = perdida_total.float()

  

  # print(perdida_total.dtype)

  perdida_total.backward() # se obtienen los gradientes
  opt.step() # se actualizan todos los parámetros del modelo


  with th.no_grad():
    perdida_paso = perdida_total.cpu().numpy() # convertimos la pérdida (instancia de
                                         # Tensor de orden 0) a NumPy, para
                                         # lo que es necesario moverla a CPU
    # metricas_paso = metrica(y_hat, y)

    metrica_genero_paso = metrica_genero(y_hat_genero, y_genero.float())
    metrica_edad_paso = metrica_edades(y_hat_edad, y_edad)
    metrica_reg_paso = metrica_edadN(y_hat_reg, y_reg) 

    weightedf1= f1_score(y_pred_genero.cpu(), y_pred, average='weighted')


  # return perdida_paso, metricas_paso
  return perdida_paso, metrica_edad_paso, metrica_genero_paso, metrica_reg_paso, weightedf1


import copy

def entrena(modelo,
            fp_edades,
            fp_genero,
            fp_reg,
            metrica_edades,
            metrica_genero,
            metrica_edadN,
            opt,
            entdl,
            valdl,
            disp,
            ckptpath,
            n_epocas = 10,
            tbdir = 'runs/'):
  n_lotes_ent = len(entdl)
  n_lotes_val = len(valdl)

  hist = {'perdida_ent':np.zeros(n_epocas),
          'weightedF1_ent': np.zeros(n_epocas),
          'weightedF1_val': np.zeros(n_epocas),
          'perdida_val': np.zeros(n_epocas),
          'Accuracy_Edades_ent': np.zeros(n_epocas),
          'Accuracy_Edades_val': np.zeros(n_epocas),

          'Accuracy_Genero_ent': np.zeros(n_epocas),
          'Accuracy_Genero_val': np.zeros(n_epocas),

          'MSE_Edad_ent': np.zeros(n_epocas),
          'MSE_Edad_val': np.zeros(n_epocas)}

  # tbwriter = SummaryWriter(tbdir)
  perdida_min = th.inf
  mejor_modelo = copy.deepcopy(modelo)


  for e in range(n_epocas):
    # bucle de entrenamiento
    modelo.train()
    for Xlote, ylote_edades, ylote_genero, ylote_reg, *_ in entdl:
      Xlote = Xlote.to(disp)
      ylote_edades = ylote_edades.to(disp)
      ylote_genero = ylote_genero.to(disp)
      ylote_reg = ylote_reg.to(disp)

      # perdida_paso, perdida_edad_paso, perdida_genero_paso, perdida_reg_paso, weightedf1
      perdida_paso, metrica_edad_paso, metrica_genero_paso, metrica_reg_paso, weightedf1 = paso_ent(modelo,
                                            fp_edades,
                                            fp_genero,
                                            fp_reg,
                                            metrica_edades,
                                            metrica_genero,
                                            metrica_edadN,
                                            opt,
                                            Xlote,
                                            ylote_edades,
                                            ylote_genero,
                                            ylote_reg)

      hist['perdida_ent'][e] += perdida_paso
      hist['weightedF1_ent'][e] += weightedf1
      hist['Accuracy_Edades_ent'][e] += metrica_edad_paso
      hist['Accuracy_Genero_ent'][e] += metrica_genero_paso
      hist['MSE_Edad_ent'][e] += metrica_reg_paso

    # bucle de validación
    modelo.eval()
    with th.no_grad():
      for Xlote, ylote_edades, ylote_genero, ylote_reg, *_  in valdl:
        Xlote = Xlote.to(disp)
        ylote_edades = ylote_edades.to(disp)
        ylote_genero = ylote_genero.to(disp)
        ylote_reg = ylote_reg.to(disp)

        y_hat_edades, y_hat_genero, y_hat_reg = modelo(Xlote)

        y_hat_genero = y_hat_genero.squeeze().float()
        y_hat_reg = y_hat_reg.squeeze().float()

        # sacamos las probabilidades
        y_prob = F.softmax(y_hat_edades, 1)

        # sacamos las clases
        y_pred = torch.argmax(y_prob, 1)

        weightedf1= f1_score(ylote_edades.cpu(), y_pred.cpu().numpy(), average='weighted')

        perdida_genero = F.binary_cross_entropy(y_hat_genero, ylote_genero.float()) #fp_edades
        perdida_edad = F.cross_entropy(y_hat_edades, ylote_edades) #fp_genero
        perdida_reg = F.mse_loss(y_hat_reg, ylote_reg) #fp_reg

        # Puedes ajustar los pesos según sea necesario
        w_genero = 1.0
        w_edad = 1.0
        w_reg = 1.0

        # Calcular la pérdida total como la suma ponderada de las pérdidas individuales
        perdida_total = w_genero * perdida_genero.float() + w_edad * perdida_edad.float() + w_reg * perdida_reg.float()

        metrica_genero_val = metrica_genero(y_hat_genero, ylote_genero.float())
        metrica_edad_val = metrica_edades(y_hat_edades, ylote_edades)
        metrica_reg_val = metrica_edadN(y_hat_reg, ylote_reg)


        hist['perdida_val'][e] += perdida_total
        hist['weightedF1_val'][e] += weightedf1
        hist['Accuracy_Edades_val'][e] += metrica_edad_val
        hist['Accuracy_Genero_val'][e] += metrica_genero_val
        hist['MSE_Edad_val'][e] += metrica_reg_val

    hist['weightedF1_ent'][e] /=  n_lotes_ent
    hist['perdida_ent'][e] /=  n_lotes_ent
    hist['perdida_ent'][e] =  hist['perdida_ent'][e]*100
    hist['Accuracy_Edades_ent'][e] /= n_lotes_ent
    hist['Accuracy_Edades_ent'][e] *= 100

    hist['Accuracy_Genero_ent'][e] /= n_lotes_ent
    hist['Accuracy_Genero_ent'][e] *= 100

    hist['MSE_Edad_ent'][e] /= n_lotes_ent
    hist['MSE_Edad_ent'][e] *= 100


    hist['weightedF1_val'][e] /=  n_lotes_val
    hist['perdida_val'][e] /=  n_lotes_val
    hist['perdida_val'][e] =  hist['perdida_val'][e]*100

    hist['Accuracy_Edades_val'][e] /= n_lotes_val
    hist['Accuracy_Edades_val'][e] *= 100

    hist['Accuracy_Genero_val'][e] /= n_lotes_val
    hist['Accuracy_Genero_val'][e] *= 100

    hist['MSE_Edad_val'][e] /= n_lotes_val
    hist['MSE_Edad_val'][e] *= 100
    # guardamos checkpoint y copiamos pesos y sesgos del modelo
    # actual si disminuye la metrica a monitorear
    if hist['perdida_val'][e] < perdida_min:
      mejor_modelo.load_state_dict(modelo.state_dict())
      guarda_ckpt(ckptpath, modelo, e, opt)

    # registra_info_tboard(tbwriter, e, hist)

    wandb.log({"Validation_acc_genero": hist["Accuracy_Genero_val"][e],
               "Validation_acc_edades": hist["Accuracy_Edades_val"][e], "Validation_F1_edades": hist["weightedF1_val"][e],
               "Validation_MSE_reg": hist["MSE_Edad_val"][e],
                "Validation_loss_total": hist["perdida_val"][e] })
    
    wandb.log({"Train_acc_genero": hist["Accuracy_Genero_ent"][e], 
               "Train_acc_edades": hist["Accuracy_Edades_ent"][e], "Train_F1_edades": hist["weightedF1_ent"][e], 
               "Train_MSE_reg": hist["MSE_Edad_ent"][e],
                "Train_loss_total": hist["perdida_ent"][e] })

    print(f'\nÉpoca {e}:\n '
          'ENTRENAMIENTO: \n'
          f'weighted_F1(E) = {hist["weightedF1_ent"][e]:.3f},\n '
          f'Perdida(E) = {hist["perdida_ent"][e]:.3f}, \n'
          f'Accuracy_Edades(E) = {hist["Accuracy_Edades_ent"][e]:.3f},\n '
          f'Accuracy_Genero(E) = {hist["Accuracy_Genero_ent"][e]:.3f},\n'
          f'MSE_Edad(E) = {hist["MSE_Edad_ent"][e]:.3f},\n '
          'VALIDACIÓN: \n'
          f'weighted_F1(V) = {hist["weightedF1_val"][e]:.3f},\n  '
          f'Perdida(V) = {hist["perdida_val"][e]:.3f},\n  '
          f'Accuracy_Edades(V) = {hist["Accuracy_Edades_val"][e]:.3f}\n '
          f'Accuracy_Genero(V) = {hist["Accuracy_Genero_val"][e]:.3f}\n '
          f'MSE_Edad(V) = {hist["MSE_Edad_val"][e]:.3f}\n '
          '---------------------------------------------------------------------')

  return modelo, mejor_modelo, hist


DC = 'cuda:1' if th.cuda.is_available() else 'cpu'
LOGDIR = './logs/'
N_EPOCAS = 100

# compute_xvect = Xvector('cpu')
# input_feats = th.rand([5, 10, 40])
# outputs = compute_xvect(input_feats)
# outputs.shape

# Xvector()

# class Speechbrain_Xvector(nn.Module):
#     def __init__(self,d_modelo, n_class_edades, n_gen):
#         super().__init__()

#         self.xvect = Xvector()

#         self.age_regressor = nn.Sequential(
#             nn.Linear(d_modelo,d_modelo),
#             nn.ReLU(),
#             nn.Linear(d_modelo,1)
#         )
#     def forward(self,x):
#         print(x.shape)
#         x= x.flatten(start_dim=1)
#         print(x.shape)
#         xvector= self.xvect(x)
#         print(xvector.shape)
#         age_num= self.age_regressor(xvector)

#         return age_num


import torch as th
import torch.nn as nn
import torch.nn.functional as F
from pooling import StatsPooling, AttnPooling
from TDNN import TDNN

class FrotEnd(nn.Module):
    def __init__(self,dim_inicial, clases_age=NUM_CLASES_AGE, dropout=0.0, extract=False):
        super(FrotEnd, self).__init__()

        self.age_classification = nn.Sequential(
            nn.Linear(dim_inicial,dim_inicial),
            nn.ReLU(),
            nn.BatchNorm1d(dim_inicial),
            nn.Linear(dim_inicial,clases_age),
            # nn.Softmax()
        )

        self.gender = nn.Sequential(
            nn.Linear(dim_inicial,dim_inicial),
            nn.ReLU(),
            nn.BatchNorm1d(dim_inicial),
            nn.Linear(dim_inicial,1),
            nn.Sigmoid()
        )

        self.age_regression = nn.Sequential(
            nn.Linear(dim_inicial,dim_inicial),
            nn.ReLU(),
            nn.BatchNorm1d(dim_inicial),
            nn.Linear(dim_inicial,1)
        )

    def forward(self, x):
        age_classes = self.age_classification(x)
        age_classes = age_classes
        # age_classes = age_classes.float()
        
        gender = self.gender(x)
        gender = gender.squeeze().float()

        age_regression = self.age_regression(x)
        age_regression = age_regression.squeeze().float()

        return age_classes , gender , age_regression


class Xvector_Gen(nn.Module):
    def __init__(self,dim_inicial, dropout=0.0, extract=False):
        super(Xvector_Gen, self).__init__()

        self.tdnn1 = TDNN(dim_inicial, 400, 3, 1, 2, dropout)
        self.tdnn2 = TDNN(400, 400, 3, 2, 2, dropout)
        self.tdnn3 = TDNN(400, 400, 3, 3, 3, dropout)
        self.tdnn4 = TDNN(400, 400, 1, 1, 0, dropout)
        self.tdnn5 = TDNN(400, 1500, 1, 1, 0, dropout)
        # Statistics pooling layer
        self.pooling = StatsPooling()

        # Segment-level
        self.affine6 = nn.Linear(2 * 1500, 400)
        # self.batchnorm6 = nn.BatchNorm1d(512, eps=0.001, momentum=0.99,
        #                                  affine=False)
        # self.affine7 = nn.Linear(512, 512)
        # self.batchnorm7 = nn.BatchNorm1d(512, eps=0.001, momentum=0.99,
        #                                  affine=False)
        # self.output = nn.Linear(512, salida)
        # self.output.weight.data.fill_(0.)
        self.frontEnd = FrotEnd(400)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.flatten(start_dim=1)
        # print(x.shape)
        x= x.unsqueeze(dim=1)
        # print(x.shape)
        # Frame-level
        x= x.permute(0,2,1)
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)

        # Statistics pooling layer
        x = self.pooling(x, 2)

        # Segment-level
        x = self.affine6(x)
        x = self.relu(x)
        
        class_edad, genero, edad_num = self.frontEnd(x)

        return class_edad.float(), genero.float(), edad_num.float()


        return x


# class Red_Xvect(nn.Module):
#     def __init__(self,dim_inicial, clases_age=9, clases_gen=2, dropout=0.0, extract=False):
#         super(Red_Xvect, self).__init__()

#         self.Xvect = Xvector_Gen(dim_inicial)
#         self.frontEnd = FrotEnd(400)

#     def forward(self,x):
#         xvect= self.Xvect(x)
#         class_edad, genero, edad_num = self.frontEnd(xvect)

#         return class_edad, genero, edad_num

test_layer = Xvector_Gen(1863)


summary(test_layer, (40, 1,1863), device='cpu',col_names=['input_size', 'output_size', 'num_params'])

# train_batch = next(iter(dl_ent))

# train_batch[0].shape


# Red_xvect = Xvector_Gen(1863)
# edad, gen, edNum = Red_xvect(train_batch[0])

# gen.shape
# gen = gen.squeeze().float()
# gen.dtype

from torch.optim import Adam
red= Xvector_Gen(1863)
red.to(DC)
perdida_edades = nn.CrossEntropyLoss(weight=None,
                              reduction='mean',
                              label_smoothing=0.01)
perdida_genero = nn.BCELoss(weight=None, reduction='mean')
perdida_edadN = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

metrica_edadN = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

opt = Adam(red.parameters(),
           lr=0.00001)


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="XVector",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.00001,
    "architecture": "Xvector",
    "dataset": "CommonVoice",
    "epochs": 100,
    }
)


red_transformerSpeech, mejor_transformerSpeech, hist_transformerSpeech = entrena(red,
                                   perdida_edades,
                                   perdida_genero,
                                   perdida_edadN,
                                   exactitud,
                                   exactitud,
                                   metrica_edadN,
                                   opt,
                                   dl_ent,
                                   dl_val,
                                   DC,
                                   LOGDIR + '/red_xvector_multitask1.pt',
                                   n_epocas=N_EPOCAS,
                                   tbdir = LOGDIR)
