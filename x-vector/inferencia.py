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

# Redes
from Models import QuartzNet, LSTMDvector, Xvector

TASK_ = 1
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
SAMPLE_RATE = 16000
# tamaño de la ventana
n_fft = 1024
# tamaño del salto
hop_length = n_fft // 2

#Datasets dataloaders

#-------------------------------------- TIMIT ------------------------------
class TIMIT(Dataset):

    def __init__(self, df, waveform_tsfm=identity, label_tsfm=identity, cut=False, cut_sec=1, task=3):
        self.waveform_tsfm = waveform_tsfm
        self.label_tsfm = label_tsfm
        self.df = df
        self.cut = cut
        self.cut_sec = cut_sec
        self.task = task

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

        x = self.waveform_tsfm(waveform)


        if self.task == 0:
            return x, genero
        elif self.task == 1:
            return x, edad
        elif self.task == 2:
            return x, edad_num
        else:
            return x, edad , genero , edad_num
    
    def __len__(self):
        return (len(self.df))
    
# ----------------------------------------------- CASUAL CONVERSATIONS V2 -------------------------------------------------
    
class CCV2(Dataset):

    def __init__(self, df, waveform_tsfm=identity, label_tsfm=identity, cut=False, cut_sec=1, task=3):
        self.waveform_tsfm = waveform_tsfm
        self.label_tsfm = label_tsfm
        self.df = df
        self.cut = cut
        self.cut_sec = cut_sec
        self.task = task

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
            print(path)
            waveform, sample_rate = librosa.load(path, sr = SAMPLE_RATE)

        x = self.waveform_tsfm(waveform)

        if self.task == 0:
            return x, genero
        elif self.task == 1:
            return x, edad
        elif self.task == 2:
            return x, edad_num
        else:
            return x, edad , genero , edad_num
    
    def __len__(self):
        return (len(self.df))
    

# ------------------------------ COMMON VOICE ----------------------------------------------------------
    
class CommonVoice2(Dataset):

    def __init__(self, df, waveform_tsfm=identity, label_tsfm=identity, cut=False, cut_sec=1, task=3):
        self.waveform_tsfm = waveform_tsfm
        self.label_tsfm = label_tsfm
        self.df = df
        self.cut = cut
        self.cut_sec = cut_sec
        self.task = task

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

        # print(path)
        # waveform, sample_rate, label, *_ = super().__getitem__(i)
        x = self.waveform_tsfm(waveform)

        if self.task == 0:
            return x, genero
        elif self.task == 1:
            return x, edad
        elif self.task == 2:
            return x, edad_num
        else:
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
        else:
          return waveform
        
# ----------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------
        
# DATA
        
# CCV2 classes
CLASSES_AGE = (
    'teens', 'twenties', 'thirties', 'forties', 'fifties',
    'sixties', 'seventies', 'eighties'
)

NUM_CLASSES_GEN = 2
NUM_CLASES_AGE = len(CLASSES_AGE)

df_Timit_train = pd.read_csv("./data/train_data_all_completo.csv")
df_CCv2_train = pd.read_csv('./audios_CCV2_Train.csv')

df_Timit_test = pd.read_csv("./data/test_data_all_completo.csv")
df_CCv2_test = pd.read_csv('./audios_CCV2_Test.csv')

# df_ent, df_val = train_test_split(df_Timit,
#                                   test_size=0.2,
#                                   shuffle=True)

# df_ent = df_ent.reset_index(drop=True)
# df_val = df_val.reset_index(drop=True)

df_ent= df_Timit_train
# df_ent= df_CCv2_train
# df_ent= df_Timit_train

df_val= df_Timit_test
# df_val= df_CCv2_test
# df_val= df_Timit_test

#----------------------------------------------------------------------------------------------------------------------------------

# transform_type - tipo de transformación del audio
# 0 - Ninguna transformación solo el waveform
# 1 - Espectograma
# 2 - MFCC

BATCH_SIZE = 100

# DataLoaders

ds_ent = TIMIT(
    # directorio de datos
    df = df_ent,
    # transformación de la forma de onda
    waveform_tsfm=WaveformPadTruncate(transform_type=2, secs=1),
    # transformación de etiqueta
    label_tsfm=label2index_age,
    # si va a haber corte de audio
    cut=True,
    # corte de cuantos segundos
    cut_sec=1,
    # Tarea que se va a hacer
    task=TASK_
)

dl_ent= DataLoader(
    ds_ent,
    # tamaño del lote
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

# --------------------------

ds_val = TIMIT(
    # directorio de datos
    df = df_val,
    # transformación de la forma de onda
    waveform_tsfm=WaveformPadTruncate(transform_type=2, secs=1),
    # transformación de etiqueta
    label_tsfm=label2index_age,
    # si va a haber corte de audio
    cut=True,
    # corte de cuantos segundos
    cut_sec=1,
    # Tarea que se va a hacer
    task=TASK_
)

dl_val= DataLoader(
    ds_val,
    # tamaño del lote
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)


def eval_epoch(dl, model, device, num_batches=None, task=0):

    print(task)
    """Evalua una época"""
    metrica_edadN = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    # Modelo en modo de evaluación
    # Congelar las estadísticas σ y μ
    model.eval()

    # evitamos que se registren las operaciones
    # en la gráfica de cómputo
    with torch.no_grad():

        
        # validación de la época con num_batches
        # si num_batches==None, se usan todos los lotes
        for batch in take(dl, num_batches):

            x = batch[0].to(device)
            y_true = batch[1].to(device)

            if task != 3:
                losses, accs, F1, MSE = [], [], [], []
                # hacemos inferencia para obtener los logits
                y_hat = model(x)
            else:
                losses, accs_age, accs_gender, F1, MSE = [], [], [], [], []
                y_hat_edad, y_hat_genero, y_hat_reg = model(x)

            if task == 0:
                loss = F.binary_cross_entropy(y_hat, y_true.float()) #fp_genero 
                acc = accuracy_score(y_true ,y_hat.round())

                # guardamos históricos
                losses.append(loss.item())
                accs.append(acc.item())

            elif task == 1:

                loss = F.cross_entropy(y_hat, y_true) #fp_edades
                # computamos las probabilidades
                y_prob = F.softmax(y_hat, 1)
                # obtenemos la clase predicha
                y_pred = torch.argmax(y_prob, 1)

                acc = accuracy_score(y_true, y_pred)
                f1= f1_score(y_true.cpu(), y_pred, average='weighted')

                # guardamos históricos
                losses.append(loss.item())
                accs.append(acc.item())
                F1.append(f1.item())


            elif task == 2:
                loss = F.mse_loss(y_hat, y_true.float()) #fp_reg

                mse = metrica_edadN(y_hat, y_true)

                                # guardamos históricos
                losses.append(loss.item())
                MSE.append(mse.item())

            else:
                y_edad = batch[1].to(device)
                y_genero = batch[2].to(device)
                y_reg = batch[3].to(device)

                # sacamos las probabilidades de las clases de edad
                y_prob = F.softmax(y_hat_edad, 1)

                # sacamos las clases
                y_pred = torch.argmax(y_prob, 1).detach().cpu().numpy()

                loss_reg = F.mse_loss(y_hat_reg, y_reg.float()) #fp_reg
                loss_gen = F.cross_entropy(y_hat_edad, y_edad) #fp_genero
                loss_age = F.binary_cross_entropy(y_hat_genero, y_genero.float()) #fp_edades

                loss = loss_reg + loss_gen + loss_age

                acc_gen = accuracy_score(y_genero,y_hat_genero.round())
                acc_age = accuracy_score(y_edad, y_pred)
                mse = metrica_edadN(y_hat_reg, y_reg) 

                f1= f1_score(y_edad.cpu(), y_pred, average='weighted')

                                # guardamos históricos
                losses.append(loss.item())
                accs_age.append(acc_age.item())
                accs_gender.append(acc_gen.item())
                F1.append(f1.item())
                MSE.append(mse.item())

            # computamos la exactitud
            # acc = (y_true == y_pred).type(torch.float32).mean()
        if task ==0:
            loss = np.mean(losses)
            acc = np.mean(accs)

            return loss, acc
        elif task == 1:
            loss = np.mean(losses)
            acc = np.mean(accs)
            f1 = np.mean(F1)

            return loss, acc, f1
        elif task == 2:
            loss = np.mean(losses)
            mse = np.mean(MSE)

            return loss, mse
        else:
            loss = np.mean(losses)
            acc_gen = np.mean(accs_gender)
            acc_age = np.mean(accs_age)
            f1 = np.mean(F1)
            mse = np.mean(MSE)

            return loss, acc_age, acc_gen, f1, mse

#---------------------------------------------------------------------------------------------------
     
# model= LSTMDvector(63, num_classes=NUM_CLASES_AGE, task=3)
# model= QuartzNet(30, task=3)
# model= Xvector((30*63), num_classes=NUM_CLASES_AGE, task=TASK)

#________________________________________________________________________________________________

#----- TASK = 0 GENDER TIMIT ----
# QUARTZNET
# './logs/Quartznet/red_Dvector_GENDER_TIMIT_0.00001_30MFCC.pt' #revisar Dvector
# Dvector

# Xvector
# './logs/Xvector/red_Xvector_GENDER_TIMIT_0.00001_30MFCC.pt'

#----- TASK = 0 GENDER CCV2 -------

#________________________________________________________________________________________________

#--- TASK = 1 AGE CLASS TIMIT -----------
# QUARTZNET
# './logs/Quartznet/red_Quartznet_AGE_CLASS_TIMIT_0.00001_.pt'

# Dvector
# './logs/Dvector/red_Dvector_AGE_CLASS_TIMIT_0.00001_.pt'

# Xvector
# './logs/Xvector/red_Xvector_AGE_CLASS_TIMIT_0.00001_30MFCC.pt'

#---- TASK = 1 AGE CLASS CCV2 ---------


#________________________________________________________________________________________________

#--- TASK = 2 AGE REG TIMIT -------
# QUARTZNET
# './logs/Quartznet/red_Quartznet_AGE_REG_TIMIT_0.00001_.pt'
        
# Dvector
# './logs/Dvector/red_Dvector_AGE_REG_TIMIT_0.00001_.pt'

# Xvector
# './logs/Xvector/red_Xvector_AGE_REG_TIMIT_0.001_30MFCC.pt'
# './logs/Xvector/red_Xvector_AGE_REG_TIMIT_0.0001_30MFCC.pt'
# './logs/Xvector/red_Xvector_AGE_REG_TIMIT_0.00001_30MFCC.pt'
# './logs/Xvector/red_Xvector_AGE_REG_TIMIT_0.000001_30MFCC.pt'

#----- TASK = 2 AGE REG CCV2 -------

#________________________________________________________________________________________________


#----- TASK = 3 MULTITASK TIMIT -------
# QUARTZNET
# './logs/Quartznet/red_Quartznet_TIMIT_Multitask.pt'
# Dvector
# './logs/Dvector/red_Dvector_Multitask_TIMIT_0.00001_.pt'
# Xvector
# './logs/Xvector/red_Xvector_Multitask_TIMIT_0.0001_.pt'
# './logs/Xvector/red_Xvector_Multitask_TIMIT_0.00001_.pt'

#------- TASK = 3 MULTITASK CCv2 --------
# QUARTZNET
# './logs/Quartznet/red_QuartzNet_Multitask_CCV2_0.00001_.pt'
# Dvector
# './logs/Dvector/red_Dvector_Multitask_CCV2_0.00001_.pt'
# Xvector
# './logs/Xvector/red_Xvector_Multitask_CCV2_0.0001_.pt'


MODEL= 'red_Quartznet_AGE_CLASS_TIMIT_0.00001_.pt'
PATH = './logs/Quartznet/'

# model= LSTMDvector(63, num_classes=NUM_CLASES_AGE, task=3)
model= QuartzNet(30, task=TASK_)
# model= Xvector((30*63), num_classes=NUM_CLASES_AGE, task=TASK_)

model.load_state_dict(torch.load(PATH + MODEL)['model_state_dict'])


print(MODEL + '   TIMIT Test' )

model.eval()


device2 = torch.device('cpu')

if TASK_ == 0:
    trn_loss, trn_acc_gen = eval_epoch(dl_ent, model.to(device2), device2, task=0)
    tst_loss, tst_acc_gen = eval_epoch(dl_val, model.to(device2), device2, task=0)

    print(f'trn_acc_gen={trn_acc_gen:5.2f} tst_acc_gen={tst_acc_gen:5.2f}')
    print(f'trn_loss={trn_loss:6.2f} tst_loss={tst_loss:6.2f}')

elif TASK_ == 1:
    trn_loss, trn_acc_age, trn_f1 = eval_epoch(dl_ent, model.to(device2), device2, task=1)
    tst_loss, tst_acc_age, tst_f1 = eval_epoch(dl_val, model.to(device2), device2, task=1)

    print(f'trn_acc_age={trn_acc_age:5.2f} tst_acc_age={tst_acc_age:5.2f}')
    print(f'trn_f1={trn_f1:5.2f} tst_f1={tst_f1:5.2f}')
    print(f'trn_loss={trn_loss:6.2f} tst_loss={tst_loss:6.2f}')

elif TASK_ == 2:
    trn_loss, trn_mse = eval_epoch(dl_ent, model.to(device2), device2, task=2)
    tst_loss, tst_mse = eval_epoch(dl_val, model.to(device2), device2, task=2)

    print(f'trn_mse={trn_mse:5.2f} tst_mse={tst_mse:5.2f}')
    print(f'trn_loss={trn_loss:6.2f} tst_loss={tst_loss:6.2f}')
else:
    tst_loss, tst_acc_age, tst_acc_gen, tst_f1, tst_mse = eval_epoch(dl_val, model.to(device2), device2, task=3)
    trn_loss, trn_acc_age, trn_acc_gen, trn_f1, trn_mse = eval_epoch(dl_ent, model.to(device2), device2, task=3)

    print(f'trn_acc_gen={trn_acc_gen:5.2f} tst_acc_gen={tst_acc_gen:5.2f}')

    print(f'trn_acc_age={trn_acc_age:5.2f} tst_acc_age={tst_acc_age:5.2f}')
    print(f'trn_f1={trn_f1:5.2f} tst_f1={tst_f1:5.2f}')

    print(f'trn_mse={trn_mse:5.2f} tst_mse={tst_mse:5.2f}')

    print(f'trn_loss={trn_loss:6.2f} tst_loss={tst_loss:6.2f}')


# trn_loss, trn_acc, trn_f1 = eval_epoch(dl_ent, model.to(device2), device2, task=1)
# trn_loss, trn_acc = eval_epoch(dl_ent, model.to(device2), device2, task=0)

# tst_loss, tst_acc, tst_f1 = eval_epoch(dl_val, model.to(device2), device2, task=1)
# tst_loss, tst_acc = eval_epoch(dl_val, model.to(device2), device2, task=0)

# print(f'trn_acc_age={trn_acc_age:5.2f} tst_acc_age={tst_acc_age:5.2f}')
# print(f'trn_f1={trn_f1:5.2f} tst_f1={tst_f1:5.2f}')

# print(f'trn_acc_gen={trn_acc_gen:5.2f} tst_acc_gen={tst_acc_gen:5.2f}')

# print(f'trn_mse={trn_mse:5.2f} tst_mse={tst_mse:5.2f}')

# print(f'trn_loss={trn_loss:6.2f} tst_loss={tst_loss:6.2f}')

# CCV2 classes
CLASSES_AGE = (
    'teens', 'twenties', 'thirties', 'fourties', 'fifties',
    'sixties', 'seventies', 'eighties'
)

CLASSES_GENDER = ('Hombre', 'Mujer')

# Diccionario de etiqueta a la clase:
cls_idx = {i: c for i, c in enumerate(CLASSES_AGE)}

# Función para pasar un lote de etiquetas a sus clases:
def names_batch(targets):
  cls_idxs = [cls_idx[y.item()] for y in targets]
  return cls_idxs

# Diccionarios para contar las predicciones de cada clase
correct_pred = {classname: 0 for classname in CLASSES_AGE}
total_pred = {classname: 0 for classname in CLASSES_AGE}

with torch.no_grad():
    for data in dl_val:
        if TASK_==0:
            x, genero  = data
            y_hat_genero = model(x)

            # _, predictions = torch.max(y_hat_genero, 1)
            # Obtenemos la predicción correcta para cada clase
            for label, prediction in zip(genero, y_hat_genero):
                if label == prediction:
                    correct_pred[CLASSES_GENDER[label.item()]] += 1
                total_pred[CLASSES_GENDER[label.item()]] += 1

        elif TASK_ ==1:
            x, edad = data
            y_hat_edad = model(x)

            _, predictions = torch.max(y_hat_edad, 1)
            # Obtenemos la predicción correcta para cada clase
            for label, prediction in zip(edad, predictions):
                if label == prediction:
                    correct_pred[CLASSES_AGE[label]] += 1
                total_pred[CLASSES_AGE[label]] += 1

        elif TASK_ == 2:
            x, edad_num = data
            y_hat_reg = model(x)

        else:
            x, edad , genero , edad_num = data
            y_hat_edad, y_hat_genero, y_hat_reg = model(x)

            _, predictions = torch.max(y_hat_edad, 1)
            # Obtenemos la predicción correcta para cada clase
            for label, prediction in zip(edad, predictions):
                if label == prediction:
                    correct_pred[CLASSES_AGE[label]] += 1
                total_pred[CLASSES_AGE[label]] += 1

if TASK_ == 0:
    # Accuracy para cada clase
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] > 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        else:
            print(f'Accuracy for class: {classname:5s} is 0 %')
elif TASK_ == 1 or TASK_ == 3:
        # Accuracy para cada clase
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] > 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        else:
            print(f'Accuracy for class: {classname:5s} is 0 %')