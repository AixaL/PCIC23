import torch as th
from torch import nn
import torchaudio
import torchaudio.transforms as T

import librosa
import scipy.io.wavfile as wav


import os
import csv
import umap
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from magenta.models.nsynth import utils
# from magenta.models.nsynth.wavenet import fastgen

import torchaudio
from speechbrain.pretrained import EncoderClassifier

import nemo.collections.asr as nemo_asr

np.random.seed(8)


#------------------------ WAV2VEC ------------------------

bundle = torchaudio.pipelines.WAV2VEC2_BASE

model = bundle.get_model()

waveform, sample_rate = torchaudio.load("D:/Dtataset 2/1/cut30/0000_portuguese_nonscripted_1.wav")

# audio_data
# extended_wav = th.from_numpy(audio_data)
# extended_wav.reshape(1, -1)

# print(extended_wav.shape)

features, _ = model.extract_features(waveform)

features[0].shape





#---------------- ECAPATDNN -----------------------

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='ecapa_tdnn')

embs = speaker_model.get_embedding('audio_path')

# --------------------------------------------------

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

signal, fs =torchaudio.load('tests/samples/ASR/spk1_snt1.wav')
embeddings = classifier.encode_batch(signal)




#---------------------- UMAP ------------------------

def get_scaled_umap_embeddings(features, neighbour, distance):
    
    embedding = umap.UMAP(n_neighbors=neighbour,
                          min_dist=distance,
                          metric='correlation').fit_transform(features)
    scaler = MinMaxScaler()
    scaler.fit(embedding)
    return scaler.transform(embedding)

umap_embeddings_mfccs = []
umap_embeddings_wavenet = []
neighbours = [5, 10, 15, 30, 50]
distances = [0.000, 0.001, 0.01, 0.1, 0.5]
for i, neighbour in enumerate(neighbours):
    for j, distance in enumerate(distances):
        umap_mfccs = get_scaled_umap_embeddings(mfcc_features,
                                                neighbour,
                                                distance)
        umap_wavenet = get_scaled_umap_embeddings(wavenet_features,
                                                  neighbour,
                                                  distance)
        umap_embeddings_mfccs.append(umap_mfccs)
        umap_embeddings_wavenet.append(umap_wavenet)
        
        mfcc_key = 'umapmfcc{}{}'.format(i, j) 
        wavenet_key = 'umapwavenet{}{}'.format(i, j) 
        
        all_json[mfcc_key] = transform_numpy_to_json(umap_mfccs)
        all_json[wavenet_key] = transform_numpy_to_json(umap_wavenet)


fig, ax = plt.subplots(nrows=len(neighbours), 
                       ncols=len(distances),
                       figsize=(30, 30))

for i, row in enumerate(ax):
    for j, col in enumerate(row):
        current_plot = i * len(iterations) + j
        col.scatter(umap_embeddings_mfccs[current_plot].T[0], 
                    umap_embeddings_mfccs[current_plot].T[1], 
                    s=1)
plt.show()