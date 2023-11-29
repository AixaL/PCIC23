import torch as th
import torch.nn as nn
import torch.nn.functional as F
from pooling import StatsPooling, AttnPooling
from TDNN import TDNN

# X-vector general para usar la salida 'embedding' y pasarlo por cada clasificador
class Xvector_Gen(nn.module):
    def __init__(self,dim_inicial, salida=240, dropout=0.0, extract=False):
        super(Xvector_Gen, self).__init__()

        self.tdnn1 = TDNN(dim_inicial, 512, 5, 1, 2, dropout)
        self.tdnn2 = TDNN(512, 512, 3, 2, 2, dropout)
        self.tdnn3 = TDNN(512, 512, 3, 3, 3, dropout)
        self.tdnn4 = TDNN(512, 512, 1, 1, 0, dropout)
        self.tdnn5 = TDNN(512, 1500, 1, 1, 0, dropout)
        # Statistics pooling layer
        self.pooling = StatsPooling()

        # Segment-level
        self.affine6 = nn.Linear(2 * 1500, 512)
        # self.batchnorm6 = nn.BatchNorm1d(512, eps=0.001, momentum=0.99,
        #                                  affine=False)
        # self.affine7 = nn.Linear(512, 512)
        # self.batchnorm7 = nn.BatchNorm1d(512, eps=0.001, momentum=0.99,
        #                                  affine=False)
        # self.output = nn.Linear(512, salida)
        # self.output.weight.data.fill_(0.)


    def forward(self, x):
        # Frame-level
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)

        # Statistics pooling layer
        x = self.pooling(x, 2)

        # Segment-level
        x = self.affine6(x)

        return x