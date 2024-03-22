import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import math
from pooling import StatsPooling, AttnPooling
from TDNN import TDNN
from speechbrain.pretrained import EncoderClassifier

# ---------------------------------- Modulo FrontEnd ----------------------------------

# Tasks
# 0 - Gender
# 1 - Age class
# 2 - Age Regression
# 3 - All Multitask


class FrotEnd(nn.Module):
    def __init__(self,dim_inicial, clases_age=8, dropout=0.0, extract=False, task1=0, task2=1):
        super(FrotEnd, self).__init__()
        self.task1= task1
        self.task2= task2

        self.age_classification = nn.Sequential(
            nn.Linear(dim_inicial,dim_inicial),
            nn.ReLU(),
            nn.BatchNorm1d(dim_inicial),
            nn.Linear(dim_inicial,clases_age),
            nn.Softmax()
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

    def forward(self, x1, x2):

        if self.task1 == 0 and self.task2== 1: #Gender
            gender = self.gender(x1)
            gender = gender.squeeze().float()

            age_classes = self.age_classification(x2)
            age_classes = age_classes

            return gender, age_classes
        
        elif self.task == 0 and self.task2 ==2: #Age class

            gender = self.gender(x1)
            gender = gender.squeeze().float()

            age_regression = self.age_regression(x2)
            age_regression = age_regression.squeeze().float()

            return gender, age_regression
            
        else: #All 
            age_classes = self.age_classification(x1)
            age_classes = age_classes
            # age_classes = age_classes.float()

            age_regression = self.age_regression(x2)
            age_regression = age_regression.squeeze().float()

            gender = self.gender(x1)
            # gender = self.gender(x3)
            gender = gender.squeeze().float()

            return age_classes , gender , age_regression

# ---------------------------------------- Quartznet ---------------------------------------------

# blocks
def dense_norm_relu(in_size, out_size):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.BatchNorm1d(out_size),
        nn.ReLU()
    )

def conv_bn_act(in_size, out_size, kernel_size, stride=1, dilation=1, padding=0):
    return nn.Sequential(
        nn.Conv1d(in_size, out_size, kernel_size, stride, padding=padding, dilation=dilation),
        nn.BatchNorm1d(out_size),
        nn.ReLU()
    )


def sepconv_bn(in_size, out_size, kernel_size, stride=1, dilation=1, padding=None):
    if padding is None:
        padding = (kernel_size-1)//2
    return nn.Sequential(
        torch.nn.Conv1d(in_size, in_size, kernel_size, 
                        stride=stride, dilation=dilation, groups=in_size,
                        padding=padding),
        torch.nn.Conv1d(in_size, out_size, kernel_size=1),
        nn.BatchNorm1d(out_size)
    )


class QnetBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride=1,R=5):
        super().__init__()
       
        self.layers = nn.ModuleList(sepconv_bn(in_size, out_size, kernel_size, stride))
        for i in range(R - 1):
            self.layers.append(nn.ReLU())
            self.layers.append(sepconv_bn(out_size, out_size, kernel_size, stride))
        self.layers = nn.Sequential(*self.layers)

        self.residual = nn.ModuleList()
        #Point wise convolution
        self.residual.append(torch.nn.Conv1d(in_size, out_size, kernel_size=1))
        self.residual.append(torch.nn.BatchNorm1d(out_size))
        self.residual = nn.Sequential(*self.residual)

    def forward(self, x):
        return F.relu(self.residual(x) + self.layers(x))


class CrossStitch(nn.Module):
    def __init__(self, input_size):
        super(CrossStitch, self).__init__()
        # print(input_size)
        self.weights = nn.Parameter(torch.eye(input_size,dtype=torch.float32))

    def forward(self, input1, input2):


        input1_flattened = input1.view(input1.size(0), -1)
        input2_flattened = input2.view(input2.size(0), -1)

        concatenated_inputs = torch.cat((input1_flattened, input2_flattened), dim=1)

        cross_stitch = torch.eye(concatenated_inputs.size(1), dtype=torch.float32, requires_grad=True)

     
        # output = torch.matmul(concatenated_inputs, cross_stitch)
        output = torch.matmul(concatenated_inputs, self.weights)
        

        output1 = output[:, :input1_flattened.size(1)].view(input1.size())
        output2 = output[:, input1_flattened.size(1):].view(input2.size())

        return output1, output2



class QuartzNet_Cross1(nn.Module):
    def __init__(self, n_mels, num_classes=8, task1=0, task2=1):
        super().__init__()
        self.task1= task1
        self.task2= task2

        self.num_classes = num_classes

        self.c1_task1 = conv_bn_act(n_mels, 128, kernel_size=33,padding=(33-1)//2, stride=2)
        self.c1_task2 = conv_bn_act(n_mels, 128, kernel_size=33, padding=(33-1)//2, stride=2)
        self.block1_task1 = QnetBlock(128, 128, 33, 1, R=1)
        self.block2_task1 = QnetBlock(128, 256, 39, 1, R=1)
        self.block3_task1 = QnetBlock(256, 256, 51, 1, R=1)
        self.block4_task1 = QnetBlock(256, 256, 63, 1, R=1)
        self.block5_task1 = QnetBlock(256, 256, 75, 1, R=1)

        self.block1_task2 = QnetBlock(128, 128, 33, 1, R=1)
        self.block2_task2 = QnetBlock(128, 256, 39, 1, R=1)
        self.block3_task2 = QnetBlock(256, 256, 51, 1, R=1)
        self.block4_task2 = QnetBlock(256, 256, 63, 1, R=1)
        self.block5_task2 = QnetBlock(256, 256, 75, 1, R=1)
    
        self.cross_stitch1 = CrossStitch((256*32)*2) #16384
        # self.cross_stitch1 = CrossStitch((128*32)*2) #16384
        # self.cross_stitch2 = CrossStitch(2 * 256)
        # self.cross_stitch3 = CrossStitch(2*256)

        self.pooling_task1 = StatsPooling()
        self.pooling_task2 = StatsPooling()

        # self.lin_task1 = nn.Linear(256, 256)
        # self.lin_task2 = nn.Linear(256, 256)

        self.c2_task1 = conv_bn_act(256, 256, kernel_size=87,padding="same")
        self.c2_task2 = conv_bn_act(256, 256, kernel_size=87,padding="same")

        self.c3_task1 = conv_bn_act(256, 512, kernel_size=1)
        self.c3_task2 = conv_bn_act(256, 512, kernel_size=1)

        self.c4_task1 = conv_bn_act(512, 256, kernel_size=1, dilation=2)
        self.c4_task2 = conv_bn_act(512, 256, kernel_size=1, dilation=2)

        self.frontEnd = FrotEnd(256 * 32, clases_age= self.num_classes, task1=self.task1, task2=self.task2)


    def forward(self, x):

        c1_task1 = self.c1_task1(x)
        c1_task2 = self.c1_task2(x)

        #BLOQUE 1

        #Red 1
        x1 = self.block1_task1(c1_task1)
        # x1 = self.pooling(x1)

        #Red 2
        x2 = self.block1_task2(c1_task2)
        # x2 = self.pooling(x2)

        #BLOQUE 2
        #Red 1
        x1 = self.block2_task1(x1)
        # x1 = self.pooling(x1)
        #Red 2
        x2 = self.block2_task2(x2)
        # x2 = self.pooling(x2)

        #Cross2
        # output1, output2 = self.cross_stitch2(x1,x2)

        #BLOQUE 3
        #Red 1
        x1 = self.block3_task1(x1)
        x2 = self.block3_task2(x2)

        #Cross1
        # output1, output2 = self.cross_stitch1(x1,x2)

        x1 = self.block4_task1(x1)
        x2 = self.block4_task2(x2)
        
        x1 = self.block5_task1(x1)
        x2 = self.block5_task2(x2)
        # x2 = self.pooling_task2(x2)

        #Cross3
        output1, output2 = self.cross_stitch1(x1,x2)
        
       
        x1 = self.c2_task1(output1)
        x2 = self.c2_task2(output2)


        x1 = self.c3_task1(x1)
        x2 = self.c3_task2(x2)


        x1 = self.c4_task1(x1)
        x2 = self.c4_task2(x2)

        X1_flattened = x1.view(-1, 256 * 32)
        X2_flattened = x2.view(-1, 256 * 32)
        
        genero, edad = self.frontEnd(X1_flattened, X2_flattened)

        return edad.float(), genero.float()
        
#------------------------ Quartz 2 cross-------------------------

class QuartzNet_Cross2(nn.Module):
    def __init__(self, n_mels, num_classes=8, task1=0, task2=1):
        super().__init__()
        self.task1= task1
        self.task2= task2

        self.num_classes = num_classes

        self.c1_task1 = conv_bn_act(n_mels, 64, kernel_size=33,padding=(33-1)//2, stride=2)
        self.c1_task2 = conv_bn_act(n_mels, 64, kernel_size=33, padding=(33-1)//2, stride=2)
        
        self.dropOut1 = nn.Dropout(p=0.2)
        self.block1_task1 = QnetBlock(64, 64, 33, 1, R=1)
        self.dropOut2 = nn.Dropout(p=0.2)
        self.block2_task1 = QnetBlock(64, 128, 39, 1, R=1)
        self.dropOut3 = nn.Dropout(p=0.2)
        self.block3_task1 = QnetBlock(128, 128, 51, 1, R=1)
        self.dropOut4 = nn.Dropout(p=0.2)
        self.block4_task1 = QnetBlock(128, 128, 63, 1, R=1)
        self.dropOut5 = nn.Dropout(p=0.2)
        self.block5_task1 = QnetBlock(128, 128, 75, 1, R=1)

        self.dropOut1_2 = nn.Dropout(p=0.2)
        self.block1_task2 = QnetBlock(64, 64, 33, 1, R=1)
        self.dropOut2_2 = nn.Dropout(p=0.2)
        self.block2_task2 = QnetBlock(64, 128, 39, 1, R=1)
        self.dropOut3_2 = nn.Dropout(p=0.2)
        self.block3_task2 = QnetBlock(128, 128, 51, 1, R=1)
        self.dropOut4_2 = nn.Dropout(p=0.2)
        self.block4_task2 = QnetBlock(128, 128, 63, 1, R=1)
        self.dropOut5_2 = nn.Dropout(p=0.2)
        self.block5_task2 = QnetBlock(128, 128, 75, 1, R=1)
    
        # self.cross_stitch1 = CrossStitch((256*32)*2) #16384
        # self.cross_stitch1 = CrossStitch((128*32)*2) #16384
        self.cross_stitch1 = CrossStitch((64*32)*2) #16384
        self.cross_stitch2 = CrossStitch((128*32)*2)
        # self.cross_stitch2 = CrossStitch((256*32)*2)
        # self.cross_stitch3 = CrossStitch(2*256)

        self.pooling_task1 = StatsPooling()
        self.pooling_task2 = StatsPooling()

        # self.lin_task1 = nn.Linear(256, 256)
        # self.lin_task2 = nn.Linear(256, 256)

        self.c2_task1 = conv_bn_act(128, 128, kernel_size=87,padding="same")
        self.c2_task2 = conv_bn_act(128, 128, kernel_size=87,padding="same")

        self.c3_task1 = conv_bn_act(128, 256, kernel_size=1)
        self.c3_task2 = conv_bn_act(128, 256, kernel_size=1)

        self.c4_task1 = conv_bn_act(256, 128, kernel_size=1, dilation=2)
        self.c4_task2 = conv_bn_act(256, 128, kernel_size=1, dilation=2)

        self.frontEnd = FrotEnd(128 * 32, clases_age= self.num_classes, task1=self.task1, task2=self.task2)


    def forward(self, x):

        x1 = self.c1_task1(x)
        x2 = self.c1_task2(x)

        #BLOQUE 1

        #Red 1
        x1 = self.dropOut1(x1)
        x1 = self.block1_task1(x1)
        # x1 = self.pooling(x1)

        #Red 2
        x2 = self.dropOut1_2(x2)
        x2 = self.block1_task2(x2)
        # x2 = self.pooling(x2)


        #Cross1
        output1, output2 = self.cross_stitch1(x1,x2)

        #BLOQUE 2
        #Red 1
        x1 = self.dropOut2(output1)
        x1 = self.block2_task1(x1)
        # x1 = self.pooling(x1)
        #Red 2
        x2 = self.dropOut2_2(output2)
        x2 = self.block2_task2(x2)
        # x2 = self.pooling(x2)

        #Cross2
        # output1, output2 = self.cross_stitch2(x1,x2)

        #BLOQUE 3
        #Red 1
        x1 = self.dropOut3(x1)
        x1 = self.block3_task1(x1)
        x2 = self.dropOut3_2(x2)
        x2 = self.block3_task2(x2)

        #Cross1
        # output1, output2 = self.cross_stitch1(x1,x2)
        x1 = self.dropOut4(x1)
        x1 = self.block4_task1(x1)
        x2 = self.dropOut4_2(x2)
        x2 = self.block4_task2(x2)


        #Cross3
        output1, output2 = self.cross_stitch2(x1,x2)

        x1 = self.dropOut5(output1)
        x1 = self.block5_task1(x1)
        x2 = self.dropOut5_2(output2)
        x2 = self.block5_task2(x2)
        # x2 = self.pooling_task2(x2)

        
        x1 = self.c2_task1(x1)
        x2 = self.c2_task2(x2)


        x1 = self.c3_task1(x1)
        x2 = self.c3_task2(x2)


        x1 = self.c4_task1(x1)
        x2 = self.c4_task2(x2)

        X1_flattened = x1.view(-1, 128 * 32)
        X2_flattened = x2.view(-1, 128 * 32)
        
        genero, edad = self.frontEnd(X1_flattened, X2_flattened)

        return edad.float(), genero.float()
        

# ------------------------------------ D-Vector ---------------------------------------------------

class LSTMDvector(nn.Module):
    """LSTM-based d-vector."""
    def __init__(self, input_size, hidden_size=256 , embedding_size=256, num_layers=3,task=3, num_classes=8):
        super(LSTMDvector, self).__init__()

        self.task= task
        self.num_layers=num_layers
        self.hidden_size= hidden_size
        self.num_classes = num_classes
        
        self.lin = nn.Linear(input_size, input_size)
        self.norm = nn.BatchNorm1d(1)
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=num_layers,dropout=0.3, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers=num_layers,dropout=0.3, batch_first=True)
        self.lstm3 = nn.LSTM(input_size, hidden_size, num_layers=num_layers,dropout=0.3, batch_first=True)

        # Capa lineal para obtener el d-vector
        self.linear = nn.Linear(hidden_size, embedding_size)
        self.relu= nn.ReLU()

        self.frontEnd = FrotEnd(embedding_size,task=self.task, clases_age= self.num_classes)

    def forward(self, x):
        # Set initial hidden and cell states

        batch_size = x.size(0)
        # x = x.unsqueeze(1)  # Add a batch dimension
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()

        lstm_out1, _ = self.lstm1(x, (h0, c0))
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out3, _ = self.lstm3(lstm_out2)

        d_vector = self.linear(lstm_out1[:, -1, :])

        if self.task==3:
            class_edad, genero, edad_num = self.frontEnd(d_vector)

            return class_edad.float(), genero.float(), edad_num.float()
        else:
            single_class = self.frontEnd(d_vector)

            return single_class.float()

# ---------------------------------------- X-Vector ---------------------------------------------------

class Xvector(nn.Module):
    def __init__(self,dim_inicial, dropout=0.0, extract=False, task=3, num_classes=8):
        super(Xvector, self).__init__()
        self.task= task
        self.num_classes = num_classes

        self.tdnn1 = TDNN(dim_inicial, 400, 3, 1, 2, dropout)
        self.tdnn2 = TDNN(400, 400, 3, 2, 2, dropout)
        self.tdnn3 = TDNN(400, 400, 3, 3, 3, dropout)
        self.tdnn4 = TDNN(400, 400, 1, 1, 0, dropout)
        self.tdnn5 = TDNN(400, 1500, 1, 1, 0, dropout)
        # Statistics pooling layer
        self.pooling = StatsPooling()

        self.affine6 = nn.Linear(2 * 1500, 400)
        self.frontEnd = FrotEnd(400,clases_age= self.num_classes, task=self.task)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.flatten(start_dim=1)
        x= x.unsqueeze(dim=1)
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
        
        if self.task==3:
            class_edad, genero, edad_num = self.frontEnd(x)

            return class_edad.float(), genero.float(), edad_num.float()
        else:
            single_class = self.frontEnd(x)

            return single_class.float()


# -------------------------------- RESNET - VOXCELEB------------------------------------
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-resnet-voxceleb")

# Descongela las últimas capas
for module in classifier.modules():
    for param in module.parameters():
        param.requires_grad = False


class ResNetWithFrotEnd(nn.Module):
    def __init__(self, frotend_dim=256, num_classes=8):
        super(ResNetWithFrotEnd, self).__init__()
        self.resnet =  classifier # Asegúrate de proporcionar los parámetros adecuados
        self.frotend = FrotEnd(dim_inicial=256)

    def forward(self, x):
        # Pasa la entrada a través de la ResNet
        resnet_output = self.resnet.encode_batch(x, normalize=True)

        # Pasa la salida de la ResNet a través de FrotEnd
        age_classes, gender, age_regression = self.frotend(resnet_output)

        return age_classes, gender, age_regression