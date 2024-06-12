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
    def __init__(self,dim_inicial, clases_age=3, dropout=0.0, extract=False, task=0):
        super(FrotEnd, self).__init__()
        self.task= task

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

    def forward(self, x):

        if self.task == 0: #Gender
            gender = self.gender(x)
            gender = gender.squeeze().float()
            return gender
        
        elif self.task == 1: #Age class
            age_classes = self.age_classification(x)
            age_classes = age_classes
            return age_classes
            
        elif self.task == 2: #Age Regression
            age_regression = self.age_regression(x)
            age_regression = age_regression.squeeze().float()
            return age_regression
        
        else: #All 
            age_classes = self.age_classification(x)
            age_classes = age_classes
            # age_classes = age_classes.float()

            age_regression = self.age_regression(x)
            age_regression = age_regression.squeeze().float()

            gender = self.gender(x)
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


        # input1_flattened = input1.view(input1.size(0), -1)
        input1_flattened = input1.reshape(input1.size(0), -1)
        input2_flattened = input2.reshape(input1.size(0), -1)
        # input2_flattened = input2.view(input2.size(0), -1)

        concatenated_inputs = torch.cat((input1_flattened, input2_flattened), dim=1)

        cross_stitch = torch.eye(concatenated_inputs.size(1), dtype=torch.float32, requires_grad=True)

     
        # output = torch.matmul(concatenated_inputs, cross_stitch)
        output = torch.matmul(concatenated_inputs, self.weights)
        

        output1 = output[:, :input1_flattened.size(1)].view(input1.size())
        output2 = output[:, input1_flattened.size(1):].view(input2.size())

        return output1, output2


class QuartzNet_single_gender(nn.Module):
    def __init__(self, n_mels, num_classes=3, task=0):
        super(QuartzNet_single_gender, self).__init__()
        self.task= task

        self.num_classes = num_classes

        # self.dropOut1 = nn.Dropout(p=0.2)
        self.c1_task1 = conv_bn_act(n_mels, 64, kernel_size=33,padding=(33-1)//2, stride=2)
        
        self.block1_task1 = QnetBlock(64, 64, 33, 1, R=1)
        self.block2_task1 = QnetBlock(64, 128, 39, 1, R=1)
        self.block3_task1 = QnetBlock(128, 128, 51, 1, R=1)
        self.block4_task1 = QnetBlock(128, 128, 63, 1, R=1)
        self.block5_task1 = QnetBlock(128, 128, 75, 1, R=1)

        self.c2_task1 = conv_bn_act(128, 128, kernel_size=87, padding='same')
        self.c3_task1 = conv_bn_act(128, 256, kernel_size=1)
        self.c4_task1 = conv_bn_act(256, 128, kernel_size=1, dilation=2)

        self.frontEnd = nn.Sequential(
            nn.Linear(128 * 32,128 * 32),
            nn.ReLU(),
            nn.BatchNorm1d(128 * 32),
            nn.Linear(128 * 32,1),
            nn.Sigmoid()
        )
        # self.frontEnd = FrotEnd(256 * 32, clases_age= self.num_classes, task=self.task)


    def forward(self, x):
        # x = self.dropOut1(x)
        c1_task1 = self.c1_task1(x)

        #BLOQUES
        x1 = self.block1_task1(c1_task1)
        x1 = self.block2_task1(x1)
        x1 = self.block3_task1(x1)
        x1 = self.block4_task1(x1)
        x1 = self.block5_task1(x1)

        # x1 = self.dropOut2(x1)
        x1 = self.c2_task1(x1)

        # x1 = self.dropOut3(x1)
        x1 = self.c3_task1(x1)
        
        # x1 = self.dropOut4(x1)
        x1 = self.c4_task1(x1)


        X1_flattened = x1.view(-1, 128 * 32)
        
        single_class = self.frontEnd(X1_flattened)

        # print(single_class)
        single_class = single_class.squeeze()

        return single_class.float()



class QuartzNet_single_age_f0(nn.Module):
    def __init__(self, n_mels, num_classes=3, task=0):
        super().__init__()
        self.task= task

        self.num_classes = num_classes

        self.dropOut1 = nn.Dropout(p=0.2)
        self.c1_task1 = conv_bn_act(1, 64, kernel_size=33,padding=(33-1)//2, stride=2)
        # self.c1_task1 = conv_bn_act(n_mels, 64, kernel_size=33,padding=(33-1)//2, stride=2)
        
        self.block1_task1 = QnetBlock(64, 64, 33, 1, R=1)
        self.block2_task1 = QnetBlock(64, 128, 39, 1, R=1)
        self.block3_task1 = QnetBlock(128, 128, 51, 1, R=1)
        self.block4_task1 = QnetBlock(128, 128, 63, 1, R=1)
        self.block5_task1 = QnetBlock(128, 128, 75, 1, R=1)

        self.dropOut2 = nn.Dropout(p=0.2)
        self.c2_task1 = conv_bn_act(128, 128, kernel_size=87, padding='same')
        self.dropOut3 = nn.Dropout(p=0.2)
        self.c3_task1 = conv_bn_act(128, 256, kernel_size=1)
        self.dropOut4 = nn.Dropout(p=0.2)
        self.c4_task1 = conv_bn_act(256, 128, kernel_size=1, dilation=2)

        

        self.frontEnd = nn.Sequential(
            nn.Linear(128 * 32,128 ),
            # nn.Linear(128 * 63,128 ),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,self.num_classes),
            nn.Softmax()
        )
        # self.frontEnd = FrotEnd(256 * 32, clases_age= self.num_classes, task=self.task)


    def forward(self, x):
        # print(self.num_classes)
        # print(x.shape)
        # print(x.unsqueeze(0).shape)
        x = x.unsqueeze(1)
        # print(x)
        c1_task1 = self.c1_task1(x)

        #BLOQUES
        x1 = self.block1_task1(c1_task1)
        x1 = self.block2_task1(x1)
        x1 = self.block3_task1(x1)
        x1 = self.block4_task1(x1)
        x1 = self.block5_task1(x1)

        x1 = self.dropOut2(x1)
        x1 = self.c2_task1(x1)

        x1 = self.dropOut3(x1)
        x1 = self.c3_task1(x1)
        
        x1 = self.dropOut4(x1)
        x1 = self.c4_task1(x1)

        # print(x1.shape)


        X1_flattened = x1.view(-1, 128 * 32)
        # X1_flattened = x1.view(-1, 128 * 63)
        
        single_class = self.frontEnd(X1_flattened)

        # print(single_class)

        return single_class.float()


class QuartzNet_single_age(nn.Module):
    def __init__(self, n_mels, num_classes=3, task=0):
        super().__init__()
        self.task= task

        self.num_classes = num_classes

        self.dropOut1 = nn.Dropout(p=0.2)
        # self.c1_task1 = conv_bn_act(1, 64, kernel_size=33,padding=(33-1)//2, stride=2)
        self.c1_task1 = conv_bn_act(n_mels, 64, kernel_size=33,padding=(33-1)//2, stride=2)
        
        self.block1_task1 = QnetBlock(64, 64, 33, 1, R=1)
        self.block2_task1 = QnetBlock(64, 128, 39, 1, R=1)
        self.block3_task1 = QnetBlock(128, 128, 51, 1, R=1)
        self.block4_task1 = QnetBlock(128, 128, 63, 1, R=1)
        self.block5_task1 = QnetBlock(128, 128, 75, 1, R=1)

        self.dropOut2 = nn.Dropout(p=0.2)
        self.c2_task1 = conv_bn_act(128, 128, kernel_size=87, padding='same')
        self.dropOut3 = nn.Dropout(p=0.2)
        self.c3_task1 = conv_bn_act(128, 256, kernel_size=1)
        self.dropOut4 = nn.Dropout(p=0.2)
        self.c4_task1 = conv_bn_act(256, 128, kernel_size=1, dilation=2)

        

        self.frontEnd = nn.Sequential(
            nn.Linear(128 * 32,128 ),
            # nn.Linear(128 * 63,128 ),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,self.num_classes),
            nn.Softmax()
        )
        # self.frontEnd = FrotEnd(256 * 32, clases_age= self.num_classes, task=self.task)


    def forward(self, x):
        # print(self.num_classes)
        # print(x.shape)
        # print(x.unsqueeze(0).shape)
        # x = x.unsqueeze(1)
        # print(x)
        c1_task1 = self.c1_task1(x)

        #BLOQUES
        x1 = self.block1_task1(c1_task1)
        x1 = self.block2_task1(x1)
        x1 = self.block3_task1(x1)
        x1 = self.block4_task1(x1)
        x1 = self.block5_task1(x1)

        x1 = self.dropOut2(x1)
        x1 = self.c2_task1(x1)

        x1 = self.dropOut3(x1)
        x1 = self.c3_task1(x1)
        
        x1 = self.dropOut4(x1)
        x1 = self.c4_task1(x1)

        # print(x1.shape)


        X1_flattened = x1.view(-1, 128 * 32)
        # X1_flattened = x1.view(-1, 128 * 63)
        
        single_class = self.frontEnd(X1_flattened)

        # print(single_class)

        return single_class.float()

# ---------------------------------------- Ensamble ---------------------------------------------------

class Ensamble(nn.Module):
    def __init__(self, modelA, modelB, task1=0, task2=1, num_classes=3):
        super(Ensamble, self).__init__()
        self.task1= task1
        self.task2= task2
        self.num_classes = num_classes

        self.modelA = modelA
        self.modelB = modelB
        # self.classifier = FrotEnd(256 * 32, clases_age= self.num_classes, task=self.task2)
        
    def forward(self, x):
        with torch.no_grad():
            x1 = self.modelA(x)
        # print(x1)
        x1_expanded = x1.unsqueeze(2).expand(-1, 30, 126)
        # x1_expanded = x1.unsqueeze(1).unsqueeze(2).expand(-1, 30, 63)
        # print(x1_expanded.shape)
        x = torch.cat((x1_expanded, x), dim=1)
        x = self.modelB(x)
        # print(x)
        return x