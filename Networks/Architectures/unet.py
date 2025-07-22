import torch.nn as nn
import torch.nn.functional as F

######################################################################################
#
# CLASS DESCRIBING A FOOL MODEL ARCHITECTURE
# An instance of Net has been created in the model.py file
# 
######################################################################################

class Net(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.nb_channel = param["MODEL"]["NB_CHANNEL"]
        self.conv1 = nn.Conv2d(3, self.nb_channel, 3, padding='same') #first parameter = input channel(3 for RGB) kernel size = 3 for (3x3 conv)
        self.conv2 = nn.Conv2d(self.nb_channel, self.nb_channel, 3, padding='same')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(self.nb_channel, (self.nb_channel)*2, 3, padding='same')
        self.conv4 = nn.Conv2d((self.nb_channel)*2, (self.nb_channel)*2, 3, padding='same')

        self.pool2 = nn.MaxPool2d(2, 2)#good
        self.conv5 = nn.Conv2d((self.nb_channel)*8, (self.nb_channel)*8, 3, padding='same')
        self.conv6 = nn.Conv2d((self.nb_channel)*8, (self.nb_channel)*8, 3, padding='same')

        self.pool3 = nn.MaxPool2d(2, 2)#good
        self.conv7 = nn.Conv2d((self.nb_channel)*16, (self.nb_channel)*16, 3, padding='same')
        self.conv8 = nn.Conv2d((self.nb_channel)*16, (self.nb_channel)*16, 3, padding='same')
        #self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))

        x = self.pool2(x)
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))

        x = self.pool3(x)
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        #x = self.sigmoid(self.conv4(x))
        return x
