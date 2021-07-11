import torch.nn as nn

hidden = lambda c_in, c_out: nn.Sequential(
     nn.Conv3d(c_in, c_out, (3,3,3)), # Convolutional layer
     nn.BatchNorm3d(c_out), # Batch Normalization layer
     nn.ReLU(), # Activational layer
     nn.MaxPool3d(2) # Pooling layer
 )


class MriNet(nn.Module):
    def __init__(self, c):
        super(MriNet, self).__init__()
        self.hidden1 = hidden(1, c)
        self.hidden2 = hidden(c, 2*c)
        self.hidden3 = hidden(2*c, 4*c)
        self.linear = nn.Linear(128*5*7*5, 2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x



