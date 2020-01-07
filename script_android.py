import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__)
#definire retea neuronala ->6 clase
class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__() #transforma o retea simpla in una de tip pytorch

        #convolutional layers ; fc= fully connected layers
        #in_features = 3, out_features = 100
        self.fc1 = nn.Linear(3, 100) #am 3 intrari
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        #self.fc5 = nn.Linear(100, 100)
        #self.fc6 = nn.Linear(100, 100)
        self.fc7 = nn.Linear(100, 7) #am 6 iesiri=> cu 6 da eraore, cu 7 nu da!!
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
        #x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x

scripted_module = torch.jit.script(Net())