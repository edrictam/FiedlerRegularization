"""
Define the neural network models used and a getter
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse.linalg as spsp

def get_model(modelname, hyperparams):
    input_size = hyperparams["input_size"]
    hidden_size = hyperparams["hidden_size"]
    num_classes = hyperparams["num_classes"]
    if modelname == "feedforward":
        return NeuralNet(input_size, hidden_size, num_classes)
    elif modelname == "feedforward_fiedler":
        return NeuralNetFiedler(input_size, hidden_size, num_classes)
    elif modelname == "feedforward_dropout":
        return NeuralNetDropout(input_size, hidden_size, num_classes, hyperparams["regularization_param"])
    else:
        raise Exception("model {} not supported".format(model))

class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu1(out)
        out = self.fc3(out)
        out = self.relu1(out)
        out = self.fc4(out)
        return out


class NeuralNetDropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, p = 0.5):
        super(NeuralNetDropout, self).__init__()
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p = p)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc4(out)
        return out

class NeuralNetFiedler(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetFiedler, self).__init__()
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)
        self.numVert = input_size + 3*hidden_size + num_classes
        self.laplacian = torch.zeros(self.numVert, self.numVert)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu1(out)
        out = self.fc3(out)
        out = self.relu1(out)
        out = self.fc4(out)
        return out


    def setLaplacian(self):
        self.laplacian = torch.zeros(self.numVert, self.numVert)
        input_size = self.input_size
        hidden_size = self.hidden_size

        layer1_range = slice(0,input_size)
        layer2_range = slice(input_size,input_size+hidden_size)
        layer3_range = slice(input_size+hidden_size,input_size+hidden_size*2)
        layer4_range = slice(input_size+hidden_size*2,input_size+hidden_size*3)
        layer5_range = slice(input_size+hidden_size*3,self.numVert)

        self.laplacian[layer1_range, layer2_range] = self.fc1.weight.t()
        self.laplacian[layer2_range, layer1_range] = self.fc1.weight
        self.laplacian[layer2_range, layer3_range] = self.fc2.weight.t()
        self.laplacian[layer3_range, layer2_range] = self.fc2.weight
        self.laplacian[layer3_range, layer4_range] = self.fc3.weight.t()
        self.laplacian[layer4_range, layer3_range] = self.fc3.weight
        self.laplacian[layer4_range, layer5_range] = self.fc4.weight.t()
        self.laplacian[layer5_range, layer4_range] = self.fc4.weight

        self.laplacian = torch.abs(self.laplacian)
        degrees = torch.sum(self.laplacian, dim = 1)
        self.laplacian = -1*self.laplacian
        ind = np.diag_indices(self.laplacian.shape[0])
        self.laplacian[ind[0], ind[1]] = degrees

    def getLaplacian(self):
        return self.laplacian

    def getNumVert(self):
        return self.numVert

    def getFiedlerVec(self):
        matrix = self.laplacian
        cov = matrix.detach().numpy()
        eigvals, eigvecs = spsp.eigs(cov, k = 2, which = "SR")
        return np.real(eigvecs[:,1])
