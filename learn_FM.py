import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# from nn_lib import (
#     MultiLayerNetwork,
#     Trainer,
#     Preprocessor,
#     save_network,
#     load_network,
# )
# from illustrate import illustrate_results_FM


def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    trainRatio = 0.8
    testRatio  = 0.1
    numEpochs  = 500
    batchsize  = 72

    dataX, dataY = dataset[:, 0:3], dataset[:, 3:6]
    # split the dataset
    dataX = torch.stack([torch.Tensor(i) for i in dataX])
    dataY = torch.stack([torch.Tensor(i) for i in dataY])
    # setup size ratios
    dataSize       = int(dataset.shape[0])
    trainingSize   = int(np.floor(dataSize * trainRatio))
    leftoverSize   = int(dataSize - trainingSize)
    testSize       = int(np.floor(leftoverSize * (testRatio/(1-trainRatio))))
    validationSize = leftoverSize - testSize

    trainX = dataX[:trainingSize]
    valX   = dataX[trainingSize:trainingSize + validationSize]
    testX  = dataX[trainingSize + validationSize:]
    trainY = dataY[:trainingSize]
    valY   = dataY[trainingSize:trainingSize + validationSize]
    testY  = dataY[trainingSize + validationSize:]
    # create your datset
    # loader = utils.DataLoader(utils.TensorDataset(dataX, dataY))
    trainloader = utils.DataLoader(
        utils.TensorDataset(trainX, trainY), batch_size=batchsize)
    validloader = utils.DataLoader(
        utils.TensorDataset(valX, valY), batch_size=batchsize)
    testloader = utils.DataLoader(
        utils.TensorDataset(testX, testY), batch_size=batchsize)
    # split datasets
    # trainloader, leftovers  = utils.random_split(loader, [trainingSize, leftoverSize])
    # validloader, testloader = utils.random_split(leftovers, [validationSize, testSize])
    # trainloader = utils.DataLoader(loader, sampler=trainloader)
    # validloader = utils.DataLoader(loader, sampler=validloader)
    # testloader  = utils.DataLoader(loader, sampler=testloader)
    # check for gpu.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # create the model.
    model =  nn.Sequential(
                nn.Linear(3, 12),
                nn.ReLU(),
                nn.Linear(12, 8),
                nn.ReLU(),
                nn.Linear(8, 4),
                nn.ReLU(),
                nn.Linear(4, 3),
                nn.ReLU()
            ).to(device)

    loss_function = nn.MSELoss()
    optimizer     = optim.Adam(model.parameters())

    for epoch in range(1, numEpochs):
        train_loss, valid_loss = [], []
        ## training part
        model.train()
        # print(trainloader)
        for data, target in trainloader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # ## evaluation part
        model.eval()

        for data, target in validloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            valid_loss.append(loss.item())
        print("Epoch:", epoch, "Training Loss: ", np.mean(
            train_loss), "Valid Loss: ", np.mean(valid_loss))

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_FM(network, prep)


if __name__ == "__main__":
    main()
