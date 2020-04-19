import torch
import torch.nn as nn
from torch import from_numpy
import logging
from evaluation import evaluate_model


# here we train the model
def train_model(regularization, hyperparams, model, train_loader, validation_loader):

    # check if gpu is available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # unpack hyperparameters
    num_epochs = hyperparams["num_epochs"]
    learning_rate = hyperparams["learning_rate"]
    regularization_param = hyperparams["regularization_param"]
    input_size = hyperparams["input_size"]

    # setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if regularization == "weight_decay":
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay = regularization_param)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)

    # perform training for each epoch, and print out performance accordingly
    for epoch in range(num_epochs):
        train_loss= train_epoch(model, train_loader, optimizer, criterion, input_size, regularization, epoch, num_epochs, device, regularization_param, print_freq = 100)
        train_acc, val_acc= evaluate_model(model, hyperparams, train_loader, validation_loader)
        logging.info("""train loss: {}, train acc: {}, val accuracy: {}""".format(train_loss, train_acc, val_acc))

    return model


# what happens inside training the model for an epoch
def train_epoch(model, train_loader, optimizer, criterion, input_size, regularization, epoch, num_epochs, device, regularization_param, print_freq = 100):
    total_step = len(train_loader)
    if regularization == "l1" or regularization == "weight_decay" or regularization == "dropout":
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if regularization == "l1":
                for param in model.parameters():
                    loss += regularization_param * torch.sum(torch.abs(param))

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i) % print_freq == 0:
                print("Epoch {}/{}, Step {}/{}, Loss: {:.4f}".format(epoch + 1, num_epochs, i, total_step, loss.item()))



    elif regularization == "fiedler":

        model.setLaplacian()
        fiedler_vec = from_numpy(model.getFiedlerVec())

        for i, (images, labels) in enumerate(train_loader):

            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            L = model.getLaplacian()

            ## the quadratic form computation below could be drastically sped up
            ## by taking into account the blockwise structure of the Laplacian in a feedforward neural network
            ## It would involve some for loop, a rough template is shown below
            ## since implementation-level optimization is not what we are interested in,
            ## we have adopted a vanilla quadratic form penalty here.
            regloss = regularization_param*torch.dot(fiedler_vec,torch.mv(L, fiedler_vec))
            loss += regloss

            # for name, param in model.named_parameters():
            #     if name == "fc1.weight":
            #         row, col = param.shape
            #         loss += regularization_param*"a quadratic form of the appropriate block in the Laplacian matrix "
            #     elif name == "fc2.weight":
            #         row, col = param.shape
            #         loss += regularization_param*"a quadratic form of the appropriate block in the Laplacian matrix "
            #     elif name == "fc3.weight":
            #         row, col = param.shape
            #          loss += regularization_param*"a quadratic form of the appropriate block in the Laplacian matrix "
            #     elif name == "fc4.weight":
            #         row, col = param.shape
            #         loss += regularization_param*"a quadratic form of the appropriate block in the Laplacian matrix "

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.setLaplacian()

            if i % print_freq == 0:
                print("Epoch {}/{}, Step {}/{}, Loss: {:.4f}".format(epoch + 1, num_epochs, i, total_step, loss.item()))

            if ((i)%100) == 0:
                fiedler_vec = from_numpy(model.getFiedlerVec())


    else:
        raise Exception("No regularization {} found".format(regularization))
    return loss.item()
