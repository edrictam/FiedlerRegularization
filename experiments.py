"""
An experiment, in our context, only depend on two input instructions:

1. Dataset used (MNIST, CIFAR10, or TCGA)
2. Regularization method used (L1, Weight Decay, Fiedler or Dropout)
"""

from datasets import get_dataloaders
from hyperparameters import get_hyperparameters
from models import get_model
from train import train_model
from evaluation import evaluate_model
from logger import logToFile


# An experiment is run in 6 simple steps
# We load the appropriate dataset
# We load the appropriate hyperparameters
# We load the appropriate model
# We train the model
# We evaluate the model
# We log the results
def run_experiment(dataset, regularization):
    train_loader, validation_loader = get_dataloaders(dataset, batch_size = 100)
    hyperparams = get_hyperparameters(dataset, regularization)
    if regularization == "dropout":
        model = get_model("feedforward_dropout", hyperparams)
    elif regularization == "weight_decay" or regularization == "l1":
        model = get_model("feedforward", hyperparams)
    elif regularization == "fiedler":
        model = get_model("feedforward_fiedler", hyperparams)
    else:
        raise Exception("regularization {} not supported".format(regularization))

    trained_model = train_model(regularization, hyperparams, model, train_loader, validation_loader)
    train_acc, val_acc = evaluate_model(trained_model, hyperparams, train_loader, validation_loader)
    filename = "logs/{}_{}.txt".format(dataset, regularization)
    logToFile(filename, """training accuracy: {}, validation accuracy: {}\n""".format(train_acc, val_acc))
