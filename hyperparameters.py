"""
Define the hyperparameters used via a dictionary and a getter
"""

def get_hyperparameters(dataset, regularization):
    hyperparameters = {
        "num_epochs": 10,
        "batch_size": 100,
        "learning_rate": 0.001,
        "hidden_size": 500
    }
    if dataset == "mnist":

        hyperparameters["input_size"] = 28*28
        hyperparameters["num_classes"] = 10

        if regularization == "dropout":
            hyperparameters["regularization_param"] = 0.4 #0.5
        elif regularization == "weight_decay":
            hyperparameters["regularization_param"] = 0.01
        elif regularization == "fiedler":
            hyperparameters["regularization_param"] = 0.01
        elif regularization == "l1":
            hyperparameters["regularization_param"] = 0.001
        else:
            raise Exception("Combination {}:{} does not exist".format(dataset, regularization))
    elif dataset == "cifar10":

        hyperparameters["input_size"] = 32*32*3
        hyperparameters["num_classes"] = 10

        if regularization == "dropout":
            hyperparameters["regularization_param"] = 0.5
        elif regularization == "weight_decay":
            hyperparameters["regularization_param"] = 0.01
        elif regularization == "fiedler":
            hyperparameters["regularization_param"] = 0.01
        elif regularization == "l1":
            hyperparameters["regularization_param"] = 0.001
        else:
            raise Exception("Combination {}:{} does not exist".format(dataset, regularization))

    elif dataset == "tcga":


        hyperparameters["hidden_size"] = 50
        hyperparameters["batch_size"] = 10
        hyperparameters["num_epochs"] = 5

        hyperparameters["input_size"] = 20531
        hyperparameters["num_classes"] = 5

        if regularization == "dropout":
            hyperparameters["regularization_param"] = 0.5
        elif regularization == "weight_decay":
            hyperparameters["regularization_param"] = 0.01
        elif regularization == "fiedler":
            hyperparameters["regularization_param"] = 0.01
        elif regularization == "l1":
            hyperparameters["regularization_param"] = 0.001
        else:
            raise Exception("Combination {}:{} does not exist".format(dataset, regularization))
    else:
        raise Exception("Combination {}:{} does not exist".format(dataset, regularization))


    return hyperparameters
