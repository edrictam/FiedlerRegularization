import torch
## Here we evaluate the model
def evaluate_model(trained_model, hyperparams, train_loader, validation_loader):
    # check if gpu is available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    with torch.no_grad():
        correct = 0
        total = 0
        input_size = hyperparams["input_size"]
        for images, labels in validation_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        testing_acc = 100*correct/total


        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        training_acc = 100*correct/total

        return training_acc, testing_acc
