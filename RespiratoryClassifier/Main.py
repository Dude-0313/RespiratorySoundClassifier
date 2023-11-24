# Description :
# Date : 11/18/2023 (18)
# Author : Dude
# URLs :
#
# Problems / Solutions :
#
# Revisions :
#
import random

import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader

from RespiratoryClassifier.PlotTrainingCurves import plot_training_curves, plot_cm
from RespiratoryClassifier.RespiratoryClassifierModel import (
    RespiratoryClassifierModel,
    RespiratoryClassifierDenseNet,
)
from RespiratoryClassifier.RespiratoryDataset import MendeleyLungSounds
from RespiratoryClassifier.TrainTestEval import (
    train_classification_model,
    train_densenet_model,
)

DATA_PATH = "C:\\Kuljeet\\WorkSpace\\PyTorch\\RespiratoryClassifier\\lung_sounds"
WAV_PATH = "Audio_Files"
ANNOTATIONS_FILE = "Data annotation.csv"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1


def LoadDatasets():
    resp_dataset = MendeleyLungSounds(DATA_PATH, WAV_PATH, ANNOTATIONS_FILE)
    input_shape = resp_dataset.get_shape()
    data_classes = resp_dataset.get_classes()
    print(data_classes.keys())
    # input_shape = (3, 128, 235)
    # data_classes = 8
    train_size = int(TRAIN_SPLIT * len(resp_dataset))
    val_size = int(VAL_SPLIT * len(resp_dataset))
    test_size = len(resp_dataset) - val_size - train_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        resp_dataset, [train_size, val_size, test_size]
    )
    data_loaders = {
        "train": DataLoader(dataset=train_dataset, batch_size=8, shuffle=True),
        "val": DataLoader(dataset=val_dataset, batch_size=8, shuffle=True),
        "test": DataLoader(dataset=test_dataset, batch_size=8, shuffle=True),
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }
    return data_loaders, dataset_sizes, input_shape, data_classes


# random split
SEED = random.randint(1, 100)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# parameters
learning_rate = 0.01
num_epochs = 50

\

#
model = RespiratoryClassifierModel(
    input_shape=input_shape, num_classes=len(data_classes)
).to(device)

# parameters
learning_rate = 0.05
num_epochs = 50

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# # Make sure you save the training curves along the way for visualization afterwards!
model, training_curves = train_classification_model(
    device,
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs=num_epochs,
)

# plot training curves
plot_training_curves(training_curves, phases=["train", "val", "test"])

# plot confusion matrix
rep = plot_cm(model, device, dataloaders, data_classes.keys(), phase="test")

# print classification report
print('============== Classification Report ==============')
print(rep)
print('==================================================')

model = RespiratoryClassifierDenseNet(
    input_shape=input_shape, num_classes=len(data_classes)
).to(device)

# parameters
learning_rate = 0.01
num_epochs = 50

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# # Make sure you save the training curves along the way for visualization afterwards!
model, training_curves = train_classification_model(
    device,
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs=num_epochs,
)

model.train_all()
# # Make sure you save the training curves along the way for visualization afterwards!
model, training_curves = train_classification_model(
    device,
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs=num_epochs,
)

# Make sure you save the training curves along the way for visualization afterwards!
# model, training_curves = train_densenet_model(
#     device,
#     model,
#     dataloaders,
#     dataset_sizes,
#     criterion,
#     optimizer,
#     scheduler,
#     num_epochs=num_epochs,
# )

# plot training

# plot training curves
plot_training_curves(training_curves, phases=["train", "val", "test"])

# plot confusion matrix
rep = plot_cm(model, device, dataloaders, data_classes.keys(), phase="test")
# print classification report
print('============== Classification Report ==============')
print(rep)
print('==================================================')
