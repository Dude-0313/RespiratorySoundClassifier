# Description :
# Date : 11/18/2023 (18)
# Author : Dude
# URLs :
#
# Problems / Solutions :
#
# Revisions :
#

import torch
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# import matplotlib
# matplotlib.use('TkAgg')

# Utility functions for plotting your results!
def plot_training_curves(
    training_curves, phases=["train", "val", "test"], metrics=["loss", "acc"]
):
    epochs = list(range(len(training_curves["train_loss"])))
    for metric in metrics:
        plt.figure()
        plt.title(f"Training curves - {metric}")
        for phase in phases:
            key = phase + "_" + metric
            if key in training_curves:
                if metric == "acc":
                    plt.plot(
                        epochs, [item.detach().cpu() for item in training_curves[key]]
                    )
                else:
                    plt.plot(epochs, training_curves[key])
        plt.xlabel("epoch")
        plt.legend(labels=phases)
        plt.show(block=True)


def classify_predictions(model, device, dataloader):
    model.eval()  # Set model to evaluate mode
    all_labels = torch.tensor([]).to(device)
    all_scores = torch.tensor([]).to(device)
    all_preds = torch.tensor([]).to(device)
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = torch.softmax(model(inputs), dim=1)
        _, preds = torch.max(outputs, 1)
        scores = outputs[:, 1]
        all_labels = torch.cat((all_labels, labels), 0)
        all_scores = torch.cat((all_scores, scores), 0)
        all_preds = torch.cat((all_preds, preds), 0)
    return (
        all_preds.detach().cpu(),
        all_labels.detach().cpu(),
        all_scores.detach().cpu(),
    )


def plot_cm(model, device, dataloaders, class_labels, phase="test"):
    class_labels = [1, 2, 3, 4, 5, 6, 7, 8]
    preds, labels, scores = classify_predictions(model, device, dataloaders[phase])

    cm = metrics.confusion_matrix(labels, preds)
    cr = metrics.classification_report(labels,preds)
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_labels
    )
    ax = disp.plot().ax_
    ax.set_title("Confusion Matrix -- counts")
    plt.show(block=True)
    return cr