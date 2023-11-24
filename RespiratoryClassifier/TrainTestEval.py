# Description :
# Date : 11/18/2023 (18)
# Author : Dude
# URLs :
#
# Problems / Solutions :
#
# Revisions :
#
import time, copy
import torch


def train_classification_model(
    device,
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25,
):
    since = time.time()

    best_model_wts = copy.deepcopy(
        model.state_dict()
    )  # keep the best weights stored separately
    best_acc = 0.0
    best_epoch = 0

    # Each epoch has a training, validation, and test phase
    phases = ["train", "val", "test"]

    # Keep track of how loss and accuracy evolves during training
    training_curves = {}
    for phase in phases:
        training_curves[phase + "_loss"] = []
        training_curves[phase + "_acc"] = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in phases:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # No need to flatten the inputs!
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    labels = labels.squeeze()
                    # loss = criterion(outputs, labels)
                    one_label = torch.nn.functional.one_hot(
                        labels, num_classes=8
                    ).float()
                    loss = criterion(outputs, one_label)

                    # backward + update weights only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            training_curves[phase + "_loss"].append(epoch_loss)
            training_curves[phase + "_acc"].append(epoch_acc)

            print(f"{phase:5} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model if it's the best accuracy (bas
            if phase == "val" and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f} at epoch {best_epoch}")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, training_curves


def train_densenet_model(
    device,
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25,
):
    since = time.time()

    best_model_wts = copy.deepcopy(
        model.state_dict()
    )  # keep the best weights stored separately
    best_acc = 0.0
    best_epoch = 0

    # Each epoch has a training, validation, and test phase
    phases = ["train", "val", "test"]

    # Keep track of how loss and accuracy evolves during training
    training_curves = {}
    for phase in phases:
        training_curves[phase + "_loss"] = []
        training_curves[phase + "_acc"] = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in phases:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + update weights only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            training_curves[phase + "_loss"].append(epoch_loss)
            training_curves[phase + "_acc"].append(epoch_acc)

            print(f"{phase:5} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model if it's the best accuracy (bas
            if phase == "val" and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f} at epoch {best_epoch}")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, training_curves
