import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from model import ResNet18
from utils import create_data_loader, train_model, device

def train_pipeline(config, images_dir, train_df, val_df):
    with mlflow.start_run(run_name="Experiment 2 - Stage 2: Model Training", nested=True):

        train_loader = create_data_loader(images_dir, train_df, config)
        val_loader = create_data_loader(images_dir, val_df, config)

        model = ResNet18(in_channels=3, num_classes=102).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config["training"]["lr"])

        best_model_path = train_model(
            model, train_loader, val_loader, loss_function, optimizer,
            num_epochs=config["training"]["num_epochs"],
            device=device
        )

    return model, best_model_path
