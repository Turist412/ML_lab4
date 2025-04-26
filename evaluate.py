import mlflow
from torch import nn

from utils import create_data_loader, test_model, device

def evaluate_model(model, images_dir, test_df, config):
    with mlflow.start_run(run_name="Experiment 2 - Stage 3: Model Evaluation", nested=True):

        test_loader = create_data_loader(images_dir, test_df, config)
        loss_function = nn.CrossEntropyLoss()
        test_model(model, test_loader, loss_function=loss_function, device=device)
