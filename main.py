import mlflow
import yaml
from download_data import download_and_prepare_data
from train import train_pipeline
from evaluate import evaluate_model

def main():
    with mlflow.start_run(run_name="Experiment 2 - Full Pipeline"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        mlflow.log_params({
            "learning_rate": config["training"]["lr"],
            "num_epochs": config["training"]["num_epochs"],
        })

        images_dir, train_df, val_df, test_df = download_and_prepare_data(config)
        model, _ = train_pipeline(config, images_dir, train_df, val_df)
        evaluate_model(model, images_dir, test_df, config)

if __name__ == "__main__":
    main()
