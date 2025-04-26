from utils import create_data_loader, test_model, device

def evaluate_model(model, images_dir, test_df, config):
    test_loader = create_data_loader(images_dir, test_df, config)
    test_model(model, test_loader, loss_fn=None, device=device)
