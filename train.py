from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import wandb

from Class.CustonPCDataset import CustomPointCloudDataset, collate_fn
from Class.PointDiffusion import PointDiffusion, GenerationModel, train

def main():
    # Initialize wandb
    wandb.init(project="point_cloud_diffusion")

    # Hyperparameters and device setup
    config = wandb.config
    config.batch_size = 8
    config.num_points = 1024
    config.num_epochs = 10
    config.learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader setup
    point_cloud_dir = 'data/2011_09_28_drive_0047_sync/velodyne_points/data/'
    imu_gps_file = 'data/2011_09_28_drive_0047_sync/oxts/data/'
    dataset = CustomPointCloudDataset(point_cloud_dir, imu_gps_file)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    # Model, loss function, and optimizer setup
    model = GenerationModel(point_dim=4, imu_gps_dim=33, hidden_size=64, output_feature_dim=384)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Train the model with early stopping
    train(model, dataloader, optimizer, criterion, config.num_epochs, device, patience=10)

    # Save the model checkpoint and log to wandb
    model_path = "model_checkpoint.pth"
    torch.save(model.state_dict(), model_path)

    # Use wandb.log_artifact to upload the model
    artifact = wandb.Artifact('model_checkpoint', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()