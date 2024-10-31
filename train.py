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

    # Hyperparameters
    config = wandb.config
    config.batch_size = 8
    config.num_points = 1024
    config.num_epochs = 5
    config.learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 및 DataLoader 설정
    point_cloud_dir = 'data/2011_09_28_drive_0047_sync/velodyne_points/data/'
    imu_gps_file = 'data/2011_09_28_drive_0047_sync/oxts/data/'
    dataset = CustomPointCloudDataset(point_cloud_dir, imu_gps_file)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    # 모델 초기화
    model = GenerationModel(point_dim=4, imu_gps_dim=33, hidden_size=64, output_feature_dim=384)
    criterion = nn.MSELoss()  # Placeholder loss function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Start training and log with wandb
    train(model, dataloader, optimizer, criterion, config.num_epochs, device)

    # Evaluate and visualize a batch
    for point_cloud, imu_gps_seq in dataloader:
        output = model(point_cloud.to(device), imu_gps_seq.to(device))
        print("Generated features shape:", output.shape)
        break  # 첫 배치만 확인
    
    # Save the model to wandb
    torch.save(model.state_dict(), "model_checkpoint.pth")
    wandb.save("model_checkpoint.pth")

if __name__ == "__main__":
    main()