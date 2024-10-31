from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm

from Class.CustonPCDataset import CustomPointCloudDataset, collate_fn
from Class.PointDiffusion import PointDiffusion, GenerationModel, train

def main():
    batch_size = 8
    num_points = 1024
    num_epochs = 5
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 및 DataLoader 설정
    point_cloud_dir = 'data/2011_09_28_drive_0047_sync/velodyne_points/data/'
    imu_gps_file = 'data/2011_09_28_drive_0047_sync/oxts/data/'
    dataset = CustomPointCloudDataset(point_cloud_dir, imu_gps_file)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # 모델 초기화
    model = GenerationModel(point_dim=4, imu_gps_dim=33, hidden_size=64, output_feature_dim=384)
    criterion = nn.MSELoss()  # Placeholder loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, dataloader, optimizer, criterion, num_epochs, device)


    # 데이터 반복
    for point_cloud, imu_gps_seq in dataloader:
        output = model(point_cloud, imu_gps_seq)
        print("Generated features shape:", output.shape)
        break  # 첫 배치만 확인

if __name__ == "__main__":
    main()