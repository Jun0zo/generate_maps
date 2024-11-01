import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import numpy as np

from .CustomPCDataset import CustomPointCloudDataset, collate_fn




class PointNetEncoder(nn.Module):
    def __init__(self, imu_gps_dim, latent_dim):
        super(PointNetEncoder, self).__init__()
        
        # PointNet layers for point cloud processing
        self.conv1 = nn.Conv1d(4, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Fully connected layers to combine with IMU/GPS data
        self.fc1 = nn.Linear(256 + imu_gps_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
 
    def forward(self, point_cloud, imu_gps_seq):
        # PointNet feature extraction
        x = F.relu(self.bn1(self.conv1(point_cloud)))  # (batch, 64, num_points)
        x = F.relu(self.bn2(self.conv2(x)))            # (batch, 128, num_points)
        x = F.relu(self.bn3(self.conv3(x)))            # (batch, 256, num_points)
        x = torch.max(x, 2, keepdim=False)[0]          # Global feature (batch, 256)
        
        # Concatenate IMU/GPS data
        imu_gps_seq = imu_gps_seq.reshape(x.size(0), -1)  # Flatten IMU/GPS data
        x = torch.cat([x, imu_gps_seq], dim=1)         # (batch, 256 + imu_gps_dim)
        
        # Fully connected layers
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class PointNetDecoder(nn.Module):
    def __init__(self, imu_gps_dim, latent_dim, num_points):
        super(PointNetDecoder, self).__init__()
        
        # Fully connected layers to combine latent space with IMU/GPS data
        self.fc1 = nn.Linear(latent_dim + imu_gps_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, num_points * 4)  # Output has the same shape as the input (x, y, z, intensity)
        self.num_points = num_points
        
    def forward(self, z, imu_gps_seq):
        # Concatenate latent vector with IMU/GPS data
        imu_gps_seq = imu_gps_seq.view(z.size(0), -1)
        x = torch.cat([z, imu_gps_seq], dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Reshape to original point cloud format
        x = x.view(-1, 4, self.num_points)  # (batch, 4, num_points)
        return x

class PointNetConditionalVAE(nn.Module):
    def __init__(self, imu_gps_dim, latent_dim, num_points):
        super(PointNetConditionalVAE, self).__init__()
        
        self.encoder = PointNetEncoder(imu_gps_dim, latent_dim)
        self.decoder = PointNetDecoder(imu_gps_dim, latent_dim, num_points)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, point_cloud, imu_gps_seq):
        mu, logvar = self.encoder(point_cloud, imu_gps_seq)
        z = self.reparameterize(mu, logvar)
        generated_point_cloud = self.decoder(z, imu_gps_seq)
        return generated_point_cloud, mu, logvar


# VAE 손실 함수 정의
def vae_loss(recon_point_cloud, point_cloud, mu, logvar, beta=0.1):
    # Reconstruction loss (MSE)
    BCE = F.mse_loss(recon_point_cloud, point_cloud, reduction="sum")
    
    # KL Divergence loss with small epsilon for stability
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp() + 1e-10)
    
    return BCE + beta * KLD

# Train 함수 수정
def train(model, dataloader, optimizer, num_epochs, device, patience=10, beta=0.1):
    model.to(device)
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, (point_cloud, imu_gps_seq) in enumerate(tepoch):
                
                # 데이터 NaN 체크
                if torch.isnan(point_cloud).any() or torch.isnan(imu_gps_seq).any():
                    print(f"NaN found in data at batch {batch_idx}. Skipping this batch.")
                    continue

                point_cloud = point_cloud.to(device)  # (batch, 4, num_points)
                imu_gps_seq = imu_gps_seq.to(device)
                
                # Forward pass
                generated_point_cloud, mu, logvar = model(point_cloud, imu_gps_seq)
                
                # 손실 함수 계산
                loss = vae_loss(generated_point_cloud, point_cloud, mu, logvar, beta)
                
                # NaN 체크 후 손실 확인
                if torch.isnan(loss).any():
                    print(f"NaN found in loss at batch {batch_idx}.")
                    continue

                epoch_loss += loss.item()
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log loss to wandb for each batch
                tepoch.set_postfix(loss=loss.item())

            # 평균 에포크 손실 계산
            avg_epoch_loss = epoch_loss / len(dataloader.dataset)
            wandb.log({"Epoch Loss": avg_epoch_loss}, step=epoch)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_no_improve = 0
                best_model_wts = model.state_dict()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} with best loss {best_loss:.4f}")
                break

    # Load best model weights before returning
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)      

def denormalize_point_cloud(point_cloud, pc_mean, pc_std, intensity_min, intensity_max):
    point_cloud[:, :3] = point_cloud[:, :3] * pc_std + pc_mean
    point_cloud[:, 3] = point_cloud[:, 3] * (intensity_max - intensity_min) + intensity_min
    return point_cloud

def denormalize_imu_gps(imu_gps_data, imu_gps_mean, imu_gps_std):
    return imu_gps_data * imu_gps_std + imu_gps_mean

def inference(model, dataloader, dataset, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        # 데이터 로드
        
        
        point_cloud, imu_gps_seq = next(iter(dataloader))
        point_cloud = point_cloud.to(device)
        imu_gps_seq = imu_gps_seq.to(device)
        
        print('sum :', point_cloud.abs().sum())
        
        print(f"Point cloud shape: {point_cloud.shape}, IMU/GPS shape: {imu_gps_seq.shape}")

        # 모델 추론
        generated_point_cloud, _, _ = model(point_cloud, imu_gps_seq)

        # 역정규화 적용
        generated_point_cloud = generated_point_cloud.cpu().numpy().reshape(-1, 4)
        generated_point_cloud = denormalize_point_cloud(
            generated_point_cloud,
            dataset.pc_mean,
            dataset.pc_std,
            dataset.intensity_min,
            dataset.intensity_max
        )

        print(f"Generated features shape after denormalization: {generated_point_cloud.shape}")
        print('sum :', np.abs(generated_point_cloud).sum())
    return generated_point_cloud


# Main 함수 수정
def main():
    # Initialize wandb
    wandb.init(project="point_cloud_vae")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(project="point_cloud_vae")

    # Hyperparameters and device setup
    config = wandb.config
    config.batch_size = 8
    config.num_points = 120000
    config.num_epochs = 100
    config.learning_rate = 0.001
    config.latent_dim = 60
    config.hidden_dim = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader setup
    point_cloud_dir = 'data/dst_zips/velodyne_point_all'
    imu_gps_file = 'data/dst_zips/oxt_all'
    
    # Dataset and DataLoader 설정
    dataset = CustomPointCloudDataset(point_cloud_dir, imu_gps_file)
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    # 모델 및 옵티마이저 설정
    model = PointNetConditionalVAE(imu_gps_dim=30, latent_dim=config.latent_dim, num_points=config.num_points)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 모델 학습
    train(model, dataloader, optimizer, config.num_epochs, device, patience=10, beta=0.1)

    # 모델 저장 및 wandb 로그
    model_path = "model_checkpoint.pth"
    torch.save(model.state_dict(), model_path)

    # Use wandb.log_artifact to upload the model
    artifact = wandb.Artifact('model_checkpoint', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()