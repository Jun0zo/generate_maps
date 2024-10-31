import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import wandb

# Initialize wandb
wandb.init(project="PointNet_RNN_Diffusion")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, optimizer, criterion, num_epochs, device, patience=10):
    model.to(device)
    model.train()

    best_loss = float('inf')
    epochs_no_improve = 0  # Early stopping patience counter
    best_model_wts = None  # To save the best model weights

    for epoch in range(num_epochs):
        epoch_loss = 0
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, (point_cloud, imu_gps_seq) in enumerate(tepoch):
                point_cloud, imu_gps_seq = point_cloud.to(device), imu_gps_seq.to(device)
                
                # Forward pass
                output = model(point_cloud, imu_gps_seq)
                
                # Placeholder target for example purposes
                target = torch.zeros_like(output, device=device)
                
                # Compute loss
                loss = criterion(output, target)
                epoch_loss += loss.item()
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log loss to wandb for each batch
                wandb.log({"Batch Loss": loss.item()})
                
                # Update tqdm progress bar
                tepoch.set_postfix(loss=loss.item())

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(dataloader)
            wandb.log({"Epoch Loss": avg_epoch_loss})
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

            # Check for early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_no_improve = 0
                best_model_wts = model.state_dict()  # Save best model weights
            else:
                epochs_no_improve += 1

            # Early stopping condition
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} with best loss {best_loss:.4f}")
                break

    # Load best model weights before returning
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

def inference(model, point_cloud, imu_gps_seq, device):
    point_cloud, imu_gps_seq = point_cloud.to(device), imu_gps_seq.to(device)
    
    # 추론
    with torch.no_grad():
        generated_features = model(point_cloud, imu_gps_seq)
    
    return generated_features

# PointNet Backbone for Point Cloud Feature Extraction
class PointNetBackbone(nn.Module):
    def __init__(self):
        super(PointNetBackbone, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, 1)  # 입력 채널 4 (x, y, z, intensity)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]  # Global max pooling
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # 256-dimensional feature vector for point cloud

# RNN Backbone for GPS/IMU Feature Extraction
class RNNBackbone(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNBackbone, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        _, (hn, _) = self.rnn(x)
        x = self.fc(hn[-1])
        return x  # output_size-dimensional feature vector for GPS/IMU

# Simple Diffusion Model
class PointDiffusion(nn.Module):
    def __init__(self, feature_dim): 
        super(PointDiffusion, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, feature_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # Generates a feature vector similar to the input

# Combined Model
class GenerationModel(nn.Module):
    def __init__(self, point_dim, imu_gps_dim, hidden_size, output_feature_dim):
        super(GenerationModel, self).__init__()
        self.pointnet_backbone = PointNetBackbone()
        self.rnn_backbone = RNNBackbone(input_size=imu_gps_dim, hidden_size=hidden_size, output_size=128)
        self.diffusion_model = PointDiffusion(feature_dim=256 + 128)
        
    def forward(self, point_cloud, imu_gps_seq):
        # Extract point cloud features
        pc_features = self.pointnet_backbone(point_cloud)
        
        # Extract GPS/IMU features
        imu_gps_features = self.rnn_backbone(imu_gps_seq)
        
        # Concatenate features
        combined_features = torch.cat([pc_features, imu_gps_features], dim=1)
        
        # Generate output using the diffusion model
        generated_features = self.diffusion_model(combined_features)
        
        return generated_features

# Example usage
def main():
    # Hyperparameters
    batch_size = 8
    num_points = 1024
    num_epochs = 5
    learning_rate = 0.001

    # Initialize dummy dataloader (replace with actual data)
    dataloader = [
        (torch.randn(batch_size, 4, num_points), torch.randn(batch_size, 30, 6))
        for _ in range(100)
    ]

    # Initialize model, criterion, and optimizer
    model = GenerationModel(point_dim=4, imu_gps_dim=6, hidden_size=64, output_feature_dim=384)
    criterion = nn.MSELoss()  # Placeholder loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, dataloader, optimizer, criterion, num_epochs, device)

if __name__ == "__main__":
    main()
