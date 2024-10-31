import torch
from Class.CustonPCDataset import CustomPointCloudDataset, collate_fn
from Class.PointDiffusion import GenerationModel

def load_model(model_path, device):
    # 모델 초기화
    model = GenerationModel(point_dim=4, imu_gps_dim=33, hidden_size=64, output_feature_dim=384)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 평가 모드 설정
    return model

def inference(model, point_cloud, imu_gps_seq, device):
    # 모델과 데이터를 평가 모드로 설정 및 장치에 할당
    model.eval()
    point_cloud, imu_gps_seq = point_cloud.to(device), imu_gps_seq.to(device)
    
    # 추론 수행
    with torch.no_grad():  # 그라디언트 비활성화
        generated_features = model(point_cloud, imu_gps_seq)
    
    return generated_features

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 학습된 모델 경로
    model_path = "model_checkpoint.pth"
    model = load_model(model_path, device)

    # 새로운 데이터 예시 (배치 크기 1의 더미 데이터 사용)
    batch_size = 1
    num_points = 1024
    sequence_length = 30
    
    # 더미 데이터 생성 (추론할 실제 데이터를 사용해야 함)
    point_cloud = torch.randn(batch_size, 4, num_points)  # 포인트 클라우드 데이터
    imu_gps_seq = torch.randn(batch_size, sequence_length, 33)  # IMU/GPS 데이터
    
    # Inference 수행
    generated_features = inference(model, point_cloud, imu_gps_seq, device)
    print("Generated features:", generated_features)
    print("Generated features shape:", generated_features.shape)

if __name__ == "__main__":
    main()
