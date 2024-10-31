import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct



class CustomPointCloudDataset(Dataset):
    def __init__(self, point_cloud_dir, imu_gps_dir):
        """
        포인트 클라우드와 GPS/IMU 데이터를 읽고 처리하는 Dataset 클래스.
        
        Parameters:
        - point_cloud_dir (str): 포인트 클라우드 데이터 (.bin 파일들)가 저장된 폴더 경로
        - imu_gps_dir (str): GPS 및 IMU 데이터가 개별 파일로 저장된 폴더 경로
        """
        self.point_cloud_dir = point_cloud_dir
        self.point_cloud_files = sorted(os.listdir(point_cloud_dir))
        self.imu_gps_dir = imu_gps_dir
        self.imu_gps_files = sorted(os.listdir(imu_gps_dir))
        
        # 포인트 클라우드 파일 수와 GPS/IMU 파일 수가 일치하는지 확인
        print(len(self.point_cloud_files), len(self.imu_gps_files))
        assert len(self.point_cloud_files) == len(self.imu_gps_files), \
            "The number of point cloud files and GPS/IMU files should be the same."
    
    def __len__(self):
        return len(self.point_cloud_files)
    
    def load_point_cloud(self, file_path):
        """ 포인트 클라우드 데이터를 .bin 파일에서 로드 """
        point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)  # (x, y, z, intensity)
        # point_cloud = np.fromfile(file_path, dtype=np.float32)
        return point_cloud

    def load_imu_gps(self, file_path):
        """
        GPS 및 IMU 데이터를 이진 파일에서 로드
        """
        data_format = 'f' * 33  # 예: 33개의 float로 구성된 포맷
        record_size = struct.calcsize(data_format)
        imu_gps_data = []
        
        with open(file_path, 'rb') as file:
            while True:
                chunk = file.read(record_size)
                if len(chunk) < record_size:
                    break
                record = struct.unpack(data_format, chunk)
                imu_gps_data.append(record)

        return np.array(imu_gps_data, dtype=np.float32)  # 배열로 변환하여 반환

    def __getitem__(self, idx):
        # 포인트 클라우드 파일 경로 설정 및 데이터 로드
        point_cloud_path = os.path.join(self.point_cloud_dir, self.point_cloud_files[idx])
        point_cloud = self.load_point_cloud(point_cloud_path)
        
        # GPS/IMU 파일 경로 설정 및 데이터 로드
        imu_gps_path = os.path.join(self.imu_gps_dir, self.imu_gps_files[idx])
        imu_gps_seq = self.load_imu_gps(imu_gps_path)  # 각 항목이 30 타임스텝이라고 가정

        # 텐서 변환
        point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float32).permute(0, 1)  # (4, num_points)
        imu_gps_tensor = torch.tensor(imu_gps_seq, dtype=torch.float32)  # (30, 6)
        
        return point_cloud_tensor, imu_gps_tensor
    

def collate_fn(batch, max_allowed_points=120000):
    """ 각 배치의 포인트 클라우드를 패딩 또는 샘플링하여 고정된 크기로 맞추는 collate_fn """
    point_clouds, imu_gps_data = zip(*batch)

    max_num_points = min(max(pc.shape[0] for pc in point_clouds), max_allowed_points)
    padded_point_clouds = []
    for pc in point_clouds:
        num_points = pc.shape[0]
        if num_points > max_num_points:
            indices = torch.randperm(num_points)[:max_num_points]
            pc = pc[indices]
        elif num_points < max_num_points:
            padding = torch.zeros((max_num_points - num_points, 4), dtype=torch.float32)
            pc = torch.cat([pc, padding], dim=0)
        padded_point_clouds.append(pc)
    
    padded_point_clouds = torch.stack(padded_point_clouds)  # (batch_size, max_num_points, 4)
    imu_gps_data = torch.stack(imu_gps_data)  # (batch_size, 30, 6)
    
    # Permute the point cloud tensor for Conv1d input
    padded_point_clouds = padded_point_clouds.permute(0, 2, 1)  # (batch_size, 4, max_num_points)
    
    return padded_point_clouds, imu_gps_data
