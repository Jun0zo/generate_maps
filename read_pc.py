import numpy as np
import os

def read_point_cloud(file_path):
    """
    포인트 클라우드 데이터를 읽어 분석하여 반환합니다.
    
    Parameters:
    - file_path (str): 포인트 클라우드 데이터가 저장된 .bin 파일 경로
    
    Returns:
    - points (list of dict): 각 포인트의 x, y, z 좌표와 intensity 값을 포함한 딕셔너리 리스트
    """
    # 포인트 클라우드 데이터 로드 (x, y, z, intensity 각 4바이트 float로 가정)
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

    # 데이터를 사람이 읽기 쉽게 딕셔너리 리스트로 변환
    points = []
    for point in point_cloud:
        x, y, z, intensity = point
        points.append({
            "x": x,           # X 좌표 (단위: m)
            "y": y,           # Y 좌표 (단위: m)
            "z": z,           # Z 좌표 (단위: m)
            "intensity": intensity  # 반사 강도 (0일 경우 강도 정보가 없을 수 있음)
        })
    
    return points

# read folder and return list of file paths
def read_folder(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


folder_list = read_folder('data/2011_09_28_drive_0047_sync/velodyne_points/data')
print('folder list:', folder_list)

for file_path in folder_list:
    point_data = read_point_cloud(file_path)
    for i, point in enumerate(point_data[:5]):
        print(f"Point {i+1}: {point}")
