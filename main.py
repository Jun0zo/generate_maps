import struct
import numpy as np


# 파일 경로
file_path = 'data/2011_09_28_drive_0047_sync/oxts/data/0000000000.txt'
# file_path = 'data/'

# 필요한 바이트 수에 맞게 data_format 수정
data_format = 'f' * 33  # 예: 33개의 float가 있는 경우
record_size = struct.calcsize(data_format)

data = []

with open(file_path, 'rb') as file:
    while True:
        chunk = file.read(record_size)
        if len(chunk) < record_size:  # 정확히 132 바이트인지 확인
            print("Incomplete chunk encountered, stopping read.")
            break
        record = struct.unpack(data_format, chunk)
        data.append(record)

print(data)