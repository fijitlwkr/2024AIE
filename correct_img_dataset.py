import os
import shutil

# 경로 설정
data_dir = "./cable_img"
default_txt = "./pascal_voc/ImageSets/Main/default.txt"
output_dir = "./Images"

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# default.txt에 나열된 파일 이름 가져오기
with open(default_txt, "r") as f:
    file_names = [line.strip() + ".jpg" for line in f.readlines()]  # .jpg 확장자 추가

# 이미지 파일 복사
for file_name in file_names:
    src_path = os.path.join(data_dir, file_name)
    if os.path.exists(src_path):
        shutil.copy(src_path, output_dir)
