import os
import shutil
import random

# 경로 설정
images_folder = "./Images"
labels_folder = "./yolo_labels"
output_images_train = "./dataset/images/train"
output_images_val = "./dataset/images/val"
output_labels_train = "./dataset/labels/train"
output_labels_val = "./dataset/labels/val"

# Train/Validation 비율
split_ratio = 0.8

# 파일 리스트 가져오기
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(".jpg")])
label_files = sorted([f for f in os.listdir(labels_folder) if f.endswith(".txt")])

# 이미지와 라벨 매칭 확인
image_label_pairs = [(img, img.replace(".jpg", ".txt")) for img in image_files if img.replace(".jpg", ".txt") in label_files]

# Train/Validation 분할
random.shuffle(image_label_pairs)
split_index = int(len(image_label_pairs) * split_ratio)
train_pairs = image_label_pairs[:split_index]
val_pairs = image_label_pairs[split_index:]

# 출력 폴더 생성
os.makedirs(output_images_train, exist_ok=True)
os.makedirs(output_images_val, exist_ok=True)
os.makedirs(output_labels_train, exist_ok=True)
os.makedirs(output_labels_val, exist_ok=True)

# 파일 이동
for img, lbl in train_pairs:
    shutil.copy(os.path.join(images_folder, img), output_images_train)
    shutil.copy(os.path.join(labels_folder, lbl), output_labels_train)

for img, lbl in val_pairs:
    shutil.copy(os.path.join(images_folder, img), output_images_val)
    shutil.copy(os.path.join(labels_folder, lbl), output_labels_val)
    