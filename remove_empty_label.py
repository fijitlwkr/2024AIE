import os

# 경로 설정
labels_train_dir = "./dataset/labels/train"
labels_val_dir = "./dataset/labels/val"
images_train_dir = "./dataset/images/train"
images_val_dir = "./dataset/images/val"

def clean_empty_labels(labels_dir, images_dir):
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        
        # 텍스트 파일이 비어 있는지 확인
        if os.path.isfile(label_path) and os.stat(label_path).st_size == 0:
            print(f"Removing empty label file: {label_path}")
            
            # 해당 라벨 파일과 연결된 이미지 삭제
            corresponding_image = os.path.join(images_dir, label_file.replace(".txt", ".jpg"))
            if os.path.exists(corresponding_image):
                print(f"Removing corresponding image: {corresponding_image}")
                os.remove(corresponding_image)
            
            # 라벨 파일 삭제
            os.remove(label_path)

# Train과 Val 데이터에서 빈 라벨 제거
clean_empty_labels(labels_train_dir, images_train_dir)
clean_empty_labels(labels_val_dir, images_val_dir)
