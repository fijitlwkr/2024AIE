import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from PIL import Image

# 경로 설정
annotations_path = "./CVAT for images 1.1_new labeling/annotations.xml"
images_dir = "./cable_img"
output_dir = "./processed_data"
yolo_labels_dir = os.path.join(output_dir, "yolo_labels")
cropped_images_dir = os.path.join(output_dir, "cropped_images")

# Train/Test 디렉토리 경로 설정
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
train_images_dir = os.path.join(train_dir, "images")
test_images_dir = os.path.join(test_dir, "images")
train_labels_dir = os.path.join(train_dir, "labels")
test_labels_dir = os.path.join(test_dir, "labels")

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)
os.makedirs(yolo_labels_dir, exist_ok=True)
os.makedirs(cropped_images_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# XML 파싱
tree = ET.parse(annotations_path)
root = tree.getroot()

# 데이터 수집
labeled_images = []
yolo_data = []
cropped_data = []

for image in root.findall('image'):
    image_name = image.get('name')
    image_path = os.path.join(images_dir, image_name)
    width = int(image.get('width'))
    height = int(image.get('height'))
    
    # 객체 태그 찾기
    polygons = image.findall('polygon')
    ellipses = image.findall('ellipse')
    objects = polygons + ellipses
    
    if not objects:  # 라벨링이 없는 이미지
        continue

    labeled_images.append(image_name)  # 라벨링된 이미지로 추가
    
    for obj in objects:
        label = obj.get('label')
        if obj.tag == "polygon":  # YOLO 형식 데이터 생성 (polygon)
            points = obj.get('points').split(";")
            x_coords = [float(point.split(",")[0]) for point in points]
            y_coords = [float(point.split(",")[1]) for point in points]
            
            xtl = min(x_coords)
            ytl = min(y_coords)
            xbr = max(x_coords)
            ybr = max(y_coords)
            
            x_center = (xtl + xbr) / 2 / width
            y_center = (ytl + ybr) / 2 / height
            box_width = (xbr - xtl) / width
            box_height = (ybr - ytl) / height
            
            yolo_data.append((image_name, label, x_center, y_center, box_width, box_height))
        
        elif label == "circle_point":  # YOLO 형식 데이터 생성 (ellipse)
            cx = float(obj.get('cx'))
            cy = float(obj.get('cy'))
            rx = float(obj.get('rx'))
            ry = float(obj.get('ry'))
            
            x_center = cx / width
            y_center = cy / height
            box_width = (2 * rx) / width
            box_height = (2 * ry) / height
            
            yolo_data.append((image_name, label, x_center, y_center, box_width, box_height))
        
        if label == "cable":  # 후처리용 데이터
            attributes = {attr.get('name'): attr.text for attr in obj.findall('attribute')}
            num_of_twist = int(attributes.get("num_of_twist", 0))
            num_of_cols = int(attributes.get("num_of_cols", 0))
            cropped_data.append((image_name, xtl, ytl, xbr, ybr, num_of_twist, num_of_cols))

# 라벨링 없는 이미지 삭제
for image_name in os.listdir(images_dir):
    if image_name not in labeled_images:
        os.remove(os.path.join(images_dir, image_name))

# YOLO 라벨 저장
for image_name, label, x_center, y_center, box_width, box_height in yolo_data:
    label_file = os.path.join(yolo_labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
    with open(label_file, "a") as f:
        class_id = {"cable": 0, "knitting_part": 1, "circle_point": 2}.get(label, -1)
        if class_id == -1:
            continue
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

# 후처리용 크롭 이미지 저장
for image_name, xtl, ytl, xbr, ybr, num_of_twist, num_of_cols in cropped_data:
    image_path = os.path.join(images_dir, image_name)
    with Image.open(image_path) as img:
        cropped_img = img.crop((xtl, ytl, xbr, ybr))
        cropped_img_dir = os.path.join(cropped_images_dir, os.path.splitext(image_name)[0])
        os.makedirs(cropped_img_dir, exist_ok=True)
        cropped_img.save(os.path.join(cropped_img_dir, f"twist_{num_of_twist}_cols_{num_of_cols}.jpg"))
        with open(os.path.join(cropped_img_dir, "attributes.txt"), "w") as f:
            f.write(f"num_of_twist: {num_of_twist}\n")
            f.write(f"num_of_cols: {num_of_cols}\n")

# Train/Test 데이터 분리
train_images, test_images = train_test_split(labeled_images, test_size=0.2, random_state=42)

# Train/Test 데이터 복사
for image_name in train_images:
    shutil.copy(os.path.join(images_dir, image_name), os.path.join(train_images_dir, image_name))
    label_file = os.path.join(yolo_labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
    if os.path.exists(label_file):
        shutil.copy(label_file, os.path.join(train_labels_dir, f"{os.path.splitext(image_name)[0]}.txt"))

for image_name in test_images:
    shutil.copy(os.path.join(images_dir, image_name), os.path.join(test_images_dir, image_name))
    label_file = os.path.join(yolo_labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
    if os.path.exists(label_file):
        shutil.copy(label_file, os.path.join(test_labels_dir, f"{os.path.splitext(image_name)[0]}.txt"))

print("데이터 전처리 및 train/test 분리 완료!")
