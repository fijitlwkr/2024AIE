import os
import xml.etree.ElementTree as ET

def voc_to_yolo(single_xml_file, output_folder, class_list):
    """
    Convert annotations in a single XML file to YOLO format (image-wise).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parse the single XML file
    tree = ET.parse(single_xml_file)
    root = tree.getroot()

    # Process each image in the XML
    for image in root.findall("image"):
        image_name = image.get("name")
        img_width = int(image.get("width"))
        img_height = int(image.get("height"))

        # Create a YOLO label file for the current image
        yolo_file = os.path.join(output_folder, image_name.replace(".jpg", ".txt"))
        with open(yolo_file, "w") as f:
            # Process ellipse annotations
            for ellipse in image.findall("ellipse"):
                class_name = ellipse.get("label")
                if class_name not in class_list:
                    continue
                class_id = class_list.index(class_name)

                # Ellipse center and radii
                cx = float(ellipse.get("cx"))
                cy = float(ellipse.get("cy"))
                rx = float(ellipse.get("rx"))
                ry = float(ellipse.get("ry"))

                # Convert ellipse to YOLO format
                x_center = cx / img_width
                y_center = cy / img_height
                width = (2 * rx) / img_width
                height = (2 * ry) / img_height

                # Write YOLO format
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

            # Process polygon annotations
            for polygon in image.findall("polygon"):
                class_name = polygon.get("label")
                if class_name not in class_list:
                    continue
                class_id = class_list.index(class_name)

                # Get polygon points
                points = polygon.get("points").split(";")
                x_coords = [float(pt.split(",")[0]) for pt in points]
                y_coords = [float(pt.split(",")[1]) for pt in points]

                # Compute bounding box around the polygon
                xmin = min(x_coords)
                ymin = min(y_coords)
                xmax = max(x_coords)
                ymax = max(y_coords)

                # Convert to YOLO format
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                # Write YOLO format
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Parameters
single_xml_file = "./cvat_project/annotations.xml"  # Path to the single XML file
output_folder = "./yolo_labels"                       # Output folder for YOLO labels
class_list = ["circle_point", "cable", "knitting_part"]  # Define your class list

# Execute the conversion
voc_to_yolo(single_xml_file, output_folder, class_list)
