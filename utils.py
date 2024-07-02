import re
import pydicom
import cv2


# Method for identifying the actual amount of images 
def assign_image_numbers(curr_row, prev_row, current_number, current_index):
    if prev_row is None:
        curr_row['Number'] = current_number
        curr_row['Index'] = current_index
    elif prev_row['AbsPath'] == curr_row['AbsPath']:
        curr_row['Number'] = current_number
        current_index += 1
    else:
        current_number += 1
        current_index = 1
        curr_row['Number'] = current_number
    
    curr_row['Index'] = current_index
    prev_row = curr_row.copy()  # Update prev_row with the current row's state
        
    return curr_row, prev_row, current_number, current_index


def assign_labels_and_names(row):
    source = row['Source']
    path = row['AbsPath']
    finding = row['Type']
    
    parts = re.split(r"[/_.]", path)

    # Assign labels based on finding type
    labels = {
        'Architectural distortion': 1,
        'Mass': 2,
        'Calcification': 3
    }
    row['Label'] = labels.get(finding, 0)  # Default to 0 if no match

    # Assign ImageName based on source. Joins specific parts taken from the path of the image to form the name of the image. 
    image_name_mapping = {
        'CESM': lambda parts: '_'.join(parts[7:11]),
        'INbreast': lambda parts: '_'.join(parts[7:12]),
        'CBIS': lambda parts: '_'.join(parts[8:13]),
        'BMCD': lambda parts: '_'.join(parts[11:15]),
        'MIAS': lambda parts: parts[7],
        'VinDr': lambda parts: parts[8]
    }
    
    row['ImageName'] = image_name_mapping.get(source, lambda parts: 'Unknown')(parts)
    return row

def image_size(image_path):
    if image_path.lower().endswith(('.dcm', '.dicom', '.DCM')):
        dcm = pydicom.dcmread(image_path)
        pixel_data = dcm.pixel_array
        height, width = pixel_data.shape
    else:
        pixel_data = cv2.imread(image_path)
        height, width = pixel_data.shape[:2]
    return height, width

def resized_normalized_coordinates(row):
    xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    path = row['AbsPath']
    height, width = image_size(path)
    
    row['pixy'], row['pixx'] = height, width
    diff = abs((height - width) / 2)
    xmin += diff
    xmax += diff
    
    scale = 1080 / height
    xmin1080, ymin1080 = xmin * scale, ymin * scale
    xmax1080, ymax1080 = xmax * scale, ymax * scale

    row.update({
        'xmin1080': xmin1080,
        'ymin1080': ymin1080,
        'xmax1080': xmax1080,
        'ymax1080': ymax1080,
        'w1080': xmax1080 - xmin1080,
        'h1080': ymax1080 - ymin1080,
        'x1': xmin1080 / 1080,
        'y1': ymin1080 / 1080,
        'x2': xmax1080 / 1080,
        'y2': ymax1080 / 1080,
        'cx': (xmin1080 + (xmax1080 - xmin1080) / 2) / 1080,
        'cy': (ymin1080 + (ymax1080 - ymin1080) / 2) / 1080,
        'nw': (xmax1080 - xmin1080) / 1080,
        'nh': (ymax1080 - ymin1080) / 1080
    })

    return row
