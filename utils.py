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


def assign_labels_and_names (row):
    source = row['Source']
    path = row['AbsPath']
    finding = row['Type']
    
    parts = re.split(r"[/_]", path)
    
    # Assign labels based on finding type
    if finding == 'Architectural distortion':
        row['Label'] = 1
    elif finding == 'Mass':
        row['Label'] = 2
    elif finding == 'Calcification':
        row['Label'] = 3
       
    # Assign ImageName based on source
    if source == 'CESM':
        row['ImageName'] = parts[7] + '_' + parts[8] + '_' + parts[9] + '_' + parts[10]
    elif source == 'INbreast':
        row['ImageName'] = parts[7] + '_' + parts[8] + '_' + parts[9] + '_' + parts[10] + '_' + parts[11]
    elif source == 'CBIS':
        row['ImageName'] = parts[8] + '_' + parts[9] + '_' + parts[10] + '_' + parts[11] + '_' + parts[12]
    elif source == 'BMCD':
        row['ImageName'] = parts[11] + '_' + parts[12] + '_' + parts[13] + '_' + parts[14]
    elif source == 'MIAS':
        row['ImageName'] = parts[7]
    elif source == 'VinDr':
        row['ImageName'] = parts[8]
        
    return row 
    
def image_size (image_path):

    if image_path.endswith ('.dcm') or image_path.endswith ('.dicom') or image_path.endswith('.DCM'): 
        dcm = pydicom.dcmread(image_path)
        pixel_data = dcm.pixel_array
        pixy, pixx = pixel_data.shape
    else:
        pixel_data = cv2.imread(image_path)
        pixy, pixx, channels = pixel_data.shape
        
    return pixy, pixx

def resized_normalized_coordinates (row):
    
    xmin = row['xmin']
    ymin = row['ymin']
    xmax = row['xmax']
    ymax = row['ymax']
    path = row['AbsPath']
    

    pixy, pixx = image_size(path)
        
    row['pixy'] = pixy
    row['pixx'] = pixx

    diff = abs((pixy-pixx)/2)
    xmin += diff
    xmax += diff

    xmin1080 = (1080 * xmin)/pixy
    ymin1080 = (1080 * ymin)/pixy
    xmax1080 = (1080 * xmax)/pixy
    ymax1080 = (1080 * ymax)/pixy

    row['xmin1080'] = xmin1080
    row['ymin1080'] = ymin1080
    row['xmax1080'] = xmax1080
    row['ymax1080'] = ymax1080
    w1080 = xmax1080 - xmin1080
    row['w1080'] = w1080
    h1080 = ymax1080 - ymin1080
    row['h1080'] = h1080

    # Both YOLO and COCO require normalized coordinates, therefore, we divide by the maximum dimension
    x1 = xmin1080 / 1080
    y1 = ymin1080 / 1080
    x2 = xmax1080 / 1080
    y2 = ymax1080 / 1080
    cx = (xmax1080 - (w1080 / 2))/1080 # this is the X coordinate for the center of the bbox
    cy = (ymax1080 - (h1080 / 2))/1080 # this is the Y coordinate for the center of the bbox
    nw = w1080 / 1080
    nh = w1080 / 1080

    row['x1'] = x1
    row['y1'] = y1
    row['x2'] = x2
    row['y2'] = y2
    row['cy'] = cy
    row['cx'] = cx
    row['nw'] = nw
    row['nh'] = nh

    return row
    

    