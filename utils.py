import pandas as pd
import re
import pydicom
import cv2
import numpy as np


# Method for identifying the actual amount of images 
def assign_image_number (excel_file):
    df = pd.read_excel(excel_file)
    current = 1
    for i in range(0, len(df)):
        if i == 0:
            df.at[i, 'Number'] = current
            continue
        prev_row = df.iloc[i - 1]
        curr_row = df.iloc[i]
        
        if prev_row['AbsPath'] == curr_row['AbsPath']:
            df.at[i, 'Number'] = current
        else: 
            current += 1
            df.at[i, 'Number'] = current
            
    df.to_excel(excel_file, index= False)
    return df
    
# Method that extracts the name of the image from its path. It may need to be modified according to the structure of your directories storing the images 
def assign_labels_and_names (df, excel_file):
    
    for i, row in df.iterrows():
        source = row['Source']
        path = row['AbsPath']
        finding = row['Type']
        
        parts = re.split(r"[/_]", path)
        
        if finding == 'Architectural distortion':
            df.at[i, 'Label'] = 1
        elif finding == 'Mass':
            df.at[i, 'Label'] =2
        elif finding == 'Calcification':
            df.at[i, 'Label'] = 3
        
        if source == 'CESM':
            df.at[i, 'ImageName'] = parts[7]+'_'+parts[8]+'_'+ parts[9]+'_' + parts[10]
        elif source == 'INbreast':
            df.at[i, 'ImageName'] = parts[7] + '_' + parts [8] + '_' + parts[9] + '_' + parts[10] + '_' + parts[11]
        elif source == 'CBIS':
            df.at[i, 'ImageName'] = parts[8] + '_'+ parts[9] + '_' + parts[10]+ '_' + parts[11] + '_' + parts[12]
        elif source == 'BMCD':
            df.at[i, 'ImageName'] = parts[11] + '_' + parts[12]+ '_' + parts[13] + '_' + parts[14]
        elif source == 'MIAS':
            df.at[i, 'ImageName'] = parts[7]
        elif source == 'VinDr':
            df.at[i, 'ImageName'] = parts[8]      

            
    df.to_excel(excel_file, index = False)
    return df
    

# This method assigns an index starting from 1 to several bboxes corresponding to the same image    
def assign_finding_index (df, excel_file):
    
    current_index = 1
    
    for i in range(0, len(df)):
        prev_row = df.iloc[i - 1]
        curr_row = df.iloc[i]
        
        # Compare relevant columns
        if (prev_row['Number'] != curr_row['Number']):
            current_index = 1
        else:
            current_index += 1

        # Assign the index value
        df.at[i, "Index"] = current_index


    df.to_excel(excel_file, index=False)
    
    
# This computes the dimensions of the original image    
def image_size (image_path):

    if image_path.endswith ('.dcm') or image_path.endswith ('.dicom') or image_path.endswith('.DCM'): 
        dcm = pydicom.dcmread(image_path)
        pixel_data = dcm.pixel_array
        pixy, pixx = pixel_data.shape
    else:
        pixel_data = cv2.imread(image_path)
        pixy, pixx, channels = pixel_data.shape
        
    return pixy, pixx

# This writes on the file the bbox coordinates computed for a square 1080 x 1080 image
def resized_normalized_coordinates (excel_file):

    df = pd.read_excel(excel_file)

    for i, row in df.iterrows():
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        path = row['AbsPath']
        identity = row['id']

        pixy, pixx = image_size(path)
        
        df.at[i, 'pixy'] = pixy
        df.at[i, 'pixx'] = pixx

        diff = abs((pixy-pixx)/2)
        xmin += diff
        xmax += diff

        xmin1080 = (1080 * xmin)/pixy
        ymin1080 = (1080 * ymin)/pixy
        xmax1080 = (1080 * xmax)/pixy
        ymax1080 = (1080 * ymax)/pixy

        df.at[i, 'xmin1080'] = xmin1080
        df.at[i, 'ymin1080'] = ymin1080
        df.at[i, 'xmax1080'] = xmax1080
        df.at[i, 'ymax1080'] = ymax1080
        w1080 = xmax1080 - xmin1080
        df.at[i, 'w1080'] = w1080
        h1080 = ymax1080 - ymin1080
        df.at[i, 'h1080'] = h1080

        # Both YOLO and COCO requiere normalized coordinated, therefore, we divide by the maximum dimension
        x1 = xmin1080 / 1080
        y1 = ymin1080 / 1080
        x2 = xmax1080 / 1080
        y2 = ymax1080 / 1080
        cx = (xmax1080 - (w1080 / 2))/1080 # this is the X coodinate for the center of the bbox
        cy = (ymax1080 - (h1080 / 2))/1080 # this is the Y coordinate for the center of the bbox
        nw = w1080 / 1080
        nh = w1080 / 1080

        df.at[i, 'x1'] = x1
        df.at[i, 'y1'] = y1
        df.at[i, 'x2'] = x2
        df.at[i, 'y2'] = y2
        df.at[i, 'cy'] = cy
        df.at[i, 'cx'] = cx
        df.at[i, 'nw'] = nw
        df.at[i, 'nh'] = nh

        print(identity)

    df.to_excel(excel_file, index=False)


# So the final dataset is randomly split into the train, valid and test categories
def assign_dl_folder (excel_file, ratio):
    df = pd.read_excel(excel_file)
    #Just to be sure...
    assert len(ratio) == 3, "Ratio must be a tuple of three elements"
    assert sum(ratio) <= 1, "Sum of ratio must be less than or equal to 1"
    
    categories = ['train', 'valid', 'test']
    
    df['set'] = np.random.choice(categories, size=len(df), p=ratio)
    
    df.to_excel(excel_file, index =False)
    
