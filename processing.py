import cv2
import numpy as np
import cupy as cp
from PIL import Image
import pydicom

# Adds pixels with a value of brightness = 0 to the longest side of the image to make it a square and avoid distortion when rescaling to the size YOLO asks for
def add_zeros_for_square(image, height, width):
    max_dim = max(height, width)
    # Convert the image to a CuPy array
    image_gpu = cp.asarray(image)
    
    # Create a new array with zeros (black background)
    pad_gpu = cp.zeros((max_dim, max_dim, image_gpu.shape[2]), dtype=image_gpu.dtype)
    
    # Calculate the offset
    offset_y = (max_dim - height) // 2
    offset_x = (max_dim - width) // 2
    
    # Place the original image in the center
    pad_gpu[offset_y:offset_y + height, offset_x:offset_x + width] = image_gpu
    
    # Convert back to numpy array if needed
    square_image = cp.asnumpy(pad_gpu)
    
    return square_image

def dicom_preprocessing (image_path):
    dicom = pydicom.dcmread(image_path)
    image = dicom.pixel_array
    image = cp.asarray(image, dtype=cp.float32)   

    # Verify if metadata contains the following attributes
    if hasattr(dicom, 'WindowWidth'):
        #Check for these attributes within the metadata
        window = dicom.WindowWidth
        level = dicom.WindowCenter
        photo_inter = dicom.PhotometricInterpretation
        model = dicom.ManufacturerModelName
        padding = getattr(dicom, 'PixelPaddingValue', 0)

        if isinstance(window, pydicom.multival.MultiValue):
            window = float(window[0])
        else:
            window = float(window)
            
        if isinstance(level, pydicom.multival.MultiValue):
            level = float(level[0])
        else:
            level = float(level)
    
        if photo_inter == 'MONOCHROME1':
            image[image==1] += padding
            level = cp.max(image) - level
            image = cp.max(image) - image 
            
        image = image.astype(cp.float32)  
            
        # Normalize pixel intensities, and convert to 8-bit
        image -= (level - window/2)
        image /= window
        image[image<0] = 0
        image[image>1] = 1
        image *= 255
            
    else:
        brightness_range = (0,255)
        photo_inter = dicom.get("PhotometricInterpretation", "Unknown")
        if photo_inter == "MONOCHROME2":
            min_value = cp.min(image)
            max_value = cp.max(image)
            image = ((image - min_value) / (max_value - min_value)) * (brightness_range[1] - brightness_range[0]) + brightness_range[0]
        else:
            image = cp.max(image) - image  # Invert pixel values for MONOCHROME1
            min_value = cp.min(image)
            max_value = cp.max(image)
            image = ((image - min_value) / (max_value - min_value)) * (brightness_range[1] - brightness_range[0]) + brightness_range[0]
                
    if image.dtype != cp.uint8:
        image = image.astype(cp.uint8)

    # If the image is grayscale, replicate it into 3 channels
    if len(image.shape) == 2:
        image = cp.stack((image,) * 3, axis=-1)
    
    return image
    
    
    
        