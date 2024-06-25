import cv2
import numpy as np
from PIL import Image
from skimage.transform import resize
from pydicom import dcmread

def reduce_poisson_noise(image):
    image_uint8 = image.astype(np.uint8)
    denoised_image = cv2.fastNlMeansDenoising(image_uint8, None, h=7)
    return denoised_image


# Adds pixels with a value of brightness = 0 to the longest side of the image to make it a square and avoid distortion when rescaling to the size YOLO asks for
def add_zeros_for_square (mg, height, width):
    max_dim = max(height, width)
    mg_image = Image.fromarray(mg) # convert from np array to PIL image
    pad = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
    offset = ((max_dim - mg_image.size[0]) // 2, (max_dim - mg_image.size[1]) // 2)
    pad.paste(mg_image, offset)
    square_image = np.array(pad) # convert back from PIL to np array 
    return (square_image)

def dicom_preprocessing (image_path, pixy, pixx):
        dicom = dcmread(image_path)
        image = dicom.pixel_array

        # Verify if metadata contains the following attributes
        if hasattr(dicom, 'WindowWidth'):
            window = dicom.WindowWidth
            level = dicom.WindowCenter
            photo_inter = dicom.PhotometricInterpretation
            model = dicom.ManufacturerModelName
            padding = dicom.PixelPaddingValue 

            if model == 'GIOTTO IMAGE 3DL' or model == 'GIOTTO CLASS':
                window = window[0]
                level = level[0]
                
            if photo_inter == 'MONOCHROME1':
                image[image==1] += padding
                level = np.max(image) - level
                image = np.max(image) - image 

            if model == 'GIOTTO IMAGE 3DL' or model == 'GIOTTO CLASS':
                if not isinstance(window, (list, tuple)):
                    window = [window]
                if not isinstance(level, (list, tuple)):
                    level = [level]
                window = window[0]
                level = level[0]
                
            size = (pixy, pixx)
            image = resize(image, output_shape=size, preserve_range=True).astype(np.float32)    
            
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
                min_value = np.min(image)
                max_value = np.max(image)
                image = ((image - min_value) / (max_value - min_value)) * (brightness_range[1] - brightness_range[0]) + brightness_range[0]
            else:
                image = np.max(image) - image  # Invert pixel values for MONOCHROME1
                min_value = np.min(image)
                max_value = np.max(image)
                image = ((image - min_value) / (max_value - min_value)) * (brightness_range[1] - brightness_range[0]) + brightness_range[0]
                

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
 
        # If grayscale, replicate into 3 channels
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        # Convert to PIL image
        image = Image.fromarray(image)
        image_np = np.array(image)
    
        return image_np