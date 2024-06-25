# DetectorDatasetMG
ImageMed YOLO-NAS is a tool specialized in medical image classification and analysis. Using advanced YOLO and NAS algorithms

To generate a:
* Rescaled to 1080 x 1080
* Normalized 
* Denoised 

grayscale/DICOM image dataset, please run the 3classdataset notebook. The algorithm can be tested before by running the testing.ipynb. 
The sample coordinates provided within the Templatedetfile.xlsx correspond to the abnormalities found on mammography images taken from publicly available datasets that can be downloaded here:

* MIAS: https://www.kaggle.com/datasets/kmader/mias-mammography (please note that it is preferable if you convert these images from .pgm to another format such as .dcm or .png)
* CBIS-DDSM https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
* INbreast https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset
* CDD-CESM https://doi.org/10.7937/29kw-ae92
* VinDr-Mammo https://doi.org/10.13026/br2v-7517
* BMCD https://doi.org/10.5281/zenodo.7969411




