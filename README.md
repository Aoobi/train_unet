# Unet training and inference repo

Repository to keep a working version of Unet training as well as inference and post-processing including analyzing results based on IEC standards.

## Requirements and installation
- Python 3.8

- Install packages using provided requirements.txt file
```bash
pip install -r requirements.txt
```

## Usage

### Training 
TBD

### Inference

#### Batch inference on a folder
Use [main_folder_inference.py](main_folder_inference.py) and change the path to the folder containing V&G images.
Standard operation involves locating fiber center and cuting the image to 768x768 - if images are already cut to required size, set `full_size = False`. 

Chose desired IEC standard, all available standards are in the **IEC_standards** folder. Set standard name in the `spec_path` variable.

```python
from yolov8_locate_fiber_and_cut import locate_cut_main
from unet_inference import unet_inference_folder
from IEC_watershed_welz_circle import analyze_to_IEC

if __name__ == '__main__':

    #select proper spec you want to evaluate against
    spec_path = r"IEC_standards/spec_IEC_ed2_SMPC_RL26dB.json"

    full_size = True # False if your images are already 768x768
    images_path = "path_to_your_folder"
    if full_size:
        locate_cut_main(images_path)

    unet_inference_folder(images_path)

    analyze_to_IEC(images_path, images_path, spec_path)

```
`locate_cut_main` currently overwrites the full size images in the folder 

If `locate_cut_main` does not find a fiber in the image it will move the image to a new subfolder **fiber_not_detected** that will not be analyzed.

`unet_inference_folder` will load the cut images and for each image save a *mask* with name **imageName_mask.png**.

`analyze_to_IEC` loads **masks** and **original images** and returns one **.png** file with marked detections and one **.txt** file with counts of detections based on selected IEC standard.

Full-size image | Cropped image | Mask 
--- | --- | --- 
![uncut](/assets/12011223_1113710-stacked_full.png) | ![cut](/assets/12011223_1113710-stacked.png) | ![mask](/assets/12011223_1113710-stacked.png_mask.png) 

Result with marked detections (IEC) |
--- |
![marked](/assets/12011223_1113710-stacked.png_inference_vgg.png) |

### Task list
- [ ] Modify behaviour of `locate_cut_main` to not overwrite images, but create cut images in new folder instead
- [ ] Add single image inference
- [x] Improve IEC defect count output formating 
- [x] Add "Pass/Fail" information to the IEC output file (currently analyzed in code but not reported) 
