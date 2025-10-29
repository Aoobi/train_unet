import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
import cv2
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from segmentation_models.utils import set_trainable

import albumentations as A

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  print("GPU not found")
  pass

class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ["non-linear defect", "linear defect", 'bgr']
    # CLASSES = ['defect', 'bgr']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
            multiplier=1
    ):
        self.ids = glob.glob(os.path.join(images_dir, "*.png"))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.multiplier = multiplier

    # def __getitem__(self, i):
    #
    #     # read data
    #     image = cv2.imread(self.images_fps[i])
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     mask = cv2.imread(self.masks_fps[i], 0)
    #
    #     # extract certain classes from mask (e.g. cars)
    #     masks = [(mask == v) for v in self.class_values]
    #     mask = np.stack(masks, axis=-1).astype('float')
    #
    #     # add background if mask is not binary
    #     if mask.shape[-1] != 1:
    #         background = 1 - mask.sum(axis=-1, keepdims=True)
    #         mask = np.concatenate((mask, background), axis=-1)
    #
    #     # apply augmentations
    #     if self.augmentation:
    #         sample = self.augmentation(image=image, mask=mask)
    #         image, mask = sample['image'], sample['mask']
    #
    #     # apply preprocessing
    #     if self.preprocessing:
    #         sample = self.preprocessing(image=image, mask=mask)
    #         image, mask = sample['image'], sample['mask']
    #
    #     return image, mask
    #
    # def __len__(self):
    #     return len(self.ids)

    def __getitem__(self, i):

        # determine the actual index and augmentation instance
        index = i // self.multiplier
        augment_instance = i % self.multiplier

        # read data
        image = cv2.imread(self.images_fps[index], 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[index], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations if it's an augmented instance
        if self.augmentation and augment_instance >= 0:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids) * self.multiplier
def get_validation_augmentation(imsize):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [A.PadIfNeeded(min_height=768, min_width=imsize, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0)]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

def unet_inference_folder(DATA_DIR, model_path=r"models/recompiled_BestModel_19_12_2023_8GPU_TRAIN_SF_VALID_SF_Frozen_continue_last_epochs_unfrozen_LR_2e-5_vgg19.h5", imsize=768):
    model1 = keras.models.load_model(model_path)
    CLASSES = ["non-linear defect", "linear defect"]
    BACKBONE = 'vgg19'
    # DATA_DIR = r"C:\Users\pawlowskj\Downloads\1_12_V&G_Euclid\1_12_V&G_Euclid\testing"

    preprocess_input = sm.get_preprocessing(BACKBONE)
    test_dataset = Dataset(
        DATA_DIR,
        DATA_DIR,
        classes=CLASSES,
        augmentation=get_validation_augmentation(imsize),
        preprocessing=get_preprocessing(preprocess_input),
    )
    n = len(test_dataset)
    ids = np.arange(len(test_dataset))

    save_mask = True
    import tqdm
    for i in tqdm.trange(len(ids)):
        image0, gt_mask = test_dataset[i]
        # image, gt_mask = test_dataset[i+1]
        # image = cv2.imread(r"E:\8_praca_magisterska\mtp_ferrule_images\labeled_images\19_04_2023_bores_seg_multiclass\val\images\bore_sm_12_158c_2.png", cv2.IMREAD_COLOR).astype("float32")
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
        import time
        startt = time.time()
        image = image0.copy() # not needed anymore:  - [np.average(image0[:,:, 0]), np.average(image0[:,:, 1]), np.average(image0[:,:, 2])]
        image = np.expand_dims(image, axis=0)
        pr_mask = model1.predict(image)
        final1 = pr_mask.squeeze().copy()
        # pr_mask1 = model.predict(image)
        # final11 = pr_mask1.squeeze().copy()
        # print(time.time()-startt)
        final = cv2.normalize(pr_mask.squeeze(), final1, 0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite(os.path.join(DATA_DIR,"test2.png"), final)
        if save_mask:
            # DATA_DIR = r"C:\Users\pawlowskj\Desktop\magisterka_test\bores\masks"
            cv2.imwrite(os.path.join(DATA_DIR, r"%s_mask.png"%test_dataset.ids[i]), final)

if __name__ == '__main__':
    pass
    # DATA_DIR = r"C:\Users\pawlowskj\Downloads\1_12_V&G_Euclid\1_12_V&G_Euclid\testing"
    # unet_inference_folder(DATA_DIR)