import os
import json
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import cv2
from PIL import Image, ImageDraw
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

# DATA_DIR = r"E:\8_praca_magisterska\mtp_ferrule_images\labeled_images\08_02_yolov8_fibers_multiclass"
# DATA_DIR = r"E:\2_DL\000_VG_training_set\20_03_2023 fixed illumination\labeled\20_03_2023_seg_multiclass"
# DATA_DIR = r"E:\2_DL\000_VG_training_set\20_03_2023 fixed illumination\labeled\20_03_2023_seg_multiclass_cropped"
DATA_DIR = r"E:\2_DL\000_VG_training_set\0_organizacja\0_gotowe_datasety\10_10_2023_all_images"


# DATA_DIR = r"E:\2_DL\000_VG_training_set\0_organizacja\0_gotowe_datasety\09_08_2023 with old 250\segmentation_masks"

DATA_DIR = r"E:\8_praca_magisterska\mtp_ferrule_images\labeled_images\07_07_2023_fibers_standard_yolov8_seg_masks"
# DATA_DIR = r"E:\8_praca_magisterska\mtp_ferrule_images\labeled_images\19_04_2023_bores_seg_multiclass"
# DATA_DIR = r"E:\8_praca_magisterska\mtp_ferrule_images\labeled_images\22_04_2023_ferrule_seg_multiclass_crop"

# DATA_DIR = r"E:\8_praca_magisterska\mtp_ferrule_images\labeled_images\01_02_yolov8_ferrule_for_segm"

# DATA_DIR = r"E:\8_praca_magisterska\mtp_ferrule_images\labeled_images\26_07_2023_ferrule_semgmentation_cut_1024"

test_bool = False

# imsize = 512
imsize = 2048
# imsize = 384
# imsize = 192
# imsize = 960
# imsize = 1600

x_train_dir = os.path.join(DATA_DIR, 'train/images')
y_train_annotations = os.path.join(DATA_DIR, 'train/annotations.json')

x_valid_dir = os.path.join(DATA_DIR, 'val/images')
y_valid_annotations = os.path.join(DATA_DIR, 'val/annotations.json')

x_test_dir = os.path.join(DATA_DIR, 'val/images')
y_test_annotations = os.path.join(DATA_DIR, 'val/annotations.json')

# x_test_dir = r"C:\Users\pawlowskj\Desktop\test"
# y_test_dir = r"C:\Users\pawlowskj\Desktop\test"

###########
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    # plt.figure(figsize=(16, 5))
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(16, 5), sharex=True, sharey=True)
    for i, (name, image) in enumerate(images.items()):
        # plt.subplot(1, n, i + 1)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(' '.join(name.split('_')).title())
        if i > 0:
            ax[i].imshow(image)
        else:
            ax[i].imshow(image, vmin=0, vmax=255, cmap='inferno')
        print(image)
    plt.show()

# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# classes for data loading and preprocessing
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

    CLASSES = ['defect', 'scratch', 'bgr']
    # CLASSES = ['defect', 'bgr']

    def __init__(
            self,
            images_dir,
            masks_dir=None,
            classes=None,
            augmentation=None,
            preprocessing=None,
            multiplier=1,
            annotations_file=None,
    ):
        if annotations_file:
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)

            self._use_coco = True
            self.images_dir = images_dir
            self.image_id_to_info = {image['id']: image for image in coco_data.get('images', [])}
            existing_ids = []
            image_name_to_id = {}
            for image in coco_data.get('images', []):
                file_name = image['file_name']
                if os.path.exists(os.path.join(images_dir, file_name)):
                    existing_ids.append(file_name)
                    image_name_to_id[file_name] = image['id']
            # keep only images that actually exist on disk
            self.ids = existing_ids
            self.image_name_to_id = image_name_to_id

            annotations_map = defaultdict(list)
            for annotation in coco_data.get('annotations', []):
                annotations_map[annotation['image_id']].append(annotation)
            self.annotations = annotations_map

            category_name_to_id = {category['name'].lower(): category['id']
                                   for category in coco_data.get('categories', [])}
            try:
                self.class_values = [category_name_to_id[cls.lower()] for cls in classes]
            except KeyError as exc:
                missing = {cls for cls in classes if cls.lower() not in category_name_to_id}
                raise ValueError(f"Missing categories in annotations file: {missing}") from exc
            self.class_id_to_index = {class_id: index for index, class_id in enumerate(self.class_values)}

            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
            self.masks_fps = None
        else:
            if masks_dir is None:
                raise ValueError("masks_dir must be provided when annotations_file is not set")
            self._use_coco = False
            self.ids = os.listdir(images_dir)
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

            # convert str names to class values on masks
            try:
                self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
            except ValueError as exc:
                missing = {cls for cls in classes if cls.lower() not in self.CLASSES}
                raise ValueError(f"Unknown classes provided: {missing}") from exc

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

        if self._use_coco:
            mask = self._load_coco_mask(self.image_name_to_id[self.ids[index]], image.shape[0], image.shape[1])
        else:
            mask = cv2.imread(self.masks_fps[index], 0)

            # extract certain classes from mask (e.g. cars)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            background = np.clip(background, 0, 1)
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

    def _load_coco_mask(self, image_id, height, width):
        mask = np.zeros((height, width, len(self.class_values)), dtype='float32')
        for annotation in self.annotations.get(image_id, []):
            class_index = self.class_id_to_index.get(annotation['category_id'])
            if class_index is None:
                continue

            segmentation = annotation.get('segmentation', [])
            rle_mask = None
            if isinstance(segmentation, list):
                pil_mask = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(pil_mask)
                for segment in segmentation:
                    if len(segment) < 6:
                        continue
                    points = [(segment[i], segment[i + 1]) for i in range(0, len(segment), 2)]
                    draw.polygon(points, outline=1, fill=1)
                rle_mask = np.array(pil_mask, dtype='float32')
            elif isinstance(segmentation, dict) and {'size', 'counts'} <= segmentation.keys():
                rle_mask = self._decode_rle(segmentation, height, width)

            if rle_mask is None:
                continue

            rle_mask = rle_mask.astype('float32')
            mask[..., class_index] = np.maximum(mask[..., class_index], rle_mask)
        return mask

    @staticmethod
    def _decode_rle(rle, height, width):
        counts = rle.get('counts')
        if isinstance(counts, list):
            rle_counts = counts
        else:
            counts_bytes = counts.encode('ascii') if isinstance(counts, str) else counts
            rle_counts = []
            current = 0
            shift = 0
            for c in counts_bytes:
                c = c - 48
                current |= (c & 0x1F) << shift
                if c & 0x20:
                    shift += 5
                else:
                    rle_counts.append(current)
                    current = 0
                    shift = 0

        decode_height, decode_width = rle.get('size', [height, width])
        mask = np.zeros(decode_height * decode_width, dtype='float32')
        index = 0
        value = 0
        for count in rle_counts:
            if index + count > mask.size:
                count = mask.size - index
            if value == 1:
                mask[index:index + count] = 1
            index += count
            value = 1 - value
            if index >= mask.size:
                break
        decoded = mask.reshape((decode_height, decode_width), order='F')
        if decoded.shape[0] != height or decoded.shape[1] != width:
            padded = np.zeros((height, width), dtype='float32')
            h = min(height, decoded.shape[0])
            w = min(width, decoded.shape[1])
            padded[:h, :w] = decoded[:h, :w]
            decoded = padded
        return decoded


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
        #################
# Lets look at data we have
# dataset = Dataset(x_train_dir, annotations_file=y_train_annotations, classes=['defect', 'scratch'])

# image, mask = dataset[0] # get some sample
# visualize(
#     image=image,
#     defects_mask=mask[..., 0].squeeze(),
#     scratches_mask=mask[..., 1].squeeze(),
#     background_mask=mask[..., 2].squeeze(),
# )
############


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define heavy augmentations
def get_training_augmentation():
    train_transform = [
        # A.augmentations.geometric.resize.LongestMaxSize(max_size=384, interpolation=cv2.INTER_AREA, always_apply=False,
        #                                                 p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        # A.RandomSizedCrop(min_max_height=(100, 600), height=imsize, width=imsize, p=0.1),

        A.ShiftScaleRotate(scale_limit=0, rotate_limit=90, shift_limit=0.3, p=0.5, border_mode=cv2.BORDER_REFLECT),

        A.PadIfNeeded(min_height=imsize, min_width=imsize, always_apply=True),
        A.RandomCrop(height=imsize, width=imsize, always_apply=True),
        # A.RandomCrop(height=192, width=192, always_apply=False),

        # A.IAAAdditiveGaussianNoise(p=0),
        # A.IAAPerspective(p=0),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.5,
        ),

        # A.OneOf(
        #     [
        #         A.Blur(blur_limit=3, p=1),
        #         A.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.5,
        # ),

        # A.OneOf(
        #     [
        #         # A.RandomBrightnessContrast(p=1),
        #         A.HueSaturationValue(p=1),
        #     ],
        #     p=0.3,
        # ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(min_height=imsize, min_width=imsize, always_apply=True),
        A.CenterCrop(height=imsize, width=imsize, always_apply=True),
        # A.augmentations.geometric.resize.LongestMaxSize(max_size=384, interpolation=cv2.INTER_AREA, always_apply=False,
        #                                                 p=1),
                                                    ]
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
#########

# Lets look at augmented data we have
# dataset = Dataset(x_train_dir, y_train_dir, classes=['defect', 'scratch'], augmentation=get_training_augmentation())
dataset = Dataset(
    x_train_dir,
    classes=['defect', 'scratch'],
    augmentation=get_training_augmentation(),
    multiplier=3,
    annotations_file=y_train_annotations,
)
#
# for i in range(len(dataset)):
#     print(dataset[i][0].shape)
# # dataset = Dataset(x_train_dir, y_train_dir, classes=['zabrudzenie_wewnatrz', 'zabrudzenie_ferrula', 'deformacja_otworu'], augmentation=get_training_augmentation())
# dataset = Dataset(x_train_dir, y_train_dir, classes=['defect', 'scratch'], augmentation=get_training_augmentation())



image, mask = dataset[0] # get some sample
visualize(
    image=image,
    contamination_mask=mask[..., 0].squeeze(),
    scr_mask=mask[..., 1].squeeze(),
    bgr_mask=mask[..., 2].squeeze(),
)
image, mask = dataset[1] # get some sample
visualize(
    image=image,
    contamination_mask=mask[..., 0].squeeze(),
    scr_mask=mask[..., 1].squeeze(),
    bgr_mask=mask[..., 2].squeeze(),
)
image, mask = dataset[2] # get some sample
visualize(
    image=image,
    contamination_mask=mask[..., 0].squeeze(),
    scr_mask=mask[..., 1].squeeze(),
    bgr_mask=mask[..., 2].squeeze(),
)
image, mask = dataset[3] # get some sample
visualize(
    image=image,
    contamination_mask=mask[..., 0].squeeze(),
    scr_mask=mask[..., 1].squeeze(),
    bgr_mask=mask[..., 2].squeeze(),
)


###############
import segmentation_models as sm

# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`

# fig, ax = plt.subplots(nrows=1, ncols=1)

# bbns = ['vgg19', 'resnet34', 'resnet152', 'seresnet152', 'densenet201', 'inceptionv3', 'mobilenetv2', 'efficientnetb7']
# bbns = np.array(['vgg19', 'resnet34', 'resnet152', 'seresnet152', 'densenet201', 'inceptionv3', 'mobilenetv2', 'efficientnetb3', 'efficientnetb5'])
bbns = np.array(['vgg19'])

# batches = np.array([6, 26, 6, 4, 4, 2, 6, 5, 3])
hists = []
# bbns = ['vgg19']

for backbon in bbns:
    # BACKBONE = 'efficientnetb7'
    BACKBONE = backbon
    # BACKBONE = 'resnet34'

    # BATCH_SIZE = batches[bbns==BACKBONE][0]
    BATCH_SIZE = 2
    CLASSES = ['defect', 'scratch']
    # CLASSES = ['zabrudzenie_wewnatrz', 'zabrudzenie_ferrula', 'deformacja_otworu']
    # CLASSES = ['defect']

    LR = 0.01
    EPOCHS = 100

    preprocess_input = sm.get_preprocessing(BACKBONE)

    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES)+1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    #create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, encoder_weights='imagenet', encoder_freeze=True)
    # model = sm.Linknet(BACKBONE, classes=n_classes, activation=activation, encoder_weights='imagenet', encoder_freeze=True)
    # model = sm.FPN(BACKBONE, classes=n_classes, activation=activation, encoder_weights='imagenet', encoder_freeze=True)
    # model = sm.PSPNet(BACKBONE, classes=n_classes, activation=activation, encoder_weights='imagenet', encoder_freeze=True)


    # define optomizer
    optim = keras.optimizers.Adam(LR)

    #dide loss for mtp fibers
    # dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.24844491, 0.74862336, 0.00293173]))
    #dice loss for cut ferrules
    # dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.33788873, 0.65312908, 0.00898219]))
    #dice loss for Wolverine
    # dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.78847956, 0.21030714, 0.0012133]), beta=1)
    #dice loss for Wolverine 09_08_2023
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.40410177, 0.59433297, 0.00156525]), beta=1)

    # fpfn_loss = sm.losses.FPFN_loss(class_weights=np.array([0.40410177, 0.59433297, 0.00156525]))

    #dice loss for MTP pin bores
    # dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.39294842, 0.60428478, 0.0027668]), beta=1)
    # dice loss for MTP ferrule
    # dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.99466914, 0.00533086]), beta=1)


    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = (1/2 * dice_loss) + (1/2 * focal_loss)
    # total_loss = (1 * dice_loss)


    # total_loss = fpfn_loss

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss
    metrics = [
                sm.metrics.IOUScore(threshold=0.3), sm.metrics.FScore(threshold=0.3, name='F1'), sm.metrics.FScore(threshold=0.3, class_indexes=[0], name='f1_defect'), sm.metrics.FScore(threshold=0.3, class_indexes=[1], name='f1_scratch')
                # sm.metrics.IOUScore(threshold=0.5, class_indexes=0), sm.metrics.FScore(threshold=0.5, class_indexes=0),
                # sm.metrics.IOUScore(threshold=0.5, class_indexes=1), sm.metrics.FScore(threshold=0.5, class_indexes=1)
                # sm.metrics.IOUScore(threshold=0.5, class_indexes=2), sm.metrics.FScore(threshold=0.5, class_indexes=2)
                ]
    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    # Dataset for train images
    train_dataset = Dataset(
        x_train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
        multiplier=1,
        annotations_file=y_train_annotations,
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        x_valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
        annotations_file=y_valid_annotations,
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)


    # check shapes for errors
    # assert train_dataloader[0][0].shape == (BATCH_SIZE, imsize, imsize, 3)
    # assert train_dataloader[0][1].shape == (BATCH_SIZE, imsize, imsize, n_classes)

    # define callbacks for learning rate scheduling and best checkpoints saving

    log_dir = os.path.join(DATA_DIR, BACKBONE+"_384size_27_10_2023")
    print(log_dir)
    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(DATA_DIR, r"%s_best_model_384size_27_10_2023.h5"%BACKBONE), save_weights_only=False, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=0.0001),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=200)
    ]
    # model.load_weights(os.path.join(DATA_DIR, r"best_model.h5"))

    # train model
    history = model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )
    hists.append(history)


if test_bool:
    model_list = []

    for backbon in bbns:
        BACKBONE = 'efficientnetb3'
        BACKBONE = 'vgg19'
        # BACKBONE = backbon

        BATCH_SIZE = 1
        CLASSES = ['defect', 'scratch']
        # CLASSES = ['zabrudzenie_wewnatrz', 'zabrudzenie_ferrula', 'deformacja_otworu']
        # CLASSES = ['contamination']
        LR = 0.001
        EPOCHS = 60

        preprocess_input = sm.get_preprocessing(BACKBONE)

        # define network parameters
        n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES)+1)  # case for binary and multiclass segmentation
        activation = 'sigmoid' if n_classes == 1 else 'softmax'

        #create model
        model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, encoder_weights='imagenet', encoder_freeze=True)

        # model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, encoder_weights='imagenet', encoder_freeze=False)

        # define optomizer
        optim = keras.optimizers.Adam(LR)

        # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
        dice_loss = sm.losses.DiceLoss(class_weights=np.array([1.0, 0.0]))
        focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)

        # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
        # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        # compile keras model with defined optimozer, loss and metrics
        model.compile(optim, total_loss, metrics)

        test_dataset = Dataset(
            x_test_dir,
            classes=CLASSES,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocess_input),
            annotations_file=y_test_annotations,
        )
        test_dataset = Dataset(
            r"C:\Users\pawlowskj\Downloads\1_12_V&G_Euclid\1_12_V&G_Euclid\V&G_results",
            r"C:\Users\pawlowskj\Downloads\1_12_V&G_Euclid\1_12_V&G_Euclid\V&G_results",
            classes=CLASSES,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocess_input),
        )
        test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

        # load best weights
        model.load_weights(os.path.join(r"E:\2_DL\000_VG_training_set\0_organizacja\0_gotowe_datasety\10_10_2023_all_images_masks", r"BestModel_02_11_2023_multiGPU_vgg19.h5"))
        # model.load_weights(os.path.join(DATA_DIR, r"vgg19_best_model_25_09_2023.h5"))
        # model.save(os.path.join(DATA_DIR, r"full_model_vgg19_best_model_13_07_2023.h5"))

        model_list.append(model)

        # scores = model.evaluate_generator(test_dataloader)
        #
        # print("Loss: {:.5}".format(scores[0]))
        # for metric, value in zip(metrics, scores[1:]):
        #     print("mean {}: {:.5}".format(metric.__name__, value))

        n = len(test_dataset)
        ids = np.arange(len(test_dataset))
        # ids = [1]

model.compile(optim, tf.keras.losses.MeanSquaredError(), tf.keras.metrics.BinaryCrossentropy())
model.save(os.path.join(r"E:\PyCharm_projects\VG_Unet", r"recompiled_with_standard_loss_BestModel_02_11_2023_multiGPU_vgg19"))
model1 = keras.models.load_model(r"models/unet_vgg19_06_dec_2023.h5")
# #

save_mask = True
copy_if_found = False
import tqdm
for i in tqdm.trange(len(ids)):
    image0, gt_mask = test_dataset[i]
    # image, gt_mask = test_dataset[i+1]
    # image = cv2.imread(r"E:\8_praca_magisterska\mtp_ferrule_images\labeled_images\19_04_2023_bores_seg_multiclass\val\images\bore_sm_12_158c_2.png", cv2.IMREAD_COLOR).astype("float32")
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    import time
    startt = time.time()
    image = image0.copy() - [np.average(image0[:,:, 0]), np.average(image0[:,:, 1]), np.average(image0[:,:, 2])]
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
        cv2.imwrite(os.path.join(r"C:\Users\pawlowskj\Downloads\1_12_V&G_Euclid\1_12_V&G_Euclid\V&G_results", r"%s_recompile_test_mask.png"%test_dataset.ids[i]), final)
    if copy_if_found:
        if (final[:, :, 1]>250).any():
            import shutil
            shutil.copyfile(r"E:\2_DL\000_VG_training_set\0_organizacja\zbieranie na produkcji lipiec sierpien 2023\0_110244416_0_0-stacked\%s.png"% test_dataset.ids[i], r"E:\2_DL\000_VG_training_set\0_organizacja\zbieranie na produkcji lipiec sierpien 2023\08_08_scratches\%s.png"% test_dataset.ids[i])
            cv2.imwrite(os.path.join(DATA_DIR, r"%s_.png" % test_dataset.ids[i]), image0)


    # visualize(
    #     image=denormalize(image.squeeze()),
    #     gt_mask=gt_mask.squeeze(),
    #     pr_mask=pr_mask.squeeze(),
    # )
#
# n = 5
# ids = np.random.choice(np.arange(len(test_dataset)), size=n)
#
# for i in ids:
#     image, gt_mask = test_dataset[i]
#     image = np.expand_dims(image, axis=0)
#     pr_mask = model.predict(image)
#     image, gt_mask = test_dataset[i]
#     visualize(
#         # image=image,
#         image=denormalize(image),
#         gt_mask=gt_mask.squeeze(),
#         pr_mask=pr_mask.squeeze(),
#     )