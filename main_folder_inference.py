from yolov8_locate_fiber_and_cut import locate_cut_main
from unet_inference import unet_inference_folder
from IEC_watershed_welz_circle import analyze_to_IEC

if __name__ == '__main__':

    #select proper spec you want to evaluate against
    spec_path = r"IEC_standards/spec_IEC_ed2_SMPC_RL26dB.json"

    full_size = False
    images_path = r"E:\2_DL\000_VG_training_set\scratche dla Kuby"
    if full_size:
        locate_cut_main(images_path)

    unet_inference_folder(images_path, model_path=r"models/recompiled_BestModel_19_12_2023_8GPU_TRAIN_SF_VALID_SF_Frozen_continue_last_epochs_unfrozen_LR_2e-5_vgg19.h5")

    analyze_to_IEC(images_path, images_path, spec_path)