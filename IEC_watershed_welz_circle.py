from sklearn.cluster import OPTICS
import os
import glob
import matplotlib
import tqdm
import other_smallest_circle_method
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import json

def scratch_width(points):
    slope, intercept, r_value, p_value, std_err = stats.linregress(points[:, 0], points[:, 1])
    d = np.abs(slope * points[:, 0] - points[:, 1] + intercept) / np.sqrt(slope ** 2 + 1)
    percentile_width = 2 * np.percentile(d, 90)
    return percentile_width, slope, intercept
def trivial_circle(points):
    """Compute the circle for 0, 1, 2 or 3 points."""
    if len(points) == 0:
        return (0, 0, 0)
    elif len(points) == 1:
        return (points[0][0], points[0][1], 0)
    elif len(points) == 2:
        middle = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
        radius = np.sqrt((points[0][0] - middle[0]) ** 2 + (points[0][1] - middle[1]) ** 2)
        return (*middle, radius)
    else:
        ax, ay = points[0]
        bx, by = points[1]
        cx, cy = points[2]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        radius = np.sqrt((ax - ux) ** 2 + (ay - uy) ** 2)
        return (ux, uy, radius)

def welzl(points, boundary_points):
    """Welzl's algorithm."""
    if len(points) == 0 or len(boundary_points) == 3:
        return trivial_circle(boundary_points)

    # Copy the list of points to avoid modifying the input
    points_copy = list(points)
    p = points_copy.pop()
    circle = welzl(points_copy, boundary_points)

    if np.sqrt((circle[0] - p[0]) ** 2 + (circle[1] - p[1]) ** 2) <= circle[2]:
        return circle
    return welzl(points_copy, boundary_points + [p])

def cstm_jet(x):
    return plt.cm.jet((np.clip(x,2,10)-2)/8.)

def distance_func(center, point):
    try:
        return np.linalg.norm(center - point, axis=1)
    except np.AxisError:
        return np.linalg.norm(center - point)

def categorize_and_count_defects(defects, specifications):
    # Splitting defects based on their zone
    zone_A_defects = defects[defects[:, 1] <= 25]
    zone_B_defects = defects[(defects[:, 1] > 25) & (defects[:, 1] <= 110)]

    # Count defects in Zone A by size bucket
    zone_A_counts = {
        '< 2': np.sum(zone_A_defects[:, 0] < 2),
        '2-3': np.sum((2 <= zone_A_defects[:, 0]) & (zone_A_defects[:, 0] <= 3)),
        '> 3': np.sum(zone_A_defects[:, 0] > 3)
    }

    # Count defects in Zone B by size bucket
    zone_B_counts = {
        '<= 25': np.sum(zone_B_defects[:, 0] <= 25),
        '> 25': np.sum(zone_B_defects[:, 0] > 25)
    }

    # Check against specifications
    for bucket, count in zone_A_counts.items():
        if count > specifications['A']['defects'][bucket]:
            return (False, zone_A_counts, zone_B_counts)
    for bucket, count in zone_B_counts.items():
        if count > specifications['B']['defects'][bucket]:
            return (False, zone_A_counts, zone_B_counts)

    return (True, zone_A_counts, zone_B_counts)

def categorize_and_count_scratches(defects, specifications):
    # Splitting defects based on their zone
    try:
        zone_A_defects = defects[defects[:, 1].astype(int) == 1]

        # Count defects in Zone A by size bucket
        zone_A_counts = {
            '< 3': np.sum(zone_A_defects[:, 0] < 2),
            '3-4': np.sum((3 <= zone_A_defects[:, 0]) & (zone_A_defects[:, 0] <= 4)),
            '> 4': np.sum(zone_A_defects[:, 0] > 3)
        }
        # Check against specifications
        for bucket, count in zone_A_counts.items():
            if count > specifications['A']['defects'][bucket]:
                return (False, zone_A_counts)
    except Exception as e:
        print(e)

    return (True, zone_A_counts)


# Check the defects against the specifications



def cluster_defects(points, im, x, y):
    image = np.zeros((im.shape[1], im.shape[0])).astype('uint8')
    image[points[:, 0].astype(int), points[:, 1].astype(int)] = 255

    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((1, 1)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)

    yhat_defects = labels.copy()

    c = plt.cm.jet(yhat_defects)
    welzl_circles = np.zeros((len(np.unique(yhat_defects))-1, 3))

    for label in np.unique(yhat_defects):
        if label != 0:  # Exclude noise
            cluster_points = np.c_[x[labels==label], y[labels==label]]
            circle = other_smallest_circle_method.make_circle(cluster_points)
            welzl_circles[label-1] = np.array(circle)

    welzl_circles_array = welzl_circles
    dist_to_center = distance_func(np.array([im.shape[1] / 2, im.shape[0] / 2]), welzl_circles_array[:, [0, 1]])
    welzl_circles_array = np.c_[welzl_circles_array*[1,1,2], dist_to_center]
    return welzl_circles_array

def count_defects(defect_list, specification):
    pass_fail_results = np.ones((defect_list.shape[0]), dtype=bool)
    specification_counts = np.zeros((specification.shape[0]))
    for dd in range(defect_list.shape[0]):
        for sp in range(specification.shape[0]):
            if (defect_list[dd,2] > specification[sp,2] and defect_list[dd,2] < specification[sp,3]):
                specification_counts[sp] += 1
            if (defect_list[dd,2] > specification[sp,3] and sp==specification.shape[0]-1):
                specification_counts[sp] += 1
                pass_fail_results[dd] = False

    if any(specification_counts > specification[:, -1]):
        result_in_zone = False
    else:
        result_in_zone = True

    return result_in_zone, pass_fail_results, specification_counts


def cluster_scratches_to_IEC(cluster_model, points):
    specifications = {
        'A': {
            'scratches': {
                '< 2': float('inf'),
                '3-4': 1,
                '> 4': 0
            }
        },
    }
    scratches = []
    slopes = []
    intercepts = []
    is_in_A_zone = []
    try:
        yhat_defects = cluster_model.fit_predict(points)

        # ax[1].scatter(data_d[:, 0], data_d[:, 1], c=yhat_defects, cmap='jet', s=0.001, marker="*")
        c = plt.cm.jet(yhat_defects)

        for label in np.unique(yhat_defects):
            if label != -1:  # Exclude noise
                cluster_points = points[yhat_defects == label]
                sw, slope, intercept = scratch_width(cluster_points)
                scratches.append(sw)
                slopes.append(slope)
                intercepts.append(intercept)
                y_true = slope * cluster_points[:, 0] + intercept

                perpendicular_slope = -1 / slope
                dx = (sw/2) / np.sqrt(1 + perpendicular_slope ** 2)
                dy = perpendicular_slope * dx

                # Calculate lines above and below the best fit line
                y_upper = y_true + dy
                y_lower = y_true - dy

                ax[1].plot(cluster_points[:, 0]+dx, y_upper, color='blue', linestyle='--', label='Upper line', linewidth=0.1)
                ax[1].plot(cluster_points[:, 0]-dx, y_lower, color='blue', linestyle='--', label='Lower line', linewidth=0.1)
                inside_A = False
                for i in range(len(cluster_points)):
                    if distance_func(np.array([im.shape[1] / 2, im.shape[0] / 2]),
                                                   cluster_points[i, :]) < 25*(125 / 121):
                        inside_A = True
                        break
                is_in_A_zone.append(inside_A)

        scratches_array = np.array(scratches)
        scratches_array = np.c_[scratches_array, is_in_A_zone]
        # print(scratches)
        try:
            results = categorize_and_count_scratches(scratches_array, specifications)
            print(f"\nCzy konektor spełnia wymagania scratchy: {results[0]}, \nScratche w strefie A: {results[1]}")
            return results
        except Exception as e:
            print(e)
            return np.array([0, 0])

    except Exception as e:
        print(e)
        print(f"\nCzy konektor spełnia wymagania scratchy: True, \nScratche w strefie A: 0")
    fig.tight_layout()


def parse_specifications(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    specifications = data.get('specifications', {})
    zones = data.get('zones', {})
    px_scale = data.get('pixel_scale', {})

    # Define the structure of the array
    dtype = [('zone', 'U1'), ('type', 'U10'), ('min_size', 'f4'), ('max_size', 'f4'), ('max_count', 'f4')]

    all_requirements = np.zeros((50, 5), dtype=object)
    all_requirements_not_rescaled = np.zeros((50, 5), dtype=object)

    iter = 0
    for zone in 'ABCD':
        zone_specifications = specifications.get(zone, {})

        if not zone_specifications:
            pass
        else:
            for defect_type in ['defects', 'scratches']:
                defect_data = zone_specifications.get(defect_type, [])
                if defect_data:
                    for item in defect_data:
                        min_size = item['range']['min_size']
                        max_size = np.inf if item['range']['max_size'] == 'Infinity' else item['range']['max_size']
                        max_count = np.inf if item['max_count'] == 'Infinity' else item['max_count']
                        all_requirements[iter] = [zone, defect_type[:-1] if defect_type[:-1]=='defect' else defect_type[:-2], min_size/px_scale, max_size/px_scale, max_count]
                        all_requirements_not_rescaled[iter] = [zone, defect_type[:-1] if defect_type[:-1]=='defect' else defect_type[:-2], min_size, max_size, max_count]
                        iter += 1

    return all_requirements_not_rescaled[0:iter], all_requirements[0:iter], zones, px_scale

def create_zone_info_array(zone_info, pixel_scale):
    # Initialize the array
    # Columns: Zone, Type, Inner Radius, Outer Radius
    zone_array = np.zeros((len(zone_info)+1, 4), dtype=object)

    # Fill in the array
    for idx, (zone, info) in enumerate(zone_info.items()):
        zone_type = info['type']
        inner_radius = info.get('inner_diameter', 0) / 2 / pixel_scale
        outer_radius = info.get('outer_diameter', info.get('diameter', 0)) / 2 / pixel_scale

        zone_array[idx] = [zone, zone_type, inner_radius, outer_radius]
    zone_array[-1] = ["outside", "na", outer_radius, np.inf]
    return zone_array


def pad_if_needed(image, min_height, min_width, pad_value=0, pad_mode='constant'):
    """
    Pad an image to the specified minimum height and width if needed.

    Parameters:
    - image: Input image as a numpy array of shape (H, W, C).
    - min_height: Desired minimum height of the image.
    - min_width: Desired minimum width of the image.
    - pad_value: Value to fill the pad areas with if pad_mode is 'constant'. Default is 0.
    - pad_mode: Mode of padding, can be 'constant', 'edge', or 'reflect'. Default is 'constant'.

    Returns:
    - Padded image as a numpy array.
    """

    # Calculate how much padding is needed
    height, width = image.shape[:2]
    pad_height = max(min_height - height, 0)
    pad_width = max(min_width - width, 0)

    # Determine padding for height and width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Pad the image
    if pad_mode == 'constant':
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                              mode=pad_mode, constant_values=pad_value)
    else:
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                              mode=pad_mode)

    return padded_image

def analyze_to_IEC(path_original_images, path_masks, path_spec_data):
    matplotlib.use('TkAgg')
    plot_bool = True

    p_mask = path_masks
    p_ori = path_original_images

    spec_path = path_spec_data

    spec_data_array_not_rescaled, spec_data_array, zone_info, pixel_scale = parse_specifications(spec_path)
    zone_array_info = create_zone_info_array(zone_info, pixel_scale)
    center = 384

    masks = []
    oris = []
    for file in glob.glob(os.path.join(p_mask, "*_mask.png")):
        masks.append(file)
        oris.append(file.replace(p_mask, p_ori).replace("_mask.png", ""))

    for i in tqdm.trange(len(masks)):
        defects_in_zones = np.zeros([len(zone_array_info)], dtype=object)
        scratches_in_zones = np.zeros([len(zone_array_info)], dtype=object)

        im = cv2.imread(masks[i])
        im_1 = cv2.imread(oris[i])
        try:
            assert im_1.shape[0]==im_1.shape[1]==int(center*2)
        except:
            im_1 = pad_if_needed(im_1, int(center*2), int(center*2), pad_value=0, pad_mode='constant')

        y, x = np.meshgrid(np.linspace(0, im.shape[1]-1, im.shape[1]),np.linspace(0, im.shape[0]-1, im.shape[0]))
        xf = x.flatten()
        yf = y.flatten()
        im_f_d = im[:, :, 0].flatten()
        im_f_s = im[:, :, 1].flatten()
        im_f_defo = im[:, :, 2].flatten()

        data_d = np.c_[yf[im_f_d>200], xf[im_f_d>200]]
        dist_data_d = np.sqrt((data_d[:, 0]-center)**2+(data_d[:, 1]-center)**2)

        data_s = np.c_[yf[im_f_s > 200], xf[im_f_s > 200]]
        dist_data_s = np.sqrt((data_s[:, 0] - center) ** 2 + (data_s[:, 1] - center) ** 2)

        for zz in range(len(zone_array_info)):
            defects_in_zones[zz] = data_d[np.bitwise_and(dist_data_d>zone_array_info[zz, 2], dist_data_d<zone_array_info[zz, 3])]
            scratches_in_zones[zz] = data_s[np.bitwise_and(dist_data_s>zone_array_info[zz, 2], dist_data_s<zone_array_info[zz, 3])]

            # np.savetxt("scratches.txt", np.r_[scratches_in_zones[0], scratches_in_zones[1], scratches_in_zones[2], scratches_in_zones[2]], fmt="%i")

        data = np.r_[data_d, data_s]
        # define the model
        model = OPTICS(min_samples=2, xi=0.001, min_cluster_size=2)


        if plot_bool:
            # fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(9, 5))
            ax[0].imshow(im_1)
            ax[1].imshow(im_1)

            # sumix_image = cv2.imread(masks[i].replace("_mask.png", ""))
            # sumix_image = cv2.cvtColor(sumix_image, cv2.COLOR_GRAY2RGB)
            # sumix_image_temp = np.zeros_like(sumix_image)+255
            # sumix_image = cv2.copyMakeBorder(sumix_image, 46, 46, 46, 46, cv2.BORDER_CONSTANT, value=(255,255,255))
            # ax[2].imshow(sumix_image)

            ax[0].set_title("Input")
            ax[1].set_title("Watershed clustering + IEC (smallest bounding circle)")
            # ax[2].set_title("Analiza oryginalnego programu\nMaxInspect, firmy Sumix")
        defect_counts_per_zone = np.zeros((len(zone_array_info)), dtype=object)
        scratch_counts_per_zone = np.zeros((len(zone_array_info)), dtype=object)
        try:
            # data_d = data_d[:, 1:0]
            with open(f"{oris[i]}_counts.txt", "ab") as f:
                for zz in range(len(zone_array_info)):
                    ax[0].add_artist(plt.Circle((center, center), zone_array_info[zz, -1], color='gray', linestyle='--', fill=False, linewidth=0.3))
                    ax[1].add_artist(plt.Circle((center, center), zone_array_info[zz, -1], color='gray', linestyle='--', fill=False, linewidth=0.3))
                    if zone_array_info[zz, 0]!="C":
                        try:
                            defect_counts_per_zone[zz] = cluster_defects(defects_in_zones[zz], im, x, y)
                            spec_applied = spec_data_array[np.bitwise_and(spec_data_array[:, 0]==zone_array_info[zz, 0], spec_data_array[:, 1]=='defect')]
                            pass_fail_final, pass_fail_per_defect, counts = count_defects(defect_counts_per_zone[zz], spec_applied)

                            # print(counts, spec_data_array[spec_data_array[:, 0]==zone_array_info[zz, 0]][spec_data_array[:, 1]=='defect'])
                            np.savetxt(f, np.c_[counts.reshape(len(counts), 1),spec_data_array_not_rescaled[spec_data_array_not_rescaled[:, 0]==zone_array_info[zz, 0]][spec_data_array_not_rescaled[spec_data_array_not_rescaled[:, 0]==zone_array_info[zz, 0]][:, 1]=='defect'], np.array([pass_fail_final]*len(counts)).reshape(len(counts), 1)], fmt="%s")

                            if zone_array_info[zz, 0]!="outside":
                                iterr = 0
                                for ii in range(len(defect_counts_per_zone[zz])):
                                # for center_x, center_y, radius, dist in defect_counts_per_zone[zz]:
                                    center_x, center_y, radius, dist = defect_counts_per_zone[zz][ii]
                                    circle = plt.Circle((center_x, center_y), radius/2, fill=False, edgecolor=(0, 1, 0) if pass_fail_per_defect[ii] else (1, 0, 0),
                                                        linewidth=0.2)
                                    ax[1].add_patch(circle)
                                    iterr += 1
                            else:
                                iterr = 0
                                for ii in range(len(defect_counts_per_zone[zz])):
                                # for center_x, center_y, radius, dist in defect_counts_per_zone[zz]:
                                    center_x, center_y, radius, dist = defect_counts_per_zone[zz][ii]
                                    circle = plt.Circle((center_x, center_y), radius/2, fill=False, edgecolor=(0, 0, 1),
                                                        linewidth=0.2)
                                    ax[1].add_patch(circle)
                                    iterr += 1

                        except Exception as e:
                            print(e)
                            defect_counts_per_zone[zz] = np.nan

                        #TODO clustering scratches

                        # try:
                        #     scratch_counts_per_zone[zz] = cluster_scratches_to_IEC(model, scratches_in_zones[zz])
                        # except:
                        #     defect_counts_per_zone[zz] = np.nan
            # np.savetxt(masks[i].replace("_mask.png", "_defects.txt"), np.conc)
        except Exception as e:
            print(e)
            print(oris[i])

        if plot_bool:
            fig.tight_layout()
            plt.savefig(masks[i].replace("_mask.png", "_inference_vgg.png"), dpi=1200)
            plt.close()


if __name__ == '__main__':
    pass
    # p_mask = r"C:\Users\pawlowskj\Downloads\1_12_V&G_Euclid\1_12_V&G_Euclid\testing" # mask images have name +"_mask.png"
    # p_ori = r"C:\Users\pawlowskj\Downloads\1_12_V&G_Euclid\1_12_V&G_Euclid\testing"
    # spec_path = r"IEC_standards/spec_IEC_ed2_SMPC_RL26dB.json"
    #
    # analyze_to_IEC(p_ori, p_mask, spec_path)