import argparse
import os
import glob
import pascal_voc
from cv2 import cv2
from tqdm import tqdm
from plate_generator import PlateGenerator
from transformations import *
from asset_manager import AssetManager


project_config_path = './win_conf.json'

assets = AssetManager(project_config_path)
parser = argparse.ArgumentParser(description='Reading input arguments.')
parser.add_argument('--num_out_img', default=assets.generator_config['num_out_img'], type=int)
parser.add_argument('--output_directory', default="./simulated_ocr_dataset", type=str)
parser.add_argument('--img_per_package', default=1000000, type=int)
parser.add_argument('--apply_misc_noise', default=assets.generator_config['apply_misc_noise'], type=bool)
parser.add_argument('--apply_dirt', default=assets.generator_config['apply_dirt'], type=bool)

args = parser.parse_args()
if os.path.exists(args.output_directory) is False:
    os.makedirs(args.output_directory)
annotation_path = ''
images_path = ''
xmls_path = ''
package_counter = 0
do_tranform = assets.transformations_config['do_tranform']
use_bg = assets.transformations_config['use_bg']
use_brightening = assets.transformations_config['use_brightening']
max_bright_value = assets.transformations_config['max_bright_value']
use_tiling = assets.transformations_config['use_tiling']
min_plate_width = assets.transformations_config['min_plate_width']
plate_width_factor = assets.transformations_config['plate_width_factor']
plate_image_width = assets.transformations_config['output_size'][0]
pre_distortion = assets.transformations_config['pre_distortion']
cam_step = assets.transformations_config['cam_step']
use_tiling_method = assets.transformations_config['use_tiling_method']
scale_factor = assets.transformations_config['use_tiling_scale_factor']
bg_counter = 0
tile_counter = 1
print(f'\ngenerating {args.num_out_img} images.')
current_directory = args.output_directory
if os.path.exists(current_directory) is False:
    os.mkdir(current_directory)
current_directory = args.output_directory
if  os.path.exists(current_directory) is False:
    os.mkdir(current_directory)
annotation_path = os.path.join(current_directory, 'anns')
if os.path.exists(annotation_path) is False:
    os.mkdir(annotation_path)
pascal_annotation_path = os.path.join(current_directory, 'pascal_anns')
if os.path.exists(pascal_annotation_path) is False:
    os.mkdir(pascal_annotation_path)
xmls_path = os.path.join(annotation_path, 'xmls')
if os.path.exists(xmls_path) is False:
    os.mkdir(xmls_path)
images_path = os.path.join(current_directory, '')
if os.path.exists(images_path) is False:
    os.mkdir(images_path)
last_files = glob.glob(current_directory + '//*.jpg')
start_index = len(last_files) + 30000
progress = range(start_index , start_index + args.num_out_img)
if use_tiling_method != 2:
    bg_insts, _ = pascal_voc.parse_voc_annotation('./assets/Plate_detection_images/Annotations/', './assets/Plate_detection_images/JPEG_Images/', './assets/Plate_final.txt', ['Plate'])
    # bg_insts, _ = pascal_voc.parse_voc_annotation('D:\\ANPR_yooztaa\\TestingData\\bg_images\\parking\\Parking_Car\\', 'D:\\ANPR_yooztaa\\TestingData\\bg_images\\parking\\Parking_Car\\', 'D:\\ANPR_yooztaa\\TestingData\\bg_images\\parking\\Parking_Car\\Plate_final0.txt', ['Plate'])
else:
    bg_insts = pascal_voc.parse_voc_annotation2('./assets/bg_images/cars/back/')
    #bg_insts = pascal_voc.parse_voc_annotation2('./assets/bg_images/cars/front/')
random.shuffle(bg_insts)
bg_count = len(bg_insts)
ff = 0
last_x_offset = 0
last_y_offset = 0
cam_x = -cam_step
new_image = False
augment_data = AugmentData
augment_data.init(AugmentData)
out_offset = 0
gausian_filter_sizes = [3, 5, 7, 9]
index = progress.start
while index <= progress.stop:
    ff += 1
    plate_generator = PlateGenerator(assets)
    plate_objects = []
    plate, annotation, plate_objects = plate_generator.get_rnd_plate(apply_misc_noise=args.apply_misc_noise,
                                                      apply_dirt=args.apply_dirt)
    plate_h, plate_w, _ = plate.shape
    if pre_distortion is True:
        gausian_filter_index = random.randint(0, len(gausian_filter_sizes) - 1)
        plate = cv2.blur(plate, (gausian_filter_sizes[gausian_filter_index], gausian_filter_sizes[gausian_filter_index]))
        plate = augment_data.make_random_distortion(augment_data, plate)
    if do_tranform is True:
        bg_counter += 1
        if use_bg:
            plate_img, annotation_img, plate_img_objects = perspective_transform_with_bg(plate, annotation, assets.transformations_config,
                                                                     assets.ann_step, bg_insts[bg_counter % bg_count], plate_objects)

        elif use_tiling:
                new_image = False
                plate_img = None
                annotation_img = None
                plate_img_objects = None
                cam_x = (cam_x + cam_step) % plate_image_width
                if use_tiling_method == 1:
                    last_x_offset = random.uniform(0, min_plate_width)
                    last_y_offset = random.uniform(0, min_plate_width)
                else:
                    last_x_offset = 0
                    last_y_offset = 0
                plate_width = min_plate_width
                plate_img = plate_generator.get_next_background(
                    (assets.transformations_config['output_size'][0], assets.transformations_config['output_size'][1]))

                while new_image is False:
                    bg_counter += 1
                    plate, annotation, plate_objects = plate_generator.get_rnd_plate(apply_misc_noise=args.apply_misc_noise,
                                                                      apply_dirt=args.apply_dirt)

                    gausian_filter_index = random.randint(0, len(gausian_filter_sizes) - 1)
                    plate = cv2.blur(plate, (
                    gausian_filter_sizes[gausian_filter_index], gausian_filter_sizes[gausian_filter_index]))
                    # cv2.imshow('plate', plate);  cv2.waitKey()
                    plate_img, annotation_img, plate_img_objects, last_x_offset, last_y_offset, new_image = \
                        perspective_transform_with_tiling(plate, annotation, assets.transformations_config,
                                                          bg_insts[bg_counter % bg_count], use_tiling_method, scale_factor,
                                                          assets.ann_step, plate_objects, plate_img, annotation_img,
                                                          plate_img_objects, last_x_offset, last_y_offset, cam_x, plate_width)
                    if last_x_offset > plate_image_width - plate_width * 1.5:
                        plate_width = plate_width * plate_width_factor
                        last_y_offset = int(numpy.ceil(last_y_offset + 2 * (plate_width / plate_w) * plate_h))
                        last_x_offset = random.uniform(0, plate_width * .5)




        else:
            plate_img, annotation_img, plate_img_objects = perspective_transform(plate, annotation, assets.transformations_config, assets.ann_step,plate_objects, ff)
    else:
        plate_img, annotation_img, plate_img_objects = perspective_transform_ocr(plate, annotation, assets.transformations_config, assets.ann_step,plate_objects, ff)
        # plate_img = plate
        # annotation_img = annotation
        # plate_img_objects = plate_objects
    if use_bg is False and use_brightening is True:
        plate_img = augment_data.make_random_brightening(augment_data, plate_img, max_bright_value)
    rnd_gaussian = random.randint(1, 3)
    if rnd_gaussian == 1:
        gausian_filter_size = 3
        plate_img = cv2.blur(plate_img, (gausian_filter_size, gausian_filter_size))
    plate_img_h, plate_img_w, _ = plate_img.shape
    # plate = plate_generator.fill_background(plate)
    index_with_offset = index + out_offset
    if index_with_offset % args.img_per_package == 0:

        package_counter += 1



    if plate_img_objects is not None:
        cv2.imwrite(os.path.join(images_path, f'{index_with_offset:05}.jpg'), plate_img)
        cv2.imwrite(os.path.join(annotation_path, f'{index_with_offset:05}.png'), annotation_img)

        pascal_voc.bounding_boxs_to_xml(os.path.join(pascal_annotation_path, f'{index_with_offset:05}.xml'),
                                        plate_img_w, plate_img_h, plate_img_objects, True)

        # cv2.imshow('plate', plate_img)
        # cv2.waitKey()
    else:
        continue
    index += 1
# for index in range(package_counter):
#     # pascal voc format
#     input_address = os.path.join(args.output_directory, f'{index:02}/anns')
#     bounding_rects_to_xml(input_address + '/*.png', os.path.join(input_address, 'xmls'), assets.annotations_config)
