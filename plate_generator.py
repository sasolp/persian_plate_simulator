import random
import cv2
import numpy
import pascal_voc
from asset_manager import AssetManager

which_lable = 'curr_lable'
def elementwise_add(t1, t2):
    return tuple([sum(i) for i in zip(t1, t2)])


def alpha_blend(src: numpy.ndarray, temp: numpy.ndarray, point, rand_alpha) -> numpy.ndarray:

    alpha_temp = (temp[:, :, 3] * rand_alpha) / 255.0
    alpha_n_temp = 1.0 - alpha_temp

    for ch in range(0, 3):
        src[point[0]:point[0] + temp.shape[0], point[1]:point[1] + temp.shape[1], ch] = (
                alpha_temp * temp[:, :, ch] + alpha_n_temp * src[point[0]:point[0] + temp.shape[0],
                                                             point[1]:point[1] + temp.shape[1], ch])
    return src


def annotation_blend(src: numpy.ndarray, temp: numpy.ndarray, point) -> numpy.ndarray:
    src[point[0]:point[0] + temp.shape[0], point[1]:point[1] + temp.shape[1]] = numpy.maximum(
        temp[:, :], src[point[0]:point[0] + temp.shape[0], point[1]:point[1] + temp.shape[1]])
    return src

def create_plate_object(offset: numpy.ndarray, plate_obj_h, plate_obj_w, lable):
    current_plate_object = pascal_voc.PlateObject.__new__(pascal_voc.PlateObject)
    current_plate_object.bounding_box = pascal_voc.BoundBox(offset[1], offset[0], plate_obj_w, plate_obj_h)
    current_plate_object.lable = lable
    plate_obj_points = []
    plate_obj_points.append(pascal_voc.ObjectPoint(offset[1], offset[0]))
    plate_obj_points.append(pascal_voc.ObjectPoint(offset[1] + plate_obj_w - 1, offset[0]))
    plate_obj_points.append(pascal_voc.ObjectPoint(offset[1] + plate_obj_w - 1, offset[0] + plate_obj_h - 1))
    plate_obj_points.append(pascal_voc.ObjectPoint(offset[1], offset[0] + plate_obj_h - 1))
    current_plate_object.corners = plate_obj_points
    return current_plate_object


class PlateGenerator:
    def __init__(self, asset_manager: AssetManager):
        self.asset_manager = asset_manager

    def fill_background(self, plate: numpy.ndarray) -> numpy.ndarray:
        bg = self.asset_manager.get_nxt_background()
        plate = alpha_blend(bg, plate, ((bg.shape[0] - plate.shape[0]) // 2, (bg.shape[1] - plate.shape[1]) // 2))
        return plate

    def get_next_background(self, size) -> numpy.ndarray:
        bg = self.asset_manager.get_nxt_background()
        bg_resized = cv2.resize(bg, size)
        bg_image = numpy.ones((size[0], size[1], 4), numpy.uint8) * 255
        bg_image[:, :, 0] = bg_resized[:, :, 0]
        bg_image[:, :, 1] = bg_resized[:, :, 1]
        bg_image[:, :, 2] = bg_resized[:, :, 2]
        return bg_image

    def get_rnd_plate(self, apply_dirt=False, apply_misc_noise=False) -> (numpy.ndarray, numpy.ndarray, numpy.ndarray):
        annotation_info = self.asset_manager.annotation_lables
        tsh_offsets = [[87, 265],[87, 504], [87, 583], [87, 661], [87, 739]]
        plate_objects = []
        plate_object_index = 0

        gaf_offset_top_x1 = 710
        gaf_offset_top_y1 = 43
        gaf_offset_bottom_x1 = 760
        gaf_offset_bottom_y1 = 132
        rand_alpha = random.randint(150, 255) / 255
        gaf_offsets = [[43,697 ], [43,737 ], [132, 655], [132, 690], [132, 742], [132, 777]]

        plate, annotation, plate_index = self.asset_manager.get_rnd_raw_plate()
        plate_h, plate_w, _ = plate.shape
        plate_objects.append(create_plate_object([0,0], plate_h, plate_w, 'Plate'))
        be_revert = False
        if plate_index in self.asset_manager.revert_plates:
            be_revert = True
        if plate_index != self.asset_manager.tsh_index:
            for index, (number, number_annotation, number_index) in enumerate(self.asset_manager.get_rnd_numbers()):
                if be_revert:
                    planes = cv2.split(number)
                    planes[0] = ~planes[0]
                    planes[1] = ~planes[1]
                    planes[2] = ~planes[2]
                    cv2.merge(planes, number)
                offset1 = tuple(self.asset_manager.plate_config['numbers_offset'][index])
                offset2 = (-number.shape[0] // 2, -number.shape[1] // 2)
                offset = elementwise_add(offset1, offset2)

                plate_obj_h, plate_obj_w, _ = number.shape
                plate_objects.append(create_plate_object(offset, plate_obj_h, plate_obj_w,
                                                         annotation_info[number_index + 28][which_lable]))

                alpha_blend(plate, number, offset, rand_alpha)

                annotation_blend(annotation, number_annotation, offset)
        else:
            tsh_counter = 0
            for index, (number, number_annotation, number_index) in enumerate(self.asset_manager.get_rnd_numbers()):
                if be_revert:
                    planes = cv2.split(number)
                    planes[0] = ~planes[0]
                    planes[1] = ~planes[1]
                    planes[2] = ~planes[2]
                    cv2.merge(planes, number)
                offset1 = tuple(tsh_offsets[index + 1])
                offset2 = (-number.shape[0] // 2, -number.shape[1] // 2)
                offset = elementwise_add(offset1, offset2)
                alpha_blend(plate, number, offset, rand_alpha)
                plate_obj_h, plate_obj_w, _ = number.shape
                plate_objects.append(create_plate_object(offset, plate_obj_h, plate_obj_w,
                                                         annotation_info[number_index + 28][which_lable]))
                annotation_blend(annotation, number_annotation, offset)
                tsh_counter += 1
                if tsh_counter == 4:
                    break

        if plate_index == self.asset_manager.gaf_index:
            gaf_mini_numbers = self.asset_manager.get_rnd_gaf_mini_numbers()
            gaf_miniNumbers = len(gaf_mini_numbers)
            gaf_mini_number_offset_index = 0
            top_bottom_rnd = random.randint(0, 1)
            for index, (gaf_mini_number, gaf_mini_number_annotation, gaf_mini_number_index) in enumerate(gaf_mini_numbers):
                offset1 = tuple(gaf_offsets[index + gaf_mini_number_offset_index])
                if gaf_miniNumbers == 4 and index == 0:
                    offset1 = tuple([gaf_offset_top_y1, gaf_offset_top_x1])
                    gaf_mini_number_offset_index += 1
                elif gaf_miniNumbers == 4 and index == 3:
                    offset1 = tuple([gaf_offset_bottom_y1, gaf_offset_bottom_x1])
                elif gaf_miniNumbers == 5 and index == 0 and top_bottom_rnd == 0:
                    offset1 = tuple([gaf_offset_top_y1, gaf_offset_top_x1])
                    gaf_mini_number_offset_index += 1
                elif gaf_miniNumbers == 5 and index == 4 and top_bottom_rnd == 1:
                    offset1 = tuple([gaf_offset_bottom_y1, gaf_offset_bottom_x1])

                offset2 = (-gaf_mini_number.shape[0] // 2, -gaf_mini_number.shape[1] // 2)
                offset = elementwise_add(offset1, offset2)
                plate_obj_h, plate_obj_w, _ = gaf_mini_number.shape
                plate_objects.append(create_plate_object(offset, plate_obj_h, plate_obj_w,
                                                         annotation_info[gaf_mini_number_index + 27][which_lable]))

                alpha_blend(plate, gaf_mini_number, offset, rand_alpha)
                annotation_blend(annotation, gaf_mini_number_annotation, offset)
        elif plate_index != self.asset_manager.tsh_index:
            for index, (mini_number, mini_number_annotation, mini_number_index) in enumerate(self.asset_manager.get_rnd_mini_numbers()):
                if be_revert:
                    planes = cv2.split(mini_number)
                    planes[0] = ~planes[0]
                    planes[1] = ~planes[1]
                    planes[2] = ~planes[2]
                    cv2.merge(planes, mini_number)
                offset1 = tuple(self.asset_manager.plate_config['mini_numbers_offset'][index])
                offset2 = (-mini_number.shape[0] // 2, -mini_number.shape[1] // 2)
                offset = elementwise_add(offset1, offset2)

                plate_obj_h, plate_obj_w, _ = mini_number.shape
                plate_objects.append(create_plate_object(offset, plate_obj_h, plate_obj_w,
                                                         annotation_info[mini_number_index + 27][which_lable]))
                alpha_blend(plate, mini_number, offset, rand_alpha)
                annotation_blend(annotation, mini_number_annotation, offset)

        letter, letter_annotation, letter_index = self.asset_manager.get_rnd_letter(plate_index)
        offset1 = tuple(self.asset_manager.plate_config['letter_offset'])
        if plate_index == self.asset_manager.tsh_index:
            offset1 = tuple(tsh_offsets[0])
        offset2 = (-letter.shape[0] // 2, -letter.shape[1] // 2)
        offset = elementwise_add(offset1, offset2)

        plate_obj_h, plate_obj_w, _ = letter.shape
        plate_objects.append(create_plate_object(offset, plate_obj_h, plate_obj_w,
                                                 annotation_info[letter_index][which_lable]))
        alpha_blend(plate, letter, offset, rand_alpha)
        annotation_blend(annotation, letter_annotation, offset)

        if apply_misc_noise:
            for noise_box in self.asset_manager.noise_config['misc_bounds']:
                if random.uniform(0, 1) < self.asset_manager.noise_config['misc_probability']:
                    misc = self.asset_manager.get_rnd_misc_noise()
                    x_location = random.randint(noise_box['x_min'], noise_box['x_max'])
                    y_location = random.randint(noise_box['y_min'], noise_box['y_max'])
                    alpha_blend(plate, misc, (x_location - misc.shape[0] // 2, y_location - misc.shape[1] // 2), rand_alpha)

        if apply_dirt:
            if random.uniform(0, 1) < self.asset_manager.noise_config['dirt_probability']:
                dirt = self.asset_manager.get_rnd_dirt()
                alpha_blend(plate, dirt, (0, 0), rand_alpha)
        rnd_white_bounding_box = random.randint(1, 3)
        if rnd_white_bounding_box == 1:
            white_tickness = random.randint(10, 15)
            cv2.rectangle(plate, (0,0), (plate_w - 1, plate_h - 1), (255, 255, 255, 255), white_tickness);
        return plate, annotation, plate_objects
