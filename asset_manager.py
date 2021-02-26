import glob
import json
import random
import warnings

import cv2
import numpy
from tqdm import tqdm


class AssetManager:
    __bg_counter: int

    def __init__(self, config_path, log_warning=False):
        self.log_warning = log_warning
        with open(config_path, 'r') as f:
        # no need to change the code, just tweak the config file :)
            project_config_file = json.load(f)
            # project_config_file = project_config_file.encode('utf-8')
            self.annotation_lables = []
            self.__backgrounds = []
            self.revert_plates = []
            self.tsh_index = 0
            self.gaf_index = 0
            self.__bg_counter = 0
            self.__bg_pack = 0
            self.__bg_addresses = glob.glob(project_config_file['components']['backgrounds'])
            self.annotations_config = project_config_file['annotations_config']
            self.ann_step = 255 / (len(self.annotations_config) + 1)
            self.__init_components(project_config_file)
            self.plate_config = project_config_file['plate_config']
            self.noise_config = project_config_file['noise_config']
            self.generator_config = project_config_file['generator_config']
            self.__bg_limit = project_config_file['generator_config']['background_load_limit']
            self.__init_backgrounds()
            self.transformations_config = project_config_file['transformations_config']

    def __init_components(self, project_config_file) -> None:
        print('\ninit components...')
        component_addresses = {}
        for item in project_config_file['components']:
            component_addresses[item] = glob.glob(project_config_file['components'][item])
        self.__components = {}
        self.__annotations = {}
        for item in component_addresses:
            self.__components[item] = []
            self.__annotations[item] = []
            if item != 'backgrounds':
                index = 0
                for file_name in component_addresses[item]:
                    image = cv2.imread(file_name , -1)
                    self.__components[item].append(image)
                    key = file_name.split('\\')[-1].replace('.png', '').split('_')[0]
                    print(key)
                    key = key.upper()
                    if item == 'letters' and key == 'TSH':
                        self.tsh_index = index
                    if item == 'letters' and key == 'GAF':
                        self.gaf_index = index
                    if item == 'letters' and \
                            (key == 'ALEF' or key == 'TSH' or key == 'F' or key == 'Z' or key == 'P' or key == 'TH'):
                        self.revert_plates.append(index)
                    if item == 'letters' or item == 'mini_numbers':
                        self.annotation_lables.append(project_config_file['annotations_config'][key])
                    index = index + 1
                    if key in project_config_file['annotations_config']:
                        annotation_value = numpy.uint8(project_config_file['annotations_config'][key]['curr_lable'])
                        annotation = (numpy.array(image[:, :, 3]) > 192).astype(numpy.uint8)
                        annotation = numpy.multiply(annotation, numpy.uint8(min(255, (annotation_value + 1) * self.ann_step)))
                        self.__annotations[item].append(annotation)
                    else:
                        if self.log_warning:
                            warnings.warn(
                                f'key: {key} for {file_name} not found in project_config ->'
                                f' annotations_config.\n Annotation skipped.')

    def __init_backgrounds(self):
        if self.__bg_limit * self.__bg_pack >= len(self.__bg_addresses):
            self.__bg_pack = 0
        self.__bg_counter = 0
        if self.__bg_limit == len(self.__bg_addresses) and len(self.__backgrounds) != 0:
            return
        print(f'\nloading backgrounds (pack {self.__bg_pack})...')
        if len(self.__backgrounds) == 0:
            del self.__backgrounds
        self.__backgrounds = []
        progress = tqdm(self.__bg_addresses[self.__bg_limit * self.__bg_pack:self.__bg_limit * (self.__bg_pack + 1)])
        for item in progress:
            self.__backgrounds.append(cv2.imread(item))
        self.__bg_pack += 1

    def get_bg_image_addresses(self):
        return self.__bg_addresses

    def get_rnd_raw_plate(self) -> (numpy.ndarray, numpy.ndarray, int):
        index = random.randint(0, len(self.__components['plates']) - 1)
        # index = 23
        # print(index)
        # print(len(self.__annotations))
        # print(len(self.__components))
        return self.__components['plates'][index].copy(), self.__annotations['plates'][index].copy() * 0, index

    def get_rnd_number(self) -> (numpy.ndarray, numpy.ndarray, int):
        index = random.randint(0, len(self.__components['numbers']) - 1)
        return self.__components['numbers'][index].copy(), self.__annotations['numbers'][index].copy(), index

    def get_rnd_mini_number(self, include_zero=False) -> (numpy.ndarray, numpy.ndarray, int):
        start = (1, 0)[include_zero]  # awkwardly hardcoded but works fine till '0' is the first element!
        index = random.randint(start, len(self.__components['mini_numbers']) - 1)
        return self.__components['mini_numbers'][index].copy(), self.__annotations["mini_numbers"][index].copy(), index
    def get_rnd_gaf_mini_number(self, include_zero=False) -> (numpy.ndarray, numpy.ndarray, int):
        start = (1, 0)[include_zero]  # awkwardly hardcoded but works fine till '0' is the first element!
        index = random.randint(start, len(self.__components['gaf_mini_numbers']) - 1)
        return self.__components['gaf_mini_numbers'][index].copy(), self.__annotations["gaf_mini_numbers"][index].copy(), index


    def get_rnd_letter(self, plate_index) -> (numpy.ndarray, numpy.ndarray, int):
        index = plate_index #random.randint(0, len(self.__components['letters']) - 1)
        return self.__components['letters'][index].copy(), self.__annotations['letters'][index].copy(), index

    def get_rnd_numbers(self) -> [(numpy.ndarray, numpy.ndarray, int)]:
        numbers = []
        for _ in range(5):
            numbers.append(self.get_rnd_number())
        return numbers

    def get_rnd_mini_numbers(self) -> [(numpy.ndarray, numpy.ndarray, int)]:
        mini_numbers = [self.get_rnd_mini_number(), self.get_rnd_mini_number(include_zero=True)]
        return mini_numbers
    def get_rnd_gaf_mini_numbers(self) -> [(numpy.ndarray, numpy.ndarray, int)]:
        gaf_mini_numbers = []
        counts = [4, 5, 6]
        count_index = random.randint(0, 2)

        for _ in range(counts[count_index]):
            gaf_mini_numbers.append(self.get_rnd_gaf_mini_number())
        return gaf_mini_numbers
    def get_rnd_dirt(self) -> numpy.ndarray:
        return random.choice(self.__components['dirt']).copy()

    def get_rnd_misc_noise(self) -> numpy.ndarray:
        misc = random.choice(self.__components['misc']).copy()
        return misc

    def get_nxt_background(self) -> numpy.ndarray:
        if self.__bg_counter == len(self.__backgrounds):
            self.__init_backgrounds()
        # print(self.__backgrounds)
        # print(self.__bg_counter)
        image = self.__backgrounds[self.__bg_counter]
        self.__bg_counter += 1
        return image.copy()
