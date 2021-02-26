import random
import pascal_voc
import cv2
import numpy
import numpy as np

class AugmentData():
    def init(self):
        self.gaus_filter_sizes = [1, 1, 3, 1, 1, 5, 1, 1, 7, 1, 1, 9, 1, 1, 11, 1, 1, 13]
        self.jpeg_qualities = [100, 100, 90, 100, 100, 70, 100, 100, 50]
        self.down_scales = [1, 1, 0.75, 1, 1, 0.5, 1, 1, 0.25]
        self.random_uniform = [10, 10, 20, 30, 40, 50, 60, 70, 80]

        self.gaus_filter_sizes_counter = 0
        self.jpeg_qualities_counter = 0
        self.down_scales_counter = 0
        self.random_uniform_counter = 0

        self.gaus_filter_sizes_count = len(self.gaus_filter_sizes)
        self.jpeg_qualities_count = len(self.jpeg_qualities)
        self.down_scales_count = len(self.down_scales)
        self.random_uniform_count = len(self.random_uniform)
        self.general_index = 0

    def adjust_gamma(self, image, gamma=1.5):
        try:
            invGamma = gamma
            if(invGamma != 0):
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            else:
                table = np.array([(i / 255.0) * 255 for i in np.arange(0, 256)]).astype("uint8")

        except:
            table = np.array([(i / 255.0) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)


    def make_random_brightening(self, image, max_bright_value):
        # cv2.imshow('10', image)
        self.random_uniform_counter = self.general_index % self.random_uniform_count
        self.general_index += 1
        rnd_uni = self.random_uniform[self.random_uniform_counter]
        image_h, image_w, _ = image.shape
        adjust_image = image[:, :, 0:3].copy()
        gamma = random.randrange(int(max_bright_value / 2), max_bright_value * 2, 1)
        gamma = gamma / max_bright_value
        adjust_image = self.adjust_gamma(self, image=adjust_image, gamma=gamma)
        image[:, :, 0] = adjust_image[:, :, 0]
        image[:, :, 1] = adjust_image[:, :, 1]
        image[:, :, 2] = adjust_image[:, :, 2]
        # cv2.imshow('1', image)
        # cv2.waitKey()
        return image


    def make_random_hsv(self, image, max_bright_value):
        self.random_uniform_counter = self.general_index % self.random_uniform_count
        self.general_index += 1
        rnd_uni = self.random_uniform[self.random_uniform_counter]
        image_h, image_w, _ = image.shape
        noised_image = numpy.zeros((image_w, image_h), numpy.uint8)
        noised_image = cv2.randu(noised_image, 0, rnd_uni)

        x = random.randint(0, 3)
        if x == 0:
            image[:, :, 0] = numpy.uint8(image[:, :, 0] + noised_image  )
        elif x == 1:
            image[:, :, 1] = numpy.uint8(image[:, :, 1] + noised_image  )
        elif x== 2:
            image[:, :, 2] = numpy.uint8(image[:, :, 2] + noised_image )


        cv2.imshow('1', image)
        cv2.waitKey()
        return image

    def make_random_distortion(self, image: numpy.ndarray):

        self.gaus_filter_sizes_counter = self.general_index % self.gaus_filter_sizes_count
        self.jpeg_qualities_counter = self.general_index % self.jpeg_qualities_count
        self.down_scales_counter = self.general_index % self.down_scales_count
        self.random_uniform_counter = self.general_index % self.random_uniform_count

        self.general_index += 1
        image_h, image_w, _ = image.shape

        gray_image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY);
        sobelx_image = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
        sobely_image = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
        gradient_image = cv2.add(abs(sobelx_image), abs(sobely_image))
        gradient = cv2.sumElems(gradient_image)[0] / (image_w * image_h)

        MAX_FILTER_SIZE_TH = 0.05  # based on statistical analisis
        MIN_IMAGE_WIDTH = 85
        MEAN_NORM_VALUE = 80
        MIN_GRADIENT_TO_FILTER1 = 100
        MIN_GRADIENT_TO_FILTER2 = 200


        gaus_filter_size = self.gaus_filter_sizes[self.gaus_filter_sizes_counter]
        down_scale = self.down_scales[self.down_scales_counter]

        while gaus_filter_size > 0 and gaus_filter_size / image_w > MAX_FILTER_SIZE_TH:  # gaus_filter_size must be Odd
            gaus_filter_size = gaus_filter_size - 2
        if gaus_filter_size < 0 or gradient < MIN_GRADIENT_TO_FILTER1:
            gaus_filter_size = 1
        elif gradient < MIN_GRADIENT_TO_FILTER2:
            gaus_filter_size = 3
        while image_w * down_scale < MIN_IMAGE_WIDTH:
            down_scale += 0.01
        if down_scale > 1:
            down_scale = 1
        jpeg_quality = self.jpeg_qualities[self.jpeg_qualities_counter]
        rnd_uni = self.random_uniform[self.random_uniform_counter]
        noised_image = (image[:, :, 0:3]).copy()
        cv2.randu(noised_image, 0, rnd_uni)
        noised_image = image[:, :, 0:3] + noised_image

        blured_image = cv2.GaussianBlur(noised_image, (gaus_filter_size, gaus_filter_size), cv2.BORDER_DEFAULT)

        blured_image_h, blured_image_w, _ = blured_image.shape
        down_scaled_image = cv2.resize(blured_image, (int(blured_image_w * down_scale), int(blured_image_h * down_scale)), interpolation=cv2.INTER_CUBIC)
        up_scaled_image = cv2.resize(down_scaled_image, (blured_image_w, blured_image_h), interpolation=cv2.INTER_CUBIC)
        encoded_image_data = cv2.imencode('.jpg', up_scaled_image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        encoded_image = cv2.imdecode(np.array(encoded_image_data[1]), cv2.IMREAD_COLOR)

        do_aug = random.randint(0,2)
        if do_aug == 1:
            aug_image = numpy.ones((image_h, image_w, 3), numpy.uint8)
            aug_image[:, :, 0] = gradient_image * 0.2
            aug_image[:, :, 1] = gradient_image * 0.2
            aug_image[:, :, 2] = gradient_image * 0.2
            encoded_image = cv2.add(encoded_image, aug_image)
        final_image = numpy.ones((image_h, image_w, 4), numpy.uint8)

        final_image[:, :, 0] = encoded_image[:, :, 0]
        final_image[:, :, 1] = encoded_image[:, :, 1]
        final_image[:, :, 2] = encoded_image[:, :, 2]
        final_image[:, :, 3] = final_image[:, :, 3] * 255
        # cv2.imshow('0', gray_image);cv2.waitKey();
        # cv2.namedWindow('1', cv2.WINDOW_KEEPRATIO);
        # cv2.namedWindow('2', cv2.WINDOW_KEEPRATIO);
        # cv2.namedWindow('3', cv2.WINDOW_KEEPRATIO);
        # cv2.namedWindow('4', cv2.WINDOW_KEEPRATIO);
        # cv2.namedWindow('5', cv2.WINDOW_KEEPRATIO);
        # cv2.imshow('1', noised_image);
        # cv2.imshow('2', blured_image);
        # cv2.imshow('3', down_scaled_image);
        # cv2.imshow('4', final_image);
        # cv2.imshow('5', rotated_image);
        # cv2.waitKey();
        return final_image
def perspective_transform(image: numpy.ndarray, annotation: numpy.ndarray, config, ann_step, plate_objects, ff) -> [numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    max_dalpha = config['max_dalpha'] / 180 * numpy.pi
    max_dbeta = config['max_dbeta'] / 180 * numpy.pi
    max_dgamma = config['max_dgamma'] / 180 * numpy.pi
    max_dalpha *= 0.5
    max_dbeta *= 0.5
    max_dgamma *= 1

    alpha = random.uniform(-max_dalpha, max_dalpha)
    beta = random.uniform(-max_dbeta, max_dbeta)
    gamma = random.uniform(-max_dgamma, max_dgamma)
    h = image.shape[0]
    w = image.shape[1]
    max_w = w
    min_w = config['min_plate_width']
    transformed_width = config['output_size'][0]
    transformed_height = config['output_size'][1]
    f = 1000
    ff = ff % min(transformed_width / max_w * ((0.75 * max_w) / min_w), config['max_dz'])
    dz = int(f * (1 + ff)) #f * random.uniform(1, config['max_dz'])
    tr_x = 416#ff * 3
    tr_y = 416#ff * 5
    a_1 = numpy.array([[1, 0, -w / 2],
                       [0, 1, -h / 2],
                       [0, 0, 0],
                       [0, 0, 1]])
    r_x = numpy.array([[1, 0, 0, 0],
                       [0, numpy.cos(alpha), -numpy.sin(alpha), 0],
                       [0, numpy.sin(alpha), numpy.cos(alpha), 0],
                       [0, 0, 0, 1]])
    r_y = numpy.array([[numpy.cos(beta), 0, -numpy.sin(beta), 0],
                       [0, 1, 0, 0],
                       [numpy.sin(beta), 0, numpy.cos(beta), 0],
                       [0, 0, 0, 1]])
    r_z = numpy.array([[numpy.cos(gamma), -numpy.sin(gamma), 0, 0],
                       [numpy.sin(gamma), numpy.cos(gamma), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    r = numpy.matmul(numpy.matmul(r_x, r_y), r_z)
    t = numpy.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, dz],
                     [0, 0, 0, 1]])
    a_2 = numpy.array([[f, 0, tr_x, 0],
                       [0, f, tr_y, 0],
                       [0, 0, 1, 0]])

    trans = numpy.matmul(a_2, numpy.matmul(t, numpy.matmul(r, a_1)))

    image = cv2.warpPerspective(image, trans, (config['output_size'][0], config['output_size'][1]), flags=cv2.INTER_CUBIC)
    annotation = cv2.warpPerspective(annotation, trans, (config['output_size'][0], config['output_size'][1]), flags=cv2.INTER_CUBIC)
    # bbox_list = []
    # thresh = annotation.copy()
    # thresh = (thresh > 0).astype(numpy.uint8)*255
    # kernel = numpy.ones((3, 3), numpy.uint8)
    # thresh = cv2.erode(thresh, kernel)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # bbox_list = {}
    # for cnt in contours:
    #     if cv2.contourArea(cnt) < 2:
    #         continue
    #     x, y, w, h = cv2.boundingRect(cnt)
    #
    #     mean_val = 0
    #     for pixel in cnt:
    #         mean_val += annotation[pixel[0][1]][pixel[0][0]]
    #     mean_val = round(mean_val / len(cnt))
    #     class_id = round(mean_val / ann_step)
    #     if class_id in bbox_list:
    #         bbox_list[class_id] = (min(x, bbox_list[class_id][0]), min(y, bbox_list[class_id][1]) , max(x+w, bbox_list[class_id][2]), max(y+h, bbox_list[class_id][3]))
    #     else:
    #         bbox_list[class_id] = (x, y , x + w, y + h)
    for obj in plate_objects:
        src_points = []
        for pnt in obj.corners:
            src_points.append([pnt.x, pnt.y])
        dst_points = cv2.perspectiveTransform(numpy.float32(src_points).reshape(-1, 1, 2), trans)
        obj.corners.clear()
        for pnt in dst_points:
            new_pnt = pascal_voc.ObjectPoint.__new__(pascal_voc.ObjectPoint)
            new_pnt.x = pnt[0][0]
            new_pnt.y = pnt[0][1]
            obj.corners.append(new_pnt)
        obj.bounding_box.x = min([elm.x for elm in obj.corners])
        obj.bounding_box.y = min([elm.y for elm in obj.corners])
        obj.bounding_box.width = max([elm.x for elm in obj.corners]) - obj.bounding_box.x
        obj.bounding_box.height = max([elm.y for elm in obj.corners]) - obj.bounding_box.y
        if obj.lable is not 'Plate':
            obj_center = [obj.bounding_box.x + obj.bounding_box.width * 0.5,
                          obj.bounding_box.y + obj.bounding_box.height * 0.5]
            thresh = (annotation == numpy.uint8 ((numpy.uint8(obj.lable) + 1) * ann_step)).astype(numpy.uint8)
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt_counter = 0
            for cnt in contours[0]:
                cnt_counter += 1
                if cv2.contourArea(cnt) < 5:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                box_center = [x + w * .5, y + h * .5]
                dist = numpy.sqrt((box_center[0] - obj_center[0]) * (box_center[0] - obj_center[0]) +\
                       (box_center[1] - obj_center[1]) * (box_center[1] - obj_center[1]))
                if dist > obj.bounding_box.width *0.5:
                    continue
                if cnt_counter == 1:
                    obj.bounding_box.x = x
                    obj.bounding_box.y = y
                    obj.bounding_box.width = w
                    obj.bounding_box.height = h
                xmax = obj.bounding_box.x + obj.bounding_box.width
                ymax = obj.bounding_box.y + obj.bounding_box.height
                if  xmax < x + w:
                    xmax = x + w
                if ymax < y + h:
                    ymax = y + h
                obj.bounding_box.x = min(x, obj.bounding_box.x)
                obj.bounding_box.y = min(y, obj.bounding_box.y)
                obj.bounding_box.width = xmax - obj.bounding_box.x
                obj.bounding_box.height = ymax - obj.bounding_box.y
        # box_index = -1
        # box_counter = 0
        # box_min_dist = 10000000
        # for box in bbox_list:
        #     dist = (box[0] - obj_center[0]) * (box[0] - obj_center[0]) +\
        #            (box[1] - obj_center[1]) * (box[1] - obj_center[1])
        #     if dist < box_min_dist:
        #         box_min_dist = dist
        #         box_index = box_counter
        #     box_counter += 1
        # obj.bounding_box.x        = bbox_list[box_index][0]
        # obj.bounding_box.y        = bbox_list[box_index][1]
        # obj.bounding_box.width    = bbox_list[box_index][2]
        # obj.bounding_box.height   = bbox_list[box_index][3]
    # cv2.circle(image, (config['output_size'][0] // 2, config['output_size'][1]//2), 5, (25,255,25), 5)
    cv2.circle(image, (tr_x, tr_y), 5, (25,255,25), 5)
    # cv2.namedWindow('plate', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('plate', image)
    cv2.waitKey()
    return image, annotation, plate_objects

def perspective_transform_ocr(image: numpy.ndarray, annotation: numpy.ndarray, config, ann_step, plate_objects, ff) -> [numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    max_dalpha = config['max_dalpha'] / 180 * numpy.pi
    max_dbeta = config['max_dbeta'] / 180 * numpy.pi
    max_dgamma = config['max_dgamma'] / 180 * numpy.pi
    max_dalpha *= 0.5
    max_dbeta *= 0.5
    max_dgamma *= 1

    alpha = random.uniform(-max_dalpha, max_dalpha)
    beta = random.uniform(-max_dbeta, max_dbeta)
    gamma = random.uniform(-max_dgamma, max_dgamma)
    h = image.shape[0]
    w = image.shape[1]
    max_w = w
    min_w = config['min_plate_width']
    transformed_width = w
    transformed_height = h
    f = 1000
    ff = ff % 2
    ff = ff % min(transformed_width / max_w * ((0.75 * max_w) / min_w), config['max_dz'])
    dz = int(f * (1 + ff)) #f * random.uniform(1, config['max_dz'])
    tr_x = 400#ff * 3
    tr_y = 88#ff * 5
    a_1 = numpy.array([[1, 0, -w / 2],
                       [0, 1, -h / 2],
                       [0, 0, 0],
                       [0, 0, 1]])
    r_x = numpy.array([[1, 0, 0, 0],
                       [0, numpy.cos(alpha), -numpy.sin(alpha), 0],
                       [0, numpy.sin(alpha), numpy.cos(alpha), 0],
                       [0, 0, 0, 1]])
    r_y = numpy.array([[numpy.cos(beta), 0, -numpy.sin(beta), 0],
                       [0, 1, 0, 0],
                       [numpy.sin(beta), 0, numpy.cos(beta), 0],
                       [0, 0, 0, 1]])
    r_z = numpy.array([[numpy.cos(gamma), -numpy.sin(gamma), 0, 0],
                       [numpy.sin(gamma), numpy.cos(gamma), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    r = numpy.matmul(numpy.matmul(r_x, r_y), r_z)
    t = numpy.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, dz],
                     [0, 0, 0, 1]])
    a_2 = numpy.array([[f, 0, tr_x, 0],
                       [0, f, tr_y, 0],
                       [0, 0, 1, 0]])

    trans = numpy.matmul(a_2, numpy.matmul(t, numpy.matmul(r, a_1)))

    src_points = []
    src_points.append([0, 0])
    src_points.append([0, h - 1])
    src_points.append([w - 1, 0])
    src_points.append([w - 1, h - 1])
    dst_points = cv2.perspectiveTransform(numpy.float32(src_points).reshape(-1, 1, 2), trans)
    corners = []
    bounding_box = pascal_voc.BoundBox
    for pnt in dst_points:
        new_pnt = pascal_voc.ObjectPoint.__new__(pascal_voc.ObjectPoint)
        new_pnt.x = pnt[0][0]
        new_pnt.y = pnt[0][1]
        corners.append(new_pnt)
    bounding_box.x = min([elm.x for elm in corners])
    bounding_box.y = min([elm.y for elm in corners])
    bounding_box.width = max([elm.x for elm in corners]) - bounding_box.x
    bounding_box.height = max([elm.y for elm in corners]) - bounding_box.y

    a_2 = numpy.array([[f, 0, bounding_box.width * 0.55, 0],
                       [0, f, bounding_box.height * 0.55, 0],
                       [0, 0, 1, 0]])

    trans = numpy.matmul(a_2, numpy.matmul(t, numpy.matmul(r, a_1)))

    image = cv2.warpPerspective(image, trans, (int(bounding_box.width * 1.1), int(bounding_box.height * 1.1)), flags=cv2.INTER_CUBIC)
    annotation = cv2.warpPerspective(annotation, trans, (int(bounding_box.width * 1.1), int(bounding_box.height * 1.1)), flags=cv2.INTER_CUBIC)

    for obj in plate_objects:
        src_points = []
        for pnt in obj.corners:
            src_points.append([pnt.x, pnt.y])
        dst_points = cv2.perspectiveTransform(numpy.float32(src_points).reshape(-1, 1, 2), trans)
        obj.corners.clear()
        for pnt in dst_points:
            new_pnt = pascal_voc.ObjectPoint.__new__(pascal_voc.ObjectPoint)
            new_pnt.x = pnt[0][0]
            new_pnt.y = pnt[0][1]
            obj.corners.append(new_pnt)
        obj.bounding_box.x = min([elm.x for elm in obj.corners])
        obj.bounding_box.y = min([elm.y for elm in obj.corners])
        obj.bounding_box.width = max([elm.x for elm in obj.corners]) - obj.bounding_box.x
        obj.bounding_box.height = max([elm.y for elm in obj.corners]) - obj.bounding_box.y
        if obj.lable is not 'Plate':
            obj_center = [obj.bounding_box.x + obj.bounding_box.width * 0.5,
                          obj.bounding_box.y + obj.bounding_box.height * 0.5]
            thresh = (annotation == numpy.uint8 ((numpy.uint8(obj.lable) + 1) * ann_step)).astype(numpy.uint8)
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt_counter = 0
            for cnt in contours[0]:
                cnt_counter += 1
                if cv2.contourArea(cnt) < 5:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                box_center = [x + w * .5, y + h * .5]
                dist = numpy.sqrt((box_center[0] - obj_center[0]) * (box_center[0] - obj_center[0]) +\
                       (box_center[1] - obj_center[1]) * (box_center[1] - obj_center[1]))
                if dist > obj.bounding_box.width *0.5:
                    continue
                if cnt_counter == 1:
                    obj.bounding_box.x = x
                    obj.bounding_box.y = y
                    obj.bounding_box.width = w
                    obj.bounding_box.height = h
                xmax = obj.bounding_box.x + obj.bounding_box.width
                ymax = obj.bounding_box.y + obj.bounding_box.height
                if  xmax < x + w:
                    xmax = x + w
                if ymax < y + h:
                    ymax = y + h
                obj.bounding_box.x = min(x, obj.bounding_box.x)
                obj.bounding_box.y = min(y, obj.bounding_box.y)
                obj.bounding_box.width = xmax - obj.bounding_box.x
                obj.bounding_box.height = ymax - obj.bounding_box.y

    ## cv2.circle(image, (tr_x, tr_y), 5, (25,255,25), 5)
    # cv2.namedWindow('plate', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('plate', image)
    # cv2.waitKey()
    return image, annotation, plate_objects
def perspective_transform_with_bg(image: numpy.ndarray, annotation: numpy.ndarray, config,
                                  ann_step, bg_instance, plate_objects) -> [numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    bg_image = cv2.imread(bg_instance['filename'])
    bg_image_h, bg_image_w, _ = bg_image.shape
    if bg_image_h == 0 or bg_image_w == 0:
        return [None, None, None]
    bg_gray = cv2.cvtColor(bg_image, cv2.COLOR_RGB2GRAY)
    plate_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plate_mean, plate_stddev = cv2.meanStdDev(plate_gray)
    bg_mean, bg_stddev = cv2.meanStdDev(bg_gray)
    # cv2.imshow('bg_image', bg_image)
    # cv2.imshow('bg_image0', image)
    diff = bg_mean - plate_mean - plate_stddev
    image[:,:,0:3] = cv2.add(image[:,:,0:3], diff)
    # cv2.imshow('bg_image1', image)
    # cv2.waitKey()
    bg_objects = bg_instance['object']
    rand_index = int(random.uniform(0, len(bg_objects)))
    bg_counter = 0
    bg_mask =[]
    bg_mask = numpy.zeros((bg_image_h, bg_image_w), numpy.uint8)
    for bg_obj in bg_objects:
        cv2.rectangle(bg_mask, (bg_obj['xmin'], bg_obj['ymin']), (bg_obj['xmax'], bg_obj['ymax']), 255, -1)
    bg_mask = cv2.dilate(bg_mask, numpy.ones((7,7), numpy.uint8))
    cv2.inpaint(bg_image, bg_mask, 7, cv2.INPAINT_TELEA, bg_image)
    # cv2.imshow('bg_mask', bg_mask)
    # cv2.imshow('bg_image', bg_image)
    # cv2.waitKey()
    bg_obj = bg_objects[rand_index]
    plate_width = (bg_obj['xmax'] - bg_obj['xmin'])*0.6
    h = image.shape[0]
    w = image.shape[1]
    transformed_width = bg_image_w
    transformed_height = bg_image_h
    cam_x = transformed_width // 2#int(random.uniform(0, transformed_width))
    tr_x = int(bg_obj['xmin'] + (bg_obj['xmax'] - bg_obj['xmin']) * 0.5)
    tr_y = int(bg_obj['ymin'] + (bg_obj['ymax'] - bg_obj['ymin']) * 0.5)
    dx0 = tr_x - cam_x
    dy0 = tr_y - transformed_height + 1
    dist_from_cam = numpy.sqrt(dx0 * dx0 + dy0 * dy0)
    max_dist = numpy.sqrt(transformed_width * transformed_width + transformed_height * transformed_height)
    w0 = plate_width
    h0 = plate_width * h / w

    dx_beta = tr_x - cam_x
    dy_beta = transformed_height - 1 - tr_y

    alpha = (dist_from_cam / (transformed_height * 1.5)) * numpy.pi / 2
    beta = (numpy.arctan2(dx_beta, dy_beta) / numpy.pi / 2 * config['max_dbeta'] / 180 * numpy.pi)
    gamma =  dx_beta / transformed_width * config['max_dgamma'] / 180 * numpy.pi
    factor = 1000
    plate_width *= (1 - dx_beta / transformed_width) + 0.65
    dz = w / plate_width * factor
    print(f'\nalpha: {alpha}\nbeta: {beta}\ngamma: {gamma}\n')
    a_1 = numpy.array([[1, 0, -w / 2],
                       [0, 1, -h / 2],
                       [0, 0, 0],
                       [0, 0, 1]])
    r_x = numpy.array([[1, 0, 0, 0],
                       [0, numpy.cos(alpha), -numpy.sin(alpha), 0],
                       [0, numpy.sin(alpha), numpy.cos(alpha), 0],
                       [0, 0, 0, 1]])
    r_y = numpy.array([[numpy.cos(beta), 0, -numpy.sin(beta), 0],
                       [0, 1, 0, 0],
                       [numpy.sin(beta), 0, numpy.cos(beta), 0],
                       [0, 0, 0, 1]])
    r_z = numpy.array([[numpy.cos(gamma), -numpy.sin(gamma), 0, 0],
                       [numpy.sin(gamma), numpy.cos(gamma), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    r = numpy.matmul(numpy.matmul(r_x, r_y), r_z)
    t = numpy.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, dz],
                     [0, 0, 0, 1]])
    a_2 = numpy.array([[factor, 0, tr_x, 0],
                       [0, factor, tr_y, 0],
                       [0, 0, 1, 0]])

    trans = numpy.matmul(a_2, numpy.matmul(t, numpy.matmul(r, a_1)))

    image = cv2.warpPerspective(image, trans, (bg_image_w, bg_image_h), flags=cv2.INTER_CUBIC)
    annotation = cv2.warpPerspective(annotation, trans, (bg_image_w, bg_image_h), flags=cv2.INTER_CUBIC)

    for obj in plate_objects:
        src_points = []
        for pnt in obj.corners:
            src_points.append([pnt.x, pnt.y])
        dst_points = cv2.perspectiveTransform(numpy.float32(src_points).reshape(-1, 1, 2), trans)
        obj.corners.clear()
        for pnt in dst_points:
            new_pnt = pascal_voc.ObjectPoint.__new__(pascal_voc.ObjectPoint)
            new_pnt.x = pnt[0][0]
            new_pnt.y = pnt[0][1]
            obj.corners.append(new_pnt)
        obj.bounding_box.x = min([elm.x for elm in obj.corners])
        obj.bounding_box.y = min([elm.y for elm in obj.corners])
        obj.bounding_box.width = max([elm.x for elm in obj.corners]) - obj.bounding_box.x
        obj.bounding_box.height = max([elm.y for elm in obj.corners]) - obj.bounding_box.y
        if obj.lable is not 'Plate':
            obj_center = [obj.bounding_box.x + obj.bounding_box.width * 0.5,
                          obj.bounding_box.y + obj.bounding_box.height * 0.5]
            thresh = (annotation == numpy.uint8 ((numpy.uint8(obj.lable) + 1) * ann_step)).astype(numpy.uint8)
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt_counter = 0
            for cnt in contours[0]:
                cnt_counter += 1
                if cv2.contourArea(cnt) < 5:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                box_center = [x + w * .5, y + h * .5]
                dist = numpy.sqrt((box_center[0] - obj_center[0]) * (box_center[0] - obj_center[0]) +\
                       (box_center[1] - obj_center[1]) * (box_center[1] - obj_center[1]))
                if dist > obj.bounding_box.width *0.5:
                    continue
                if cnt_counter == 1:
                    obj.bounding_box.x = x
                    obj.bounding_box.y = y
                    obj.bounding_box.width = w
                    obj.bounding_box.height = h
                xmax = obj.bounding_box.x + obj.bounding_box.width
                ymax = obj.bounding_box.y + obj.bounding_box.height
                if  xmax < x + w:
                    xmax = x + w
                if ymax < y + h:
                    ymax = y + h
                obj.bounding_box.x = min(x, obj.bounding_box.x)
                obj.bounding_box.y = min(y, obj.bounding_box.y)
                obj.bounding_box.width = xmax - obj.bounding_box.x
                obj.bounding_box.height = ymax - obj.bounding_box.y

    bg_ = numpy.zeros((bg_image_h, bg_image_w, 4), numpy.uint8)

    bg_[:, :, 0] = cv2.bitwise_and( bg_image[:, :, 0], ~image[:,:,3])
    bg_[:, :, 1] = cv2.bitwise_and( bg_image[:, :, 1], ~image[:,:,3])
    bg_[:, :, 2] = cv2.bitwise_and( bg_image[:, :, 2], ~image[:,:,3])
    bg_[:,:,3] = ~image[:,:,3]


    img = (image + bg_).copy()
    cv2.imshow('plate', img)
    cv2.waitKey()
    return img, annotation, plate_objects


def perspective_transform_with_tiling(image: numpy.ndarray, annotation: numpy.ndarray, config,
                                      bg_instance, use_tiling_method, scale_factor,
                                      ann_step, plate_objects, orig_image: numpy.ndarray,
                                      orig_annotation: numpy.ndarray, orig_plate_objects,
                                      last_x_offset, last_y_offset, cam_x,  plate_width) -> [numpy.ndarray, numpy.ndarray, numpy.ndarray, int, int, bool]:
    reset_new_image = False
    if use_tiling_method == 1:
        bg_image = cv2.imread(bg_instance['filename'])
        bg_image_h, bg_image_w, _ = bg_image.shape
        if bg_image_h == 0 or bg_image_w == 0:
            return [None, None, None]
        bg_gray = cv2.cvtColor(bg_image, cv2.COLOR_RGB2GRAY)
        plate_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        plate_mean, plate_stddev = cv2.meanStdDev(plate_gray)
        bg_mean, bg_stddev = cv2.meanStdDev(bg_gray)
        # cv2.imshow('bg_image', bg_image)
        # cv2.imshow('bg_image0', image)
        diff = bg_mean - plate_mean - plate_stddev
        image[:, :, 0:3] = cv2.add(image[:, :, 0:3], diff)
        # cv2.imshow('bg_image1', image)
        # cv2.waitKey()
        bg_objects = bg_instance['object']
        rand_index = int(random.uniform(0, len(bg_objects)))
        bg_counter = 0
        bg_mask = []
        bg_mask = numpy.zeros((bg_image_h, bg_image_w), numpy.uint8)
        for bg_obj in bg_objects:
            cv2.rectangle(bg_mask, (bg_obj['xmin'], bg_obj['ymin']), (bg_obj['xmax'], bg_obj['ymax']), 255, -1)

        bg_mask = cv2.dilate(bg_mask, numpy.ones((7, 7), numpy.uint8))
        cv2.inpaint(bg_image, bg_mask, 7, cv2.INPAINT_TELEA, bg_image)
        # cv2.imshow('bg_mask', bg_mask)
        # cv2.imshow('bg_image', bg_image)
        # cv2.waitKey()
        bg_obj = bg_objects[rand_index]

        bg_ymin = int(bg_obj['ymin'] - 0.5 * (bg_obj['ymax'] - bg_obj['ymin']))
        bg_xmin = int(bg_obj['xmin'] - 0.5 * (bg_obj['xmax'] - bg_obj['xmin']))
        if bg_xmin < 0:
            bg_xmin = 0
        if bg_ymin < 0:
            bg_ymin = 0
        bg_ymax = int(bg_obj['ymax'] + 0.5 * (bg_obj['ymax'] - bg_obj['ymin']))
        bg_xmax = int(bg_obj['xmax'] + 0.5 * (bg_obj['xmax'] - bg_obj['xmin']))

        if bg_xmax > bg_image_w - 1:
            bg_max = bg_image_w - 1
        if bg_ymax > bg_image_h - 1:
            bg_ymax = bg_image_h - 1
        bg_crop = bg_image[bg_ymin: bg_ymax, bg_xmin: bg_xmax, :].copy()

        # cv2.imshow('bg_crop', bg_crop)
        # cv2.waitKey()
        h = image.shape[0]
        w = image.shape[1]
        transformed_width = config['output_size'][0]
        transformed_height = config['output_size'][1]
        dx0 = last_x_offset - cam_x
        dy0 = last_y_offset - transformed_height +1
        dist_from_cam = numpy.sqrt(dx0*dx0 + dy0*dy0)
        max_dist = numpy.sqrt(transformed_width * transformed_width + transformed_height * transformed_height)
        w0 = plate_width
        h0 = plate_width * h / w
        tr_x = int(numpy.ceil(last_x_offset + 1 * w0))
        tr_y = int(numpy.ceil(last_y_offset + 1.5 * h0))
        bg_crop = cv2.resize(bg_crop, (int(w0 * 2), int(h0 * 2)), interpolation=cv2.INTER_CUBIC)
        bg_crop_h, bg_crop_w, _ = bg_crop.shape


        if tr_x > transformed_width - plate_width and tr_y > transformed_height - h0:
            return [orig_image, orig_annotation, orig_plate_objects, 0, 0, True]
        if tr_y > transformed_height - h0:
            return [orig_image, orig_annotation, orig_plate_objects, 0, 0, True]
        if tr_x > transformed_width - plate_width:
            last_x_offset = transformed_width
            # last_y_offset = int(numpy.ceil(tr_y + 1.5 * h0))
            return [orig_image, orig_annotation, orig_plate_objects, last_x_offset, last_y_offset, False]

        if orig_annotation is None:
            orig_annotation = numpy.zeros((transformed_width, transformed_height), numpy.uint8)
        orig_image[
        int(tr_y - bg_crop_h * 0.5):int(tr_y + bg_crop_h * 0.5),
        int(tr_x - bg_crop_w * 0.5):int(tr_x + bg_crop_w * 0.5),
        0:3] = bg_crop

        orig_image[
        int(tr_y - bg_crop_h * 0.5):int(tr_y + bg_crop_h * 0.5),
        int(tr_x - bg_crop_w * 0.5):int(tr_x + bg_crop_w * 0.5),
        3] = 255

        last_x_offset = int(numpy.ceil(tr_x + 1.5 * w0))

        dx_beta = tr_x - cam_x
        dy_beta = transformed_height - 1 - tr_y

        alpha = (1 - dist_from_cam / transformed_height) * numpy.pi / 2
        beta = (numpy.arctan2(dx_beta, dy_beta) / numpy.pi / 2 * config['max_dbeta'] / 180 * numpy.pi)
        gamma = -1 * dx_beta / transformed_width * config['max_dgamma'] / 180 * numpy.pi
        factor = 1000
        plate_width *= (1 - dx_beta / transformed_width) + 0.65
        dz = w / plate_width * factor
        print(f'\nalpha: {alpha}\nbeta: {beta}\ngamma: {gamma}\n')
        a_1 = numpy.array([[1, 0, -w / 2],
                           [0, 1, -h / 2],
                           [0, 0, 0],
                           [0, 0, 1]])
        r_x = numpy.array([[1, 0, 0, 0],
                           [0, numpy.cos(alpha), -numpy.sin(alpha), 0],
                           [0, numpy.sin(alpha), numpy.cos(alpha), 0],
                           [0, 0, 0, 1]])
        r_y = numpy.array([[numpy.cos(beta), 0, -numpy.sin(beta), 0],
                           [0, 1, 0, 0],
                           [numpy.sin(beta), 0, numpy.cos(beta), 0],
                           [0, 0, 0, 1]])
        r_z = numpy.array([[numpy.cos(gamma), -numpy.sin(gamma), 0, 0],
                           [numpy.sin(gamma), numpy.cos(gamma), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        r = numpy.matmul(numpy.matmul(r_x, r_y), r_z)
        t = numpy.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, dz],
                         [0, 0, 0, 1]])
        a_2 = numpy.array([[factor, 0, tr_x, 0],
                           [0, factor, tr_y, 0],
                           [0, 0, 1, 0]])

        trans = numpy.matmul(a_2, numpy.matmul(t, numpy.matmul(r, a_1)))
        mask_image_ = numpy.ones((image.shape[0], image.shape[1])) * 255

        image = cv2.warpPerspective(image, trans, (config['output_size'][0], config['output_size'][1]), flags=cv2.INTER_AREA)
        mask_image = cv2.warpPerspective(mask_image_, trans, (config['output_size'][0], config['output_size'][1]),
                                    flags=cv2.INTER_AREA)
        annotation = cv2.warpPerspective(annotation, trans, (config['output_size'][0], config['output_size'][1]), flags=cv2.INTER_AREA)

        for obj in plate_objects:
            src_points = []
            for pnt in obj.corners:
                src_points.append([pnt.x, pnt.y])
            dst_points = cv2.perspectiveTransform(numpy.float32(src_points).reshape(-1, 1, 2), trans)
            obj.corners.clear()
            for pnt in dst_points:
                new_pnt = pascal_voc.ObjectPoint.__new__(pascal_voc.ObjectPoint)
                new_pnt.x = pnt[0][0]
                new_pnt.y = pnt[0][1]
                obj.corners.append(new_pnt)
            obj.bounding_box.x = min([elm.x for elm in obj.corners])
            obj.bounding_box.y = min([elm.y for elm in obj.corners])
            obj.bounding_box.width = max([elm.x for elm in obj.corners]) - obj.bounding_box.x
            obj.bounding_box.height = max([elm.y for elm in obj.corners]) - obj.bounding_box.y
            if obj.lable is not 'Plate':
                obj_center = [obj.bounding_box.x + obj.bounding_box.width * 0.5,
                              obj.bounding_box.y + obj.bounding_box.height * 0.5]
                thresh = (annotation == numpy.uint8 ((numpy.uint8(obj.lable) + 1) * ann_step)).astype(numpy.uint8)
                contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt_counter = 0
                for cnt in contours[0]:
                    if cv2.contourArea(cnt) < 5:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    box_center = [x + w * .5, y + h * .5]
                    dist = numpy.sqrt((box_center[0] - obj_center[0]) * (box_center[0] - obj_center[0]) +\
                           (box_center[1] - obj_center[1]) * (box_center[1] - obj_center[1]))
                    if dist > obj.bounding_box.width *0.5:
                        continue
                    if cnt_counter == 0:
                        obj.bounding_box.x = x
                        obj.bounding_box.y = y
                        obj.bounding_box.width = w
                        obj.bounding_box.height = h
                    cnt_counter += 1
                    xmax = obj.bounding_box.x + obj.bounding_box.width
                    ymax = obj.bounding_box.y + obj.bounding_box.height
                    if  xmax < x + w:
                        xmax = x + w
                    if ymax < y + h:
                        ymax = y + h
                    obj.bounding_box.x = min(x, obj.bounding_box.x)
                    obj.bounding_box.y = min(y, obj.bounding_box.y)
                    obj.bounding_box.width = xmax - obj.bounding_box.x
                    obj.bounding_box.height = ymax - obj.bounding_box.y

        # image_gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        # thresh = (numpy.uint8( image_gray> 0) * 255)
        ## cv2.imshow('thresh', thresh );cv2.waitKey()
        # thresh = cv2.dilate(thresh, numpy.ones( (int(w0 * 0.2), int(w0 * 0.2)), dtype=numpy.uint8), iterations=1)
        # thresh = cv2.erode(thresh, numpy.ones((int(w0 * 0.2), int(w0 * 0.2)), dtype=numpy.uint8), iterations=1)

        thresh = ~numpy.uint8(mask_image)
        # thresh = cv2.dilate(thresh, numpy.ones((int(w0 * 0.03), int(w0 * 0.03)), dtype=numpy.uint8), iterations=1)
        # cv2.imshow('thresh1', thresh );cv2.waitKey()
        thresh = thresh // 255
        orig_image[:, :, 0] = (orig_image[:, :, 0] * thresh) + image[:, :, 0]
        orig_image[:, :, 1] = (orig_image[:, :, 1] * thresh) + image[:, :, 1]
        orig_image[:, :, 2] = (orig_image[:, :, 2] * thresh) + image[:, :, 2]
        # orig_image[:, :, 0:3] = (orig_image[:, :, 0:3] * orig_image[:, :, 3])+ image[:, :, 0:3]
        orig_image[:, :, 3] = orig_image[:, :, 3] + ~thresh
        # cv2.imshow('thresh', orig_image[:, :, 3])
        orig_annotation = orig_annotation + annotation
        if orig_plate_objects is None or orig_plate_objects is []:
            orig_plate_objects = plate_objects
        else:
            orig_plate_objects += plate_objects
    elif use_tiling_method == 2:

        bg_image = cv2.imread(bg_instance['filename'])
        bg_image_h, bg_image_w, _ = bg_image.shape
        if bg_image_h == 0 or bg_image_w == 0:
            return [orig_image, orig_annotation, orig_plate_objects, 0, 0, True]
        bg_gray = cv2.cvtColor(bg_image, cv2.COLOR_RGB2GRAY)
        plate_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        plate_mean, plate_stddev = cv2.meanStdDev(plate_gray)
        bg_mean, bg_stddev = cv2.meanStdDev(bg_gray)
        # cv2.imshow('bg_image', bg_image)
        # cv2.imshow('bg_image0', image)
        diff = bg_mean - plate_mean - plate_stddev
        image[:, :, 0:3] = cv2.add(image[:, :, 0:3], diff)
        # cv2.imshow('bg_image1', image)
        # cv2.waitKey()
        bg_objects = bg_instance['object']
        rand_index = int(random.uniform(0, len(bg_objects)))
        bg_counter = 0
        bg_mask = []
        bg_mask = numpy.zeros((bg_image_h, bg_image_w), numpy.uint8)
        for bg_obj in bg_objects:
            cv2.rectangle(bg_mask, (bg_obj['xmin'], bg_obj['ymin']), (bg_obj['xmax'], bg_obj['ymax']), 255, -1)
        bg_mask = cv2.dilate(bg_mask, numpy.ones((7, 7), numpy.uint8))
        cv2.inpaint(bg_image, bg_mask, 7, cv2.INPAINT_TELEA, bg_image)
        # cv2.imshow('bg_mask', bg_mask)
        # cv2.imshow('bg_image', bg_image)
        # cv2.waitKey()
        bg_obj = bg_objects[rand_index]
        plate_width = (bg_obj['xmax'] - bg_obj['xmin']) * 0.6
        h = image.shape[0]
        w = image.shape[1]
        transformed_width = bg_image_w
        transformed_height = bg_image_h
        # cam_x = transformed_width // 2  # int(random.uniform(0, transformed_width))
        tr_x = int(bg_obj['xmin'] + (bg_obj['xmax'] - bg_obj['xmin']) * 0.5)
        tr_y = int(bg_obj['ymin'] + (bg_obj['ymax'] - bg_obj['ymin']) * 0.5)
        dx0 = tr_x - cam_x
        dy0 = tr_y - transformed_height + 1
        dist_from_cam = numpy.sqrt(dx0 * dx0 + dy0 * dy0)
        max_dist = numpy.sqrt(transformed_width * transformed_width + transformed_height * transformed_height)
        w0 = plate_width
        h0 = plate_width * h / w


        orig_transformed_width = config['output_size'][0]
        orig_transformed_height = config['output_size'][1]
        dx_beta = tr_x - cam_x
        dy_beta = orig_transformed_height - 1 - tr_y
        alpha = (dist_from_cam / (orig_transformed_height * 1.5)) * numpy.pi / 2
        beta = (numpy.arctan2(dx_beta, dy_beta) / numpy.pi / 2 * config['max_dbeta'] / 180 * numpy.pi)
        gamma = dx_beta / orig_transformed_width * config['max_dgamma'] / 180 * numpy.pi
        factor = 1000
        plate_width *= (1 - dx_beta / transformed_width) + 0.65
        dz = w / plate_width * factor

        image_h, image_w, _ = orig_image.shape
        if orig_annotation is None:
            orig_annotation = numpy.zeros((image_w, image_h), numpy.uint8)
        print(f'\nalpha: {alpha}\nbeta: {beta}\ngamma: {gamma}\n')
        a_1 = numpy.array([[1, 0, -w / 2],
                           [0, 1, -h / 2],
                           [0, 0, 0],
                           [0, 0, 1]])
        r_x = numpy.array([[1, 0, 0, 0],
                           [0, numpy.cos(alpha), -numpy.sin(alpha), 0],
                           [0, numpy.sin(alpha), numpy.cos(alpha), 0],
                           [0, 0, 0, 1]])
        r_y = numpy.array([[numpy.cos(beta), 0, -numpy.sin(beta), 0],
                           [0, 1, 0, 0],
                           [numpy.sin(beta), 0, numpy.cos(beta), 0],
                           [0, 0, 0, 1]])
        r_z = numpy.array([[numpy.cos(gamma), -numpy.sin(gamma), 0, 0],
                           [numpy.sin(gamma), numpy.cos(gamma), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        r = numpy.matmul(numpy.matmul(r_x, r_y), r_z)
        t = numpy.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, dz],
                         [0, 0, 0, 1]])
        a_2 = numpy.array([[factor, 0, tr_x, 0],
                           [0, factor, tr_y, 0],
                           [0, 0, 1, 0]])

        trans = numpy.matmul(a_2, numpy.matmul(t, numpy.matmul(r, a_1)))

        image = cv2.warpPerspective(image, trans, (bg_image_w, bg_image_h), flags=cv2.INTER_CUBIC)
        annotation = cv2.warpPerspective(annotation, trans, (bg_image_w, bg_image_h), flags=cv2.INTER_CUBIC)
        new_w = int(random.randint(int(image_w * .25), int(image_w * scale_factor)))
        new_h = int(random.randint(int(image_h * .25), int(image_h * scale_factor)))
        last_x_offset0 = last_x_offset
        last_y_offset0 = last_y_offset
        last_x_offset += int(random.randint(int((image_w * scale_factor- new_w) * (scale_factor * 0.5)), int((image_w * scale_factor - new_w) * scale_factor)))
        last_y_offset += int(random.randint(int((image_h * scale_factor - new_h) * (scale_factor * 0.5)), int((image_h * scale_factor - new_h) * scale_factor)))
        if last_x_offset0 > image_w * (1 - scale_factor) and last_y_offset > image_h * (1 - scale_factor):
            last_x_offset = 0
            last_y_offset = 0
            return [orig_image, orig_annotation, orig_plate_objects, 0, 0, True]
        elif last_y_offset0 > image_h *  (1 - scale_factor) :
            last_x_offset = 0
            last_y_offset = 0
            return [orig_image, orig_annotation, orig_plate_objects, 0, 0, True]
        for obj in plate_objects:
            src_points = []
            for pnt in obj.corners:
                src_points.append([pnt.x, pnt.y])
            dst_points = cv2.perspectiveTransform(numpy.float32(src_points).reshape(-1, 1, 2), trans)
            obj.corners.clear()
            for pnt in dst_points:
                new_pnt = pascal_voc.ObjectPoint.__new__(pascal_voc.ObjectPoint)
                new_pnt.x = pnt[0][0]
                new_pnt.y = pnt[0][1]
                obj.corners.append(new_pnt)
            obj.bounding_box.x = min([elm.x for elm in obj.corners])
            obj.bounding_box.y = min([elm.y for elm in obj.corners])
            obj.bounding_box.width = max([elm.x for elm in obj.corners]) - obj.bounding_box.x
            obj.bounding_box.height = max([elm.y for elm in obj.corners]) - obj.bounding_box.y
            if obj.lable is 'Plate':
                plate_obj_xmin = int(obj.bounding_box.x)
                plate_obj_ymin = int(obj.bounding_box.y)
                plate_obj_xmax = int(obj.bounding_box.x + obj.bounding_box.width)
                plate_obj_ymax = int(obj.bounding_box.y + obj.bounding_box.height)
            if obj.lable is not 'Plate':
                obj_center = [obj.bounding_box.x + obj.bounding_box.width * 0.5,
                              obj.bounding_box.y + obj.bounding_box.height * 0.5]
                thresh = (annotation == numpy.uint8((numpy.uint8(obj.lable) + 1) * ann_step)).astype(numpy.uint8)
                contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt_counter = 0
                for cnt in contours[0]:
                    cnt_counter += 1
                    if cv2.contourArea(cnt) < 5:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    box_center = [x + w * .5, y + h * .5]
                    dist = numpy.sqrt((box_center[0] - obj_center[0]) * (box_center[0] - obj_center[0]) + \
                                      (box_center[1] - obj_center[1]) * (box_center[1] - obj_center[1]))
                    if dist > obj.bounding_box.width * 0.5:
                        continue
                    if cnt_counter == 1:
                        obj.bounding_box.x = x
                        obj.bounding_box.y = y
                        obj.bounding_box.width = w
                        obj.bounding_box.height = h
                    xmax = obj.bounding_box.x + obj.bounding_box.width
                    ymax = obj.bounding_box.y + obj.bounding_box.height
                    if xmax < x + w:
                        xmax = x + w
                    if ymax < y + h:
                        ymax = y + h
                    obj.bounding_box.x = min(x, obj.bounding_box.x)
                    obj.bounding_box.y = min(y, obj.bounding_box.y)
                    obj.bounding_box.width = xmax - obj.bounding_box.x
                    obj.bounding_box.height = ymax - obj.bounding_box.y
        for obj in plate_objects:
            obj.bounding_box.x *=  new_w / bg_image_w
            obj.bounding_box.y *= new_h / bg_image_h
            obj.bounding_box.width *= new_w / bg_image_w
            obj.bounding_box.height *= new_h / bg_image_h
            for pnt in obj.corners:
                pnt.x *=  new_w / bg_image_w
                pnt.y *=  new_h / bg_image_h
        for obj in plate_objects:
            obj.bounding_box.x += last_x_offset
            obj.bounding_box.y += last_y_offset
            for pnt in obj.corners:
                pnt.x +=  last_x_offset
                pnt.y +=  last_y_offset

        bg_ = numpy.zeros((bg_image_h, bg_image_w, 4), numpy.uint8)


        bg_[:, :, 0] = cv2.bitwise_and(bg_image[:, :, 0], ~image[:, :, 3])
        bg_[:, :, 1] = cv2.bitwise_and(bg_image[:, :, 1], ~image[:, :, 3])
        bg_[:, :, 2] = cv2.bitwise_and(bg_image[:, :, 2], ~image[:, :, 3])
        bg_[:, :, 3] = ~image[:, :, 3]
        bg_is_random = 0
        if bg_instance['filename'].find('random') >= 0:
            bg_is_random = 1
            bg_[:, :, 0:3] = bg_image
            image[:,:, 0] = 0
            image[:, :, 1] = 0
            image[:, :, 2] = 0
            image[:, :, 3] = 0
        crop_image = (image + bg_).copy()
        crop_image = cv2.resize(crop_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        annotation = cv2.resize(annotation, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        bg_mask= bg_[:, :, 0]

        bg_mask[plate_obj_ymin: plate_obj_ymax, plate_obj_xmin: plate_obj_xmax] = 255
        bg_mask = cv2.resize(bg_mask, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow('crop_image', crop_image)
        thresh = ~(numpy.uint8(bg_mask > 0) * 255)

        # cv2.imshow('thresh', thresh);cv2.waitKey()
        thresh = cv2.erode(thresh, numpy.ones((int(w0 * 0.1), int(w0 * 0.1)), dtype=numpy.uint8), iterations=1)
        thresh = cv2.dilate(thresh, numpy.ones((int(w0 * 0.1), int(w0 * 0.1)), dtype=numpy.uint8), iterations=1)

        # cv2.imshow('thresh', ~thresh);cv2.waitKey()
        thresh = thresh // 255

        orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 0] = \
            (orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 0] * thresh) +\
            crop_image[:, :, 0]
        orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 1] = \
            (orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 1] * thresh) +\
            crop_image[:, :, 1]
        orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 2] = \
            (orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 2] * thresh) +\
            crop_image[:, :, 2]
        # orig_image[:, :, 0:3] = (orig_image[:, :, 0:3] * orig_image[:, :, 3])+ image[:, :, 0:3]
        orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 3] = \
            orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 3] + thresh

        orig_annotation[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w] = \
            orig_annotation[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w] + annotation

        last_x_offset = last_x_offset + new_w

        if last_x_offset > image_w * (1 - scale_factor)  and last_y_offset > image_h * (1 - scale_factor) :
            last_x_offset = 0
            last_y_offset = 0
            reset_new_image = True
        elif last_x_offset > image_w * (1 - scale_factor) :
            last_y_offset = last_y_offset + new_h
            last_x_offset = 0
        elif last_y_offset > image_h * (1 - scale_factor) :
            last_x_offset = 0
            last_y_offset = 0
            reset_new_image = True
        if(bg_is_random == 0):
            if orig_plate_objects is None or orig_plate_objects is []:
                orig_plate_objects = plate_objects
            else:
                orig_plate_objects += plate_objects
    elif use_tiling_method == 3:
        bg_image = cv2.imread(bg_instance['filename'])
        bg_image_h, bg_image_w, _ = bg_image.shape
        if bg_image_h == 0 or bg_image_w == 0:
            return [None, None, None]
        bg_gray = cv2.cvtColor(bg_image, cv2.COLOR_RGB2GRAY)
        plate_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        plate_mean, plate_stddev = cv2.meanStdDev(plate_gray)
        bg_mean, bg_stddev = cv2.meanStdDev(bg_gray)
        # cv2.imshow('bg_image', bg_image)
        # cv2.imshow('bg_image0', image)
        diff = bg_mean - plate_mean - plate_stddev
        image[:, :, 0:3] = cv2.add(image[:, :, 0:3], diff)
        # cv2.imshow('bg_image1', image)
        # cv2.waitKey()
        bg_objects = bg_instance['object']
        rand_index = int(random.uniform(0, len(bg_objects)))
        bg_counter = 0
        bg_mask = []
        bg_mask = numpy.zeros((bg_image_h, bg_image_w), numpy.uint8)
        for bg_obj in bg_objects:
            cv2.rectangle(bg_mask, (bg_obj['xmin'], bg_obj['ymin']), (bg_obj['xmax'], bg_obj['ymax']), 255, -1)

        bg_mask = cv2.dilate(bg_mask, numpy.ones((7, 7), numpy.uint8))
        cv2.inpaint(bg_image, bg_mask, 7, cv2.INPAINT_TELEA, bg_image)

        bg_obj = bg_objects[rand_index]

        bg_ymin = int(bg_obj['ymin'] - 0.5 * (bg_obj['ymax'] - bg_obj['ymin']))
        bg_xmin = int(bg_obj['xmin'] - 0.5 * (bg_obj['xmax'] - bg_obj['xmin']))
        if bg_xmin < 0:
            bg_xmin = 0
        if bg_ymin < 0:
            bg_ymin = 0
        bg_ymax = int(bg_obj['ymax'] + 0.5 * (bg_obj['ymax'] - bg_obj['ymin']))
        bg_xmax = int(bg_obj['xmax'] + 0.5 * (bg_obj['xmax'] - bg_obj['xmin']))

        if bg_xmax > bg_image_w - 1:
            bg_max = bg_image_w - 1
        if bg_ymax > bg_image_h - 1:
            bg_ymax = bg_image_h - 1
        bg_plate_center = (int((bg_xmax + bg_xmin) * 0.5), int((bg_ymax + bg_ymin) * 0.5))
        line_count = 360
        line_thickness = 20;
        # cv2.imshow('bg_image', bg_image)
        # cv2.waitKey()
        for counter in range(0, line_count):
            line_len = random.randint(int((bg_xmax - bg_xmin) * 0.25), int((bg_xmax - bg_xmin) * 0.65))
            new_pt_x = int(bg_plate_center[0] + line_len * numpy.cos(counter * (numpy.pi * 2) / line_count))
            new_pt_y = int(bg_plate_center[1] - line_len * numpy.sin(counter * (numpy.pi * 2) / line_count))
            new_pt = (new_pt_x, new_pt_y)
            cv2.line(bg_mask, bg_plate_center, new_pt, (1), 40)
        cv2.dilate(bg_mask, numpy.ones((3,3), dtype=numpy.uint8) , iterations=line_thickness // 3)
        locations = cv2.findNonZero(bg_mask);
        x_min = min(locations[:,:,0])[0]
        y_min = min(locations[:, :, 1])[0]
        x_max = max(locations[:, :, 0])[0]
        y_max = max(locations[:, :, 1])[0]
        bg_image = bg_image[y_min: y_max, x_min: x_max, :]
        bg_mask = bg_mask[y_min: y_max, x_min: x_max]
        bg_image_h, bg_image_w, _ = bg_image.shape
        bg_image[:, :, 0] = bg_image[:, :, 0] * bg_mask
        bg_image[:, :, 1] = bg_image[:, :, 1] * bg_mask
        bg_image[:, :, 2] = bg_image[:, :, 2] * bg_mask

        # cv2.namedWindow('bg_mask', cv2.WINDOW_KEEPRATIO)
        # cv2.namedWindow('bg_mask0', cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('bg_mask', bg_image)
        # cv2.imshow('bg_mask0', bg_mask)
        # cv2.waitKey()


        bg_obj = bg_objects[rand_index]
        bg_obj['xmin'] = bg_obj['xmin'] - x_min
        bg_obj['xmax'] = bg_obj['xmax'] - x_min
        bg_obj['ymin'] = bg_obj['ymin'] - y_min
        bg_obj['ymax'] = bg_obj['ymax'] - y_min
        plate_width = (bg_obj['xmax'] - bg_obj['xmin']) * 0.6
        h = image.shape[0]
        w = image.shape[1]
        transformed_width = bg_image_w
        transformed_height = bg_image_h
        # cam_x = transformed_width // 2  # int(random.uniform(0, transformed_width))
        tr_x = int(bg_obj['xmin'] + (bg_obj['xmax'] - bg_obj['xmin']) * 0.5)
        tr_y = int(bg_obj['ymin'] + (bg_obj['ymax'] - bg_obj['ymin']) * 0.5)
        dx0 = tr_x - cam_x
        dy0 = tr_y - transformed_height + 1
        dist_from_cam = numpy.sqrt(dx0 * dx0 + dy0 * dy0)
        max_dist = numpy.sqrt(transformed_width * transformed_width + transformed_height * transformed_height)
        w0 = plate_width
        h0 = plate_width * h / w

        orig_transformed_width = config['output_size'][0]
        orig_transformed_height = config['output_size'][1]
        dx_beta = tr_x - cam_x
        dy_beta = orig_transformed_height - 1 - tr_y
        alpha = (dist_from_cam / (orig_transformed_height * 1.5)) * numpy.pi / 2
        beta = (numpy.arctan2(dx_beta, dy_beta) / numpy.pi / 2 * config['max_dbeta'] / 180 * numpy.pi)
        gamma = dx_beta / orig_transformed_width * config['max_dgamma'] / 180 * numpy.pi
        factor = 1000
        plate_width *= (1 - dx_beta / transformed_width) + 0.65
        dz = w / plate_width * factor

        image_h, image_w, _ = orig_image.shape
        if orig_annotation is None:
            orig_annotation = numpy.zeros((image_w, image_h), numpy.uint8)
        print(f'\nalpha: {alpha}\nbeta: {beta}\ngamma: {gamma}\n')
        a_1 = numpy.array([[1, 0, -w / 2],
                           [0, 1, -h / 2],
                           [0, 0, 0],
                           [0, 0, 1]])
        r_x = numpy.array([[1, 0, 0, 0],
                           [0, numpy.cos(alpha), -numpy.sin(alpha), 0],
                           [0, numpy.sin(alpha), numpy.cos(alpha), 0],
                           [0, 0, 0, 1]])
        r_y = numpy.array([[numpy.cos(beta), 0, -numpy.sin(beta), 0],
                           [0, 1, 0, 0],
                           [numpy.sin(beta), 0, numpy.cos(beta), 0],
                           [0, 0, 0, 1]])
        r_z = numpy.array([[numpy.cos(gamma), -numpy.sin(gamma), 0, 0],
                           [numpy.sin(gamma), numpy.cos(gamma), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        r = numpy.matmul(numpy.matmul(r_x, r_y), r_z)
        t = numpy.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, dz],
                         [0, 0, 0, 1]])
        a_2 = numpy.array([[factor, 0, tr_x, 0],
                           [0, factor, tr_y, 0],
                           [0, 0, 1, 0]])

        trans = numpy.matmul(a_2, numpy.matmul(t, numpy.matmul(r, a_1)))

        image = cv2.warpPerspective(image, trans, (bg_image_w, bg_image_h), flags=cv2.INTER_CUBIC)
        annotation = cv2.warpPerspective(annotation, trans, (bg_image_w, bg_image_h), flags=cv2.INTER_CUBIC)
        new_w = int(random.randint(int(image_w * .25), int(image_w * scale_factor)))
        new_h = int(random.randint(int(image_h * .25), int(image_h * scale_factor)))
        last_x_offset += int(random.randint(int((image_w * scale_factor - new_w) * (scale_factor * 0.5)),
                                            int((image_w * scale_factor - new_w) * scale_factor)))
        last_y_offset += int(random.randint(int((image_h * scale_factor - new_h) * (scale_factor * 0.5)),
                                            int((image_h * scale_factor - new_h) * scale_factor)))
        if last_x_offset > image_w * (1 - scale_factor) and last_y_offset > image_h * (1 - scale_factor):
            last_x_offset = 0
            last_y_offset = 0
            return [orig_image, orig_annotation, orig_plate_objects, 0, 0, True]
        elif last_y_offset > image_h * (1 - scale_factor):
            last_x_offset = 0
            last_y_offset = 0
            return [orig_image, orig_annotation, orig_plate_objects, 0, 0, True]
        for obj in plate_objects:
            src_points = []
            for pnt in obj.corners:
                src_points.append([pnt.x, pnt.y])
            dst_points = cv2.perspectiveTransform(numpy.float32(src_points).reshape(-1, 1, 2), trans)
            obj.corners.clear()
            for pnt in dst_points:
                new_pnt = pascal_voc.ObjectPoint.__new__(pascal_voc.ObjectPoint)
                new_pnt.x = pnt[0][0]
                new_pnt.y = pnt[0][1]
                obj.corners.append(new_pnt)
            obj.bounding_box.x = min([elm.x for elm in obj.corners])
            obj.bounding_box.y = min([elm.y for elm in obj.corners])
            obj.bounding_box.width = max([elm.x for elm in obj.corners]) - obj.bounding_box.x
            obj.bounding_box.height = max([elm.y for elm in obj.corners]) - obj.bounding_box.y
            if obj.lable is 'Plate':
                plate_obj_xmin = int(obj.bounding_box.x)
                plate_obj_ymin = int(obj.bounding_box.y)
                plate_obj_xmax = int(obj.bounding_box.x + obj.bounding_box.width)
                plate_obj_ymax = int(obj.bounding_box.y + obj.bounding_box.height)
            if obj.lable is not 'Plate':
                obj_center = [obj.bounding_box.x + obj.bounding_box.width * 0.5,
                              obj.bounding_box.y + obj.bounding_box.height * 0.5]
                thresh = (annotation == numpy.uint8((numpy.uint8(obj.lable) + 1) * ann_step)).astype(numpy.uint8)
                contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt_counter = 0
                for cnt in contours[0]:
                    cnt_counter += 1
                    if cv2.contourArea(cnt) < 5:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    box_center = [x + w * .5, y + h * .5]
                    dist = numpy.sqrt((box_center[0] - obj_center[0]) * (box_center[0] - obj_center[0]) + \
                                      (box_center[1] - obj_center[1]) * (box_center[1] - obj_center[1]))
                    if dist > obj.bounding_box.width * 0.5:
                        continue
                    if cnt_counter == 1:
                        obj.bounding_box.x = x
                        obj.bounding_box.y = y
                        obj.bounding_box.width = w
                        obj.bounding_box.height = h
                    xmax = obj.bounding_box.x + obj.bounding_box.width
                    ymax = obj.bounding_box.y + obj.bounding_box.height
                    if xmax < x + w:
                        xmax = x + w
                    if ymax < y + h:
                        ymax = y + h
                    obj.bounding_box.x = min(x, obj.bounding_box.x)
                    obj.bounding_box.y = min(y, obj.bounding_box.y)
                    obj.bounding_box.width = xmax - obj.bounding_box.x
                    obj.bounding_box.height = ymax - obj.bounding_box.y

        for obj in plate_objects:
            obj.bounding_box.x *= new_w / bg_image_w
            obj.bounding_box.y *= new_h / bg_image_h
            obj.bounding_box.width *= new_w / bg_image_w
            obj.bounding_box.height *= new_h / bg_image_h
            for pnt in obj.corners:
                pnt.x *= new_w / bg_image_w
                pnt.y *= new_h / bg_image_h
        for obj in plate_objects:
            obj.bounding_box.x += last_x_offset
            obj.bounding_box.y += last_y_offset
            for pnt in obj.corners:
                pnt.x += last_x_offset
                pnt.y += last_y_offset
        bg_ = numpy.zeros((bg_image_h, bg_image_w, 4), numpy.uint8)

        bg_[:, :, 0] = cv2.bitwise_and(bg_image[:, :, 0], ~image[:, :, 3])
        bg_[:, :, 1] = cv2.bitwise_and(bg_image[:, :, 1], ~image[:, :, 3])
        bg_[:, :, 2] = cv2.bitwise_and(bg_image[:, :, 2], ~image[:, :, 3])
        bg_[:, :, 3] = ~image[:, :, 3]
        bg_is_random = 0
        if bg_instance['filename'].find('random') >= 0:
            bg_is_random = 1
            bg_[:, :, 0:3] = bg_image
            image[:, :, 0] = 0
            image[:, :, 1] = 0
            image[:, :, 2] = 0
            image[:, :, 3] = 0
        crop_image = (image + bg_).copy()
        crop_image = cv2.resize(crop_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        annotation = cv2.resize(annotation, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        bg_mask = bg_[:, :, 0]

        bg_mask[plate_obj_ymin: plate_obj_ymax, plate_obj_xmin: plate_obj_xmax] = 255
        bg_mask = cv2.resize(bg_mask, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow('crop_image', crop_image)
        thresh = ~(numpy.uint8(bg_mask > 0) * 255)

        # cv2.imshow('thresh', thresh);cv2.waitKey()
        thresh = cv2.erode(thresh, numpy.ones((int(w0 * 0.1), int(w0 * 0.1)), dtype=numpy.uint8), iterations=1)
        thresh = cv2.dilate(thresh, numpy.ones((int(w0 * 0.1), int(w0 * 0.1)), dtype=numpy.uint8), iterations=1)

        # cv2.imshow('thresh', ~thresh);cv2.waitKey()
        thresh = thresh // 255
        crop_image = cv2.blur(crop_image, (3,3))
        orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 0] = \
            (orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 0] * thresh) + \
            crop_image[:, :, 0]
        orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 1] = \
            (orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 1] * thresh) + \
            crop_image[:, :, 1]
        orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 2] = \
            (orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 2] * thresh) + \
            crop_image[:, :, 2]
        # orig_image[:, :, 0:3] = (orig_image[:, :, 0:3] * orig_image[:, :, 3])+ image[:, :, 0:3]
        orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 3] = \
            orig_image[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w, 3] + thresh

        orig_annotation[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w] = \
            orig_annotation[last_y_offset: last_y_offset + new_h, last_x_offset: last_x_offset + new_w] + annotation

        last_x_offset = last_x_offset + new_w

        if last_x_offset > image_w * (1 - scale_factor) and last_y_offset > image_h * (1 - scale_factor):
            last_x_offset = 0
            last_y_offset = 0
            reset_new_image = True
        elif last_x_offset > image_w * (1 - scale_factor):
            last_y_offset = last_y_offset + new_h
            last_x_offset = 0
        elif last_y_offset > image_h * (1 - scale_factor):
            last_x_offset = 0
            last_y_offset = 0
            reset_new_image = True
        if (bg_is_random == 0):
            if orig_plate_objects is None or orig_plate_objects is []:
                orig_plate_objects = plate_objects
            else:
                orig_plate_objects += plate_objects
    img = orig_image.copy()
    # cv2.circle(img, (tr_x, tr_y), 5, (25, 255, 25), 5)
    # cv2.imshow('plate', img)
    # cv2.waitKey()
    return [orig_image, orig_annotation, orig_plate_objects, last_x_offset, last_y_offset, reset_new_image]