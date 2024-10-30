import torch
import torchvision
import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
from PIL import Image
from time import time
import onnxruntime as ort

import logging
logging.basicConfig(level=logging.ERROR)
# Chạy bằng onnx


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, img):
        """Return updated labels and image with added border."""
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # only scale down, do not scale up (for better val mAP)
        if not self.scaleup:
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(
                dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / \
                shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        return img


def scale_boxes(img1_shape, boxes, img0_shape):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0],
               img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / \
        2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    # clip_boxes(boxes, img0_shape)
    return boxes


def draw_boxes(src_image, boxes, scores, labels):
    color = (0, 255, 0)
    original_im = src_image.copy()
    cropped_images = []
    for box, score, lb in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box.astype('int32')
        cropped_image = original_im[ymin:ymax, xmin:xmax].copy()
        if cropped_image is not None and len(cropped_image) > 0 and len(cropped_image[0]) > 0:
            cropped_images.append(cropped_image)
            
        cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 2)
    

    return src_image, cropped_images


def to_xml(xml_path, imgname, boxes, labels, size):
    w, h = size
    root = ET.Element('annotations')
    filename = ET.SubElement(root, 'filename')
    filename.text = imgname
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    width.text, height.text = str(w), str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    for box, label in zip(boxes, labels):
        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = label
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin, ymin = ET.SubElement(
            bndbox, 'xmin'), ET.SubElement(bndbox, 'ymin')
        xmax, ymax = ET.SubElement(
            bndbox, 'xmax'), ET.SubElement(bndbox, 'ymax')
        xmin.text, ymin.text, xmax.text, ymax.text = [str(b) for b in box]
    ET.ElementTree(root).write(xml_path)


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(
        2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.
    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        agnostic=False,
        multi_label=False,
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=100,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.
    Arguments:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int): (optional) The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels
    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    # YOLOv8 model in validation model, output = (inference_out, loss_out)
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # select only inference output

    device = 'cpu'
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time()
    output = [torch.zeros((0, 6 + nm), device=device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.transpose(0, -1)[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(box)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None],
                          j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[
                conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        # sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if (time() - t) > time_limit:
            break  # time limit exceeded

    return output

## --------- ##


class Detector:
    def __init__(self, model_type='onnx'):

        self.model_type = model_type
        # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        # model_path = os.path.join(os.getcwd(), "best.onnx")
        model_path = "best_lesion_detection.onnx"
        self.session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.resizer = LetterBox((640, 640), auto=False)

        self.labels = ['polyps']
        self.shape = 640

    def preprocess(self, image):
        image = self.resizer(image)
        # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        image = np.expand_dims(image[..., ::-1], 0).transpose((0, 3, 1, 2))
        image = np.ascontiguousarray(image, dtype='float32')  # contiguous

        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image

    def __call__(self, img, threshold=0.5):
        h, w = img.shape[:2]
        tensor = self.preprocess(img)
        s = time()
        # ONNX inference
        # tensor = np.random.random((2, 3, 640, 640)).astype('float32')
        pred = self.session.run(None, {self.input_name: tensor})[0]
        pred = torch.from_numpy(pred)
        # print("pred: ", pred.shape)
        detections = non_max_suppression(
            pred, conf_thres=threshold, iou_thres=0.5)[0]
        detections = detections.numpy()
        # print("detections: ", detections.shape)
        e = time()

        if len(detections) == 0:
            return e - s, [], [], []

        boxes, scores, class_ids = detections[:,
                                              :4], detections[:, 4], detections[:, 5]
        print("class_ids: ", class_ids)

        boxes = scale_boxes((self.shape, self.shape), boxes, (h, w))

        class_names = [self.labels[int(d)] for d in class_ids]

        print([(list(b), s, c) for b, s, c in zip(boxes, scores, class_names)])

        # Arange box from top to bottom #
        indices = np.argsort(boxes[:, 3])
        corrected_boxes = [boxes[i] for i in indices]
        corrected_classes = [class_names[i] for i in indices]
        corrected_scores = [scores[i] for i in indices]
        # ***#
        return e - s, corrected_boxes, corrected_scores, corrected_classes


def detect_single(img_path: str, confidence=0.5, save_path='output', filename='filename'):
    """
    Detect an image format both png and jpg
    :param img_path: str, path to image
    :param confidence: 0 <= confidence < 1, object's confidence
    :param view: int, 0-not show, 1-show image with drew box/dot, -1-show cut ROI
    :param save: int, 0-not save, 1-save image with drew box/dot, -1-save cut ROI
    :param save_path: str, folder to save image/roi
    :return: None
    """
    # ################ #
    # INFERENCE STEP   #
    # ################ #
    predictor = Detector('onnx')
    
    s = time()
    imp = img_path
    # imp = os.path.join(img_path, 'img_0.jpg')
    image = cv2.imread(imp)
    # Check if image is not read correctly, read by PIL
    if image is None:
        im = Image.open(imp)
        image = np.array(im)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print("Cannot read image %s"%img_path + " --> continue")
        # return
    h, w = image.shape[:2]
    src_im = image.copy()
    runtime, boxes, scores, labels = predictor(image, confidence)
    e1 = time()
    print("inference time: ", runtime)
    name = img_path.split('/')[-1]
    if not boxes:
        print("Cannot find any thing in %s !" % name)
        # if save:
        #     cv2.imwrite(os.path.join(save_path, name), pad_image)
        return
    print("Found in %s! Detect time: %.3f" % (name, (e1 - s)))

    draw, cropped_images = draw_boxes(src_im.copy(), boxes, scores, labels)
    cv2.imwrite(os.path.join(save_path, f"marked-{filename}-lesion.png"), draw)

    output_path = os.path.join(save_path,f'output_{filename}_lesion')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    list_path = []
    for idx, cropped_image in enumerate(cropped_images):
        path_file = os.path.join(output_path, f"output_{filename}_lesion_{idx}.png")
        cv2.imwrite(path_file, cropped_image)
        list_path.append(path_file)
    return list_path


def is_image(name):
    return 'jpg' in name or 'png' in name or 'jpeg' in name


def infer_lesion_detection(list_endoscopy_list_path: list):
    list_path = []
    for endoscopy_file_path in list_endoscopy_list_path:
        print(endoscopy_file_path)
        file_name = endoscopy_file_path.split('/')[-1]
        save_path = endoscopy_file_path.replace(file_name, "")

        raw_name = file_name.split('.png')[0]

        list_lesion_path = detect_single(img_path=endoscopy_file_path, save_path=save_path, filename=raw_name)

        if list_lesion_path and len(list_lesion_path)>0:
            list_path = list_path + list_lesion_path
    return list_path

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--path', type=str, default="./datasets/polyps/valid/images",
                        required=False, help='Path to the test data folder')
    parser.add_argument('-t', '--threshold', type=float,
                        default=0.5, help='Object detection threshold')
    parser.add_argument('-vf', '--save_image', type=int,
                        default=1, help='1 for view image and 0 for not')
    parser.add_argument('-sf', '--save_xml', type=int,
                        default=0, help='1 for save image and 0 for not')
    parser.add_argument('-sp', '--save_path', type=str, default='result',
                        help='specific save result folder when save_image=True')
    args = parser.parse_args()

