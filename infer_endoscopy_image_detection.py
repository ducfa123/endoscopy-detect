import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
from PIL import Image
from time import time
import onnxruntime as ort
import torch
import torchvision
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


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (torch.Tensor | numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (None): The function modifies the input `coordinates` in place, by clipping each coordinate to the image boundaries.
    """
    if isinstance(coords, torch.Tensor):  # faster individually
        coords[..., 0].clamp_(0, shape[1])  # x
        coords[..., 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): The shape of the image that the coords are from.
        coords (torch.Tensor): the coords to be scaled of shape n,2.
        img0_shape (tuple): the shape of the image that the segmentation is being applied to.
        ratio_pad (tuple): the ratio of the image size to the padded image size.
        normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        coords (torch.Tensor): The scaled coordinates.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


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


def is_image(name):
    return 'jpg' in name or 'png' in name or 'jpeg' in name


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
        x1, y1 = ET.SubElement(bndbox, 'x1'), ET.SubElement(bndbox, 'y1')
        x2, y2 = ET.SubElement(bndbox, 'x2'), ET.SubElement(bndbox, 'y2')
        x3, y3 = ET.SubElement(bndbox, 'x3'), ET.SubElement(bndbox, 'y3')
        x4, y4 = ET.SubElement(bndbox, 'x4'), ET.SubElement(bndbox, 'y4')

        x1.text, y1.text, x2.text, y2.text, x3.text, y3.text, x4.text, y4.text = [
            str(b) for b in box]
    ET.ElementTree(root).write(xml_path)


def draw_keypoints(src_im, boxes, scores, labels, keypoints):
    # print("keypoints: ", keypoints)
    original_im = src_im.copy()
    cropped_images = []
    for keypoint in keypoints:
        # Tạo contours từ keypoint
        contours = np.array(keypoint, dtype=np.int32).reshape(-1, 2)
        
        x, y, w, h = cv2.boundingRect(contours)
        cropped_image = original_im[y:y+h, x:x+w].copy()
        if cropped_image is not None and len(cropped_image) > 0 and len(cropped_image[0]) > 0:
            cropped_images.append(cropped_image)
            
        if contours.shape[0] > 0:
            cv2.drawContours(src_im, [contours], -1, (0, 255, 0), 1)

        # Kiểm tra nếu contours không rỗng
        # if contours.shape[0] > 0:
            # cv2.drawContours(src_im, [contours], -1, (0, 255, 0), 1)

    return src_im, cropped_images


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
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=1,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
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

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    # shape(1,84,6300) to shape(1,6300,84)
    prediction = prediction.transpose(-1, -2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None],
                          j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            # sort by confidence and remove excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
    return output

## --------- ##


class Detector:
    def __init__(self, model_type='onnx'):

        self.model_type = model_type
        model_path = "best_endoscopy_image_detection.onnx"
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
        pred = self.session.run(None, {self.input_name: tensor})[0]
        pred = torch.from_numpy(pred)

        detections = non_max_suppression(
            pred, conf_thres=threshold, iou_thres=0.5)[0]
        detections = detections.numpy()
        e = time()

        if len(detections) == 0:
            return e - s, [], [], []

        boxes, scores, class_ids, pred_kpts = detections[:,:4], detections[:, 4], detections[:, 5], detections[:, 6:]

        pred_kpts = scale_coords((self.shape, self.shape), pred_kpts, (h, w))

        boxes = scale_boxes((self.shape, self.shape), boxes, (h, w)).round()

        class_names = [self.labels[int(d)] for d in class_ids]

        print([(list(b), s, c, kp)
              for b, s, c, kp in zip(boxes, scores, class_names, pred_kpts)])

        # Arange box from top to bottom #
        indices = np.argsort(boxes[:, 3])
        corrected_boxes = [boxes[i] for i in indices]
        corrected_classes = [class_names[i] for i in indices]
        corrected_scores = [scores[i] for i in indices]
        corrected_kpts = [pred_kpts[i] for i in indices]
        # ***#
        return e - s, corrected_boxes, corrected_scores, corrected_classes, corrected_kpts


def detect_single(img_path: str, confidence=0.5, save_path='output'):
    print(2)
    """
    Detect an image format both png and jpg
    :param img_path: str, path to image
    :param confidence: 0 <= confidence < 1, object's confidence
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
    h, w = image.shape[:2]
    src_im = image.copy()
    runtime, boxes, scores, labels, keypoints = predictor(image, confidence)
    e1 = time()
    print("inference time: ", runtime)
    name = img_path.split('/')[-1]
    if not boxes:
        print("Cannot find any thing in %s !" % name)
        return
    print("Found in %s! Detect time: %.3f" % (name, (e1 - s)))

    draw, cropped_images = draw_keypoints(src_im.copy(), boxes, scores, labels, keypoints)
    cv2.imwrite(os.path.join(save_path, 'marked_output.png'), draw)

    output_path = os.path.join(save_path,'output_endoscopy')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    list_path = []
    for idx, cropped_image in enumerate(cropped_images):
        path_file = os.path.join(output_path, f"output_endoscopy_image_{idx}.png")
        cv2.imwrite(path_file, cropped_image)
        list_path.append(path_file)
    return list_path


def infer_endoscopy_image_detection(img_path: str):
    print(1)
    file_name = img_path.split('/')[-1]
    save_path = img_path.replace(file_name, "")
    
    list_path = detect_single(img_path=img_path, save_path=save_path)
    return list_path



# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', '--path', type=str, default="./datasets/lesion/valid/images",
#                         required=False, help='Path to the test data folder')
#     parser.add_argument('-t', '--threshold', type=float,
#                         default=0.5, help='Object detection threshold')
#     parser.add_argument('-vf', '--save_image', type=int,
#                         default=1, help='1 for view image and 0 for not')
#     parser.add_argument('-sf', '--save_xml', type=int,
#                         default=1, help='1 for save image and 0 for not')
#     parser.add_argument('-sp', '--save_path', type=str, default='result',
#                         help='specific save result folder when save_image=True')
#     args = parser.parse_args()
