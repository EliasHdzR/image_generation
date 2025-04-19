import cv2
from PyQt6.QtGui import QImage

def cvimage_to_qimage(frame):
    """
    Convierte un frame de OpenCV a un QImage
    :param frame: Frame de OpenCV
    :return: QImage
    """
    height, width, channel = frame.shape
    bytes_per_line = 3 * width
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

def np_arr_to_cvimage(arr):
    """
    Convierte un array de numpy a un frame de OpenCV
    :param arr: Array de numpy
    :return: Frame de OpenCV
    """
    arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr_rgb