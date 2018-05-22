import colorContour
import cv2
import numpy as np

if __name__ == "__main__":
    in_dir = './data/test.MOV'
    out_dir = './data/out.avi'
    colorContour.detect_hand(in_dir, out_dir)
    
