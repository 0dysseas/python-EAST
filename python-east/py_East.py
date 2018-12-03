import argparse, sys
import cv2 as cv

# Parses the command line arguments
def parse_cli_arguments():
  parser = argparse.ArgumentParser(description='Tensorflow implementation of EAST: An efficient and Accurate Scene Text Detector')

  parser.add_argument('i', help='| Path to input image or video file. Skip this argument to capture frames from a camera.')
  parser.add_argument('m', help='| Path to a binary .pb file contains trained network.')
  parser.add_argument('width', help=' | 320 | Preprocess input image by resizing to a specific width. It should be multiple by 32.')
  parser.add_argument('height', help='| 320 | Preprocess input image by resizing to a specific height. It should be multiple by 32.')
  parser.add_argument('thr', help='| 0.5 | Confidence threshold.')
  parser.add_argument('nms', help='| 0.4 | Non-maximum suppression threshold.')
  
  if len(sys.argv) == 1:
    parser.print_help()
    parser.exit()

  args = parser.parse_args()




if __name__ == '__main__':
  # Parse command line arguments
  parse_cli_arguments()














