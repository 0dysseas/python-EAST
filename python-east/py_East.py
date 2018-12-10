import argparse, sys
import cv2 as cv


#Check if the model is empty
def CV_assert(model):
  assert (len(model) > 0), 'Model is empty.'


#Parses the command line arguments
def parse_cli_arguments():
  parser = argparse.ArgumentParser(description='Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of "EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)");')

  parser.add_argument('-i', help='| Path to input image or video file. Skip this argument to capture frames from a camera.',
                      type=str)
  parser.add_argument('-model', help='| Path to a binary .pb file contains trained network.',
                      type=str)
  parser.add_argument('-width', help=' | 320 | Preprocess input image by resizing to a specific width. It should be multiple by 32.',
                      type=int)
  parser.add_argument('-height', help='| 320 | Preprocess input image by resizing to a specific height. It should be multiple by 32.',
                      type=int)
  parser.add_argument('-thr', help='| 0.5 | Confidence threshold.',
                      type=float)
  parser.add_argument('-nms', help='| 0.4 | Non-maximum suppression threshold.',
                      type=float)
  
  #Show help message and exit if no arguments are passed
  if len(sys.argv) == 1:
    parser.print_help()
    parser.exit()

  args = parser.parse_args()

  return args.i, args.thr, args.nms, args.width, args.height, args.model




if __name__ == '__main__':
  #print(dir(cv.VideoCapture))
  win_name = 'EAST: An efficient and Accurate Scene Text Detector'
  outnames = ['feature_fusion/Conv_7/Sigmoid', 'feature_fustion/concat_3']

  #Parse command line arguments
  inp, conf_threshold, nms_threshold, width, height, model = parse_cli_arguments()
  
  CV_assert(model)

  #Load EAST network
  net = cv.dnn.readNet(model)
  
  #Create the VideoCapture object and capture the image or the video feed
  cap = cv.VideoCapture()
  if inp is not None:
    cap.open(inp)
  else:
    cap.open(0)

  #Create the window
  cv.namedWindow(win_name, WINDOW_NORMAL)

  while cv.waitKey(1) < 0:
    #Read frame
    has_frame, frame = cap.read()
    if not has_frame:
      cv.waitKey()
      break
    
    #Create 4 dimensional blob from frame
    blob= cv.dnn.blobFromImage(frame, 1, (width, height), (123.68, 116.78, 103.94), swapRB=True, crop=False)






