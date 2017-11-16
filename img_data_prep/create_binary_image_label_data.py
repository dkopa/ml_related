from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import shuffle

import os
import struct
import numpy as np
from scipy import misc
from skimage import color, io
from skimage.transform import resize
from matplotlib import pyplot as plt

io_dir = '/raid5/hasnat/dbs/cbsr.ia.ac.cn/' 
fileList = io_dir + 'file_lists/casia_CL_3_tr.txt'
opFileName = io_dir + 'casia_CL_3_tr.bin'

# image size specification
num_channels = 1
height = 112
width = 96

def convert_to(imageLabelList):
  num_images = len(imageLabelList)

  filename = opFileName
  print('Writing', filename)

  if os.path.isfile(opFileName):
      os.remove(opFileName)
  fo = open(opFileName, 'w')

  rndNums = np.random.permutation(num_images)
  #rndNums = range(num_images)
  
  # for imageLabel, index in zip(imageLabelList, range(0, num_images)):
  # index=0
  for index in range(num_images):
    tIndx = rndNums[index]
    imageLabel = imageLabelList[tIndx]
    print(tIndx, imageLabel)
    imfile, labelStr = imageLabel.split()
    
    tImg = misc.imread(io_dir+imfile)
    #io.imshow(tImg)
    
    if(len(tImg.shape)>2):
        tImg = color.rgb2gray(tImg)
	tImg = np.uint8(tImg * 255)

    label_ = np.int16(labelStr)    
    fo.write(struct.pack('i', label_))
    
    # write image as uint8
    for i in range(height):
      for j in range(width):
        # for k in range(num_channels):
        fo.write(struct.pack('B', tImg[i, j]))

  fo.close()

def main(argv):
  # Get the image list and labels
  imageLabelList = open(fileList, "r").readlines()

  # Convert to Examples and write the result to TFRecords.
  convert_to(imageLabelList)


if __name__ == '__main__':
  # tf.app.run()
  main(None)
