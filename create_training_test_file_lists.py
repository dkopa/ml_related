"""
Created on Fri Oct 20 22:05:08 2017

@author: hasnat
Purpose: Create training and test image list from a directory of images
"""
import os
import numpy as np
import fn_utility_data_prepare as udp

dbPath = '/home/hasnat/Desktop/torem/'
listFilePath = '/home/hasnat/Desktop/'

# file related operations
file_list_tr = listFilePath + 'torem_full_list_tr.txt'
file_list_tst = listFilePath + 'torem_full_list_tst.txt'

if os.path.exists(file_list_tr):
    os.remove(file_list_tr)

if os.path.exists(file_list_tst):
    os.remove(file_list_tst)

fid_list_tr = open(file_list_tr,'w')
fid_list_tst = open(file_list_tst,'w')


uidName = os.listdir(dbPath)
numIds = len(uidName)

idCount = 0

for i in range(numIds):
        print '============================'
        print str(i) + '<>' + uidName[i]
        print '============================' 

        tFiles = os.listdir(dbPath + uidName[i] + '/')
        
        # Get indices for training and testing
        stTrIdx, edTrIdx, stTstIdx, edTstIdx = udp.getDataSplits(len(tFiles), 0.95)
        
        # training image list
        for imgNum in range(stTrIdx,edTrIdx):
            # corresponding image file name
            imFileName = dbPath + uidName[i] + '/' + tFiles[imgNum]
            if os.path.exists(imFileName):
                fid_list_tr.write(imFileName + ' ' + str(idCount) + '\n')
            else:
                print('file not found ...')
        
        # test image list
        for imgNum in range(stTstIdx, edTstIdx):
            # corresponding image file name
            imFileName = dbPath + uidName[i] + '/' + tFiles[imgNum]
            if os.path.exists(imFileName):
                fid_list_tst.write(imFileName + ' ' + str(idCount) + '\n')
            else:
                print('file not found ...')   
        
        idCount=idCount+1
        print(idCount)
        
# close files
fid_list_tr.close()
fid_list_tst.close()