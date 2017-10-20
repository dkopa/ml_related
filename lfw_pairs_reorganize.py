#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:44:56 2017

@author: hasnat
# LFW pair file read and oreorganize the pairs as (left_file right_file label)
"""
file_name = '/home/hasnat/Desktop/pairs.txt'

lines = open(file_name).readlines()

f_handle = open('/home/hasnat/Desktop/pairs_ordered.txt','a')

lenStrNum = 4
# tlines = [x for x in lines if len(x.split()) >= 3] 
for lc in range(len(lines)):
    tsplits = lines[lc].split()
    
    if (len(tsplits) == 3):
        # Same pairs
        tLeftFile = tsplits[0] + '/' + tsplits[0] + '_' + tsplits[1].zfill(lenStrNum-len(tsplits[1])+1) + '.jpg'
        tRightFile = tsplits[0]+ '/' + tsplits[0] + '_' + tsplits[2].zfill(lenStrNum-len(tsplits[2])+1) + '.jpg'
        f_handle.writelines(tLeftFile + '\t' + tRightFile + '\t' + '1\n')
    elif (len(tsplits) == 4):
        # different pairs
        tLeftFile = tsplits[0] + '/' + tsplits[0] + '_' + tsplits[1].zfill(lenStrNum-len(tsplits[1])+1) + '.jpg'
        tRightFile = tsplits[2]+ '/' + tsplits[2] + '_' + tsplits[3].zfill(lenStrNum-len(tsplits[3])+1) + '.jpg'
        f_handle.writelines(tLeftFile + '\t' + tRightFile + '\t' + '0\n')
f_handle.close()