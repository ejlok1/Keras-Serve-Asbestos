# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:29:10 2018

@author: User
"""
import os 
import sys 

print(os.getcwd())

sys.path.append(os.path.abspath("."))
print(sys.path)
from object_detection.Object_detector import * 
