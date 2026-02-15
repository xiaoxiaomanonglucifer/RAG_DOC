#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：RAG_Automation-main 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：想去外太空的
@Date    ：2026/2/14 18:56 
'''
import torch

if torch.cuda.is_available():
    print("CUDA 可用")
else:
    print("CUDA 不可用")