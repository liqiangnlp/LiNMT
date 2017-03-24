#!/usr/bin/python
#-*- encoding:utf8 -*-

decoder_path="/home/liqiang/NiuTrans.NMT.pt2zh/config/NiuTrans.NMT-nmt-decoding-sentence.config" 
postprocess_path="/home/liqiang/NiuTrans.NMT.pt2zh/config/NiuTrans.NMT-nmt-postprocess.config"

from ctypes import *
import os 
libtest = cdll.LoadLibrary('../lib/LiNMT.so') 
libtest.python_decoder_init(decoder_path) 
libtest.python_unk_init(postprocess_path)

while True:
	decoder_do_job = libtest.python_decoder_do_job
	decoder_do_job.restype = c_char_p
	unk_do_job = libtest.python_unk_do_job
	unk_do_job.restype = c_char_p
	input_line = raw_input()
	line = decoder_do_job(input_line)
        print "[Translation] "+line
        unk_line = unk_do_job(input_line, line)
	print "[POSTPROCESS] "+unk_line

