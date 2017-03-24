#!/usr/bin/python
#-*- encoding:utf8 -*-


import SocketServer
import os
import re
import sys
from ctypes import *


preprocess_path="../config/NiuTrans.NiuSeg.ch-pt.pt.config"
decoder_path="../config/NiuTrans.NMT-nmt-decoding-sentence.config"
postprocess_path="../config/NiuTrans.NMT-nmt-postprocess.config"


class MyTCPHandler(SocketServer.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print "[Client     ] {}".format(self.client_address[0])
        segment_sentence = preprocess_do_job(self.data)
        print '[Raw Input  ] {}'.format(self.data)
        print '[Segment    ] {}'.format(segment_sentence)
        translation_result = decoder_do_job(segment_sentence)
        print '[Translation] {}'.format(translation_result)
        unk_replace = unk_do_job(segment_sentence, translation_result)
        print '[UNK replace] {}'.format(unk_replace)
        sys.stdout.flush()
        # just send back the same data, but upper-cased
        # self.request.sendall(self.data.upper())
        self.request.sendall(unk_replace)

if __name__ == "__main__":
    HOST, PORT = "10.119.186.29", 8088
    print 'Python server start!'
    # Create the server, binding to localhost on port 9999
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)

    PreProcessLib = cdll.LoadLibrary('../resource/libMTPreProcessEn2Ch.so')
    PreProcessLib.python_init(preprocess_path)
    preprocess_do_job = PreProcessLib.python_do_job
    preprocess_do_job.restype = c_char_p

    LiNMTlib = cdll.LoadLibrary('../lib/nmt.so')
    LiNMTlib.python_decoder_init(decoder_path)
    LiNMTlib.python_unk_init(postprocess_path)
    decoder_do_job = LiNMTlib.python_decoder_do_job
    decoder_do_job.restype = c_char_p
    unk_do_job = LiNMTlib.python_unk_do_job
    unk_do_job.restype = c_char_p

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()



