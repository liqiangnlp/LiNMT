#!/usr/bin/python
#-*- encoding:utf8 -*-


import SocketServer
import os
import re
import sys
from ctypes import *


preprocess_path="../config/NiuTrans.NiuSeg.ch-pt.ch.config"
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
        print '\n\n\n'
        print '[Client     ] {}'.format(self.client_address[0])
        print '[Raw Input  ] {}'.format(self.data)
        
        send_lines = ''
        receive_lines = self.data.splitlines()
        if len(receive_lines) > 5:
            send_lines = '[Warning 1001: Up to 5 lines]' 
        else:
            i = 0
            for line in receive_lines:
                line = line.strip()
                if line == '':
                    output = line;
                else:
                    i += 1
                    print '[Line {}     ] {}'.format(i,line)
                    segment_sentence = preprocess_do_job(line)
                    seg_fields = segment_sentence.split(' |||| ')
                    print '[Segment    ] {}'.format(seg_fields[0])
                    words = seg_fields[0].split()
                    print '[Words num  ] {}'.format(len(words))
                    if len(words) < 40:
                        translation_result = decoder_do_job(segment_sentence)
                        translation_fields = translation_result.split(' |||| ')
                        print '[Translation] {}'.format(translation_fields[0])
                        unk_replace = unk_do_job(segment_sentence, translation_result)
                        print '[UNK replace] {}'.format(unk_replace)
                        output = unk_replace
                    else:
                        output = '[Warning 1002: Exceeding the maximum length of 40 words]'
                send_lines += output + '\n'

        sys.stdout.flush()
        self.request.sendall(send_lines.capitalize())

if __name__ == "__main__":
    HOST, PORT = "10.119.186.29", 8090
    print 'Python server start!'
    # Create the server, binding to localhost on port 9999
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)

    PreProcessLib = cdll.LoadLibrary('../resource/libMTPreProcessCh2En.so')
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



