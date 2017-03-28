#!/usr/bin/python
#-*- encoding:utf8 -*-


import SocketServer
import os
import re
import sys
from ctypes import *


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
        # print '[Raw Input  ] {}'.format(self.data)

        receive_lines = self.data.splitlines();
        send_lines = ''
        i = 0
        for line in receive_lines:
            i += 1
            print 'i = {}'.format(i)
            line = line.strip()
            
            # do_process
            if line == '':
                print '[EMPTY      ]'
                output = line;
            else:
                print '[line {}     ] {}'.format(i,line)
                output = line;
            # end
            
            send_lines += output + '\n' 

        sys.stdout.flush()
        # just send back the same data, but upper-cased
        # self.request.sendall(self.data.upper())
        print '[send_lines ] {}'.format(send_lines)
        self.request.sendall(send_lines)

if __name__ == "__main__":
    HOST, PORT = "10.119.186.29", 9999
    print 'Python server start!'
    # Create the server, binding to localhost on port 9999
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()



