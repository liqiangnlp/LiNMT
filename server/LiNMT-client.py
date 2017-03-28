#!/usr/bin/python
#-*- encoding:utf8 -*-

import socket
import sys

HOST, PORT = "10.119.186.29", 8090
data = " ".join(sys.argv[1:])

# Create a socket (SOCK_STREAM means a TCP socket)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # Connect to server and send data
    sock.connect((HOST, PORT))
    sock.sendall(data + "\n")

    # Receive data from the server and shut down
    received = sock.recv(1024)
finally:
    sock.close()

print "[Input      ] {}".format(data)
print "[Translation] {}".format(received)


