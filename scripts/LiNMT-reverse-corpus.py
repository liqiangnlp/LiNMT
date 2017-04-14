#!/usr/bin/python
# -*- coding: utf-8 -*-


# Author: Qiang Li
# Email: liqiangneu@gmail.compile
# Time: 16:06, 14/03/2017


import re
import sys
import codecs
import argparse
import random
from io import open
argparse.open = open

reload(sys)
sys.setdefaultencoding('utf-8')


if sys.version_info < (3, 0):
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
  sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
else:
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
  sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)


def create_parser():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Reverse Corpus')

  parser.add_argument(
    '--input', '-i', type=argparse.FileType('r'), required=True,
    metavar='PATH', help='Input text (default: standard input).')

  parser.add_argument(
    '--output', '-o', type=argparse.FileType('w'), required=True,
    metavar='PATH', help='Output text')

  return parser


def reverse_corpus(ifobj, ofobj):

  i = 0
  for line in ifobj:

    i += 1
    line = line.strip()

    words = line.split()
    words.reverse()
    reverse_line = ' '.join(words)
    ofobj.write(reverse_line+'\n')
    if i % 10000 == 0:
      sys.stderr.write('\r{0}'.format(str(i)))

  sys.stderr.write('\r{0}\n'.format(str(i)))

if __name__ == '__main__':

  parser = create_parser()
  args = parser.parse_args()

  # read/write files as UTF-8
  args.input = codecs.open(args.input.name, encoding='utf-8', errors='ignore')
  args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

  reverse_corpus(args.input, args.output)

