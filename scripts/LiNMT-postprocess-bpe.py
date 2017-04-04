#!/usr/bin/python
# -*- coding: utf-8 -*-


# Author: Qiang Li
# Email: liqiangneu@gmail.compile
# Time: 15:06, 04/03/2017


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
    description='BPE PostProcess')

  parser.add_argument(
    '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
    metavar='PATH', help='Input text (default: standard input).')

  parser.add_argument(
    '--output', '-o', type=argparse.FileType('w'), required=True,
    metavar='PATH', help='Output word file')

  return parser


def bpe_postprocess(ifobj, ofobj):

  i = 0
  for line in ifobj:

    i += 1
    line = line.strip()
    line = line.replace('@@ ', '')
    ofobj.write(line+'\n')
    if i % 10000 == 0:
      sys.stdout.write('\r{0}'.format(str(i)))

  sys.stdout.write('\r{0}\n'.format(str(i)))

if __name__ == '__main__':

  parser = create_parser()
  args = parser.parse_args()

  # read/write files as UTF-8
  if args.input.name != '<stdin>':
    args.input = codecs.open(args.input.name, encoding='utf-8', errors='ignore')

  args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

  bpe_postprocess(args.input, args.output)

