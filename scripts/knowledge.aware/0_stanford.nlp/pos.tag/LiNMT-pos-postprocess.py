#!/usr/bin/python
# -*- coding: utf-8 -*-


# Author: Qiang Li
# Email: liqiangneu@gmail.compile
# Time: 13:59, 03/27/2017


import sys
import codecs
import argparse
import random
from io import open
argparse.open = open

reload(sys)
sys.setdefaultencoding('utf8')


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
    description='Stanford Pos PostProcess')

  parser.add_argument(
    '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
    metavar='PATH', help='Input text (default: standard input).')

  parser.add_argument(
    '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
    metavar='PATH', help='Output file for random template (default: standard output)')

  return parser


def pos_postprocess(ifobj, ofobj):
  for line in ifobj:
    line = line.strip()
    ofobj.write('{0}\n'.format(line))
    ifobj.readline()


if __name__ == '__main__':

  parser = create_parser()
  args = parser.parse_args()

  # read/write files as UTF-8
  if args.input.name != '<stdin>':
    args.input = codecs.open(args.input.name, encoding='utf-8')
  if args.output.name != '<stdout>':
    args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

  pos_postprocess(args.input, args.output)

