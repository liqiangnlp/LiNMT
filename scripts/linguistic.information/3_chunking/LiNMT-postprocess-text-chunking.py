#!/usr/bin/python
# -*- coding: utf-8 -*-


# Author: Qiang Li
# Email: liqiangneu@gmail.compile
# Time: 10:27, 03/30/2017


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
    description='Text Chunking')

  parser.add_argument(
    '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
    metavar='PATH', help='Input text (default: standard input).')

  parser.add_argument(
    '--outword', '-w', type=argparse.FileType('w'), required=True,
    metavar='PATH', help='Output word file')

  parser.add_argument(
    '--outlabel', '-l', type=argparse.FileType('w'), required=True,
    metavar='PATH', help='Output label file')


  return parser


def pos_postprocess(ifobj, owfobj, olfobj):

  line_word = ''
  line_label = ''
  for line in ifobj:

    line = line.strip()
    if line == '':
      line_word = line_word.strip()
      line_label = line_label.strip()
      owfobj.write('{0}\n'.format(line_word))
      olfobj.write('{0}\n'.format(line_label))
      line_word = ''
      line_label = ''
    else:
      words = line.split('\t')
      if words[0] == '':
        words[0] = 'NA'
      if words[3] == '':
        words[3] = 'O'
      line_word += ' '+words[0]
      line_label += ' '+words[3]


if __name__ == '__main__':

  parser = create_parser()
  args = parser.parse_args()

  # read/write files as UTF-8
  if args.input.name != '<stdin>':
    args.input = codecs.open(args.input.name, encoding='utf-8')

  args.outword = codecs.open(args.outword.name, 'w', encoding='utf-8')
  args.outlabel = codecs.open(args.outlabel.name, 'w', encoding='utf-8')

  pos_postprocess(args.input, args.outword, args.outlabel)

