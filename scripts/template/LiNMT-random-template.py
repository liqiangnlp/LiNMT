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
    description='Random Template')

  parser.add_argument(
    '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
    metavar='PATH', help='Input text (default: standard input).')

  parser.add_argument(
    '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
    metavar='PATH', help='Output file for random template (default: standard output)')

  parser.add_argument(
    '--log', '-l', type=argparse.FileType('w'), default=sys.stderr,
    metavar='PATH', help='Output log file (default: standard error)')

  parser.add_argument(
    '--percentage', '-p', type=float, default=0.2, metavar='PERT', help='replace percentage words into #Xi symbols. (default: %(default)s)')

  return parser


def random_template(fobj, pert, ofobj, oflog):
  for line in fobj:
    line = line.strip()
    v_words = line.split()
    # print 'word size {}'.format(len(v_words))
    oflog.write('{0}\n'.format(len(v_words)))
    output_string = ''
    i = 0;
    for word in v_words:
      # print 'random: {}'.format(random.uniform(0, len(v_words)))
      f_i = random.uniform(0, len(v_words))
      f_j = f_i / len(v_words)
      #print '{0} {1} {2}'.format(f_i, len(v_words), f_j)
      if f_j < pert:
        i += 1
        # print 'f_j = {0}, pert = {1}'.format(f_j, pert)
        output_string += ' #X'+str(i)
      else:
        output_string += ' '+word
    output_string = output_string.strip()
    ofobj.write('{0}\n'.format(output_string))


if __name__ == '__main__':

  parser = create_parser()
  args = parser.parse_args()

  # read/write files as UTF-8
  if args.input.name != '<stdin>':
    args.input = codecs.open(args.input.name, encoding='utf-8')
  if args.output.name != '<stdout>':
    args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
  if args.log.name != '<stderr>':
    args.log = codecs.open(args.log.name, 'w', encoding='utf-8')

  random_template(args.input, args.percentage, args.output, args.log)

