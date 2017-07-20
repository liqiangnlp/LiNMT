#!/usr/bin/python
# -*- coding: utf-8 -*-


# Author: Qiang Li
# Email: liqiangneu@gmail.compile
# Time: 10:27, 03/30/2017


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
  
  i = 0
  for line in ifobj:

    i += 1
    line = line.strip()
    words = line.split()
    match = 0
    lastNerTag = ''
    for word in words:
      matchObj = re.match(r'<(.*)>(.*)</\1>', word)
      if matchObj:
        #owfobj.write('{0} '.format(matchObj.group(2)))
        if matchObj.group(1) == 'NP':
          line_word += ' #'
        else:
          line_word += ' '+matchObj.group(2)
        #olfobj.write('{0} '.format('B-'+matchObj.group(1)))
        line_label += ' B-'+matchObj.group(1)
        match = 0
        lastNerTag = ''
        continue

      matchObj = re.match(r'<(.*)>(.*)', word)
      if matchObj:
        #print matchObj.group(1)
        #owfobj.write('{0} '.format(matchObj.group(2)))
        if matchObj.group(1) == 'NP':
          line_word += ' #'
        else:
          line_word += ' '+matchObj.group(2)
        #olfobj.write('{0} '.format('B-'+matchObj.group(1)))
        line_label += ' B-'+matchObj.group(1)
        match = 1
        lastNerTag = matchObj.group(1)
        continue

      matchObj = re.match(r'(.*)</(.*)>', word)
      if matchObj:
        #owfobj.write('{0} '.format(matchObj.group(1)))
        if matchObj.group(2) == 'NP':
          line_word += ' #'
        else:
          line_word += ' '+matchObj.group(1)
        #olfobj.write('{0} '.format('I-'+matchObj.group(2)))
        line_label += ' I-'+matchObj.group(2)
        match = 0
        lastNerTag = ''
        continue

      if lastNerTag == 'NP':
        line_word += ' #'
      else:
        line_word += ' ' + format(word)
      if match == 1:
        #owfobj.write('{0} '.format(word))
        #olfobj.write('{0} '.format('I-'+lastNerTag))
        line_label += ' I-'+lastNerTag
      else:
        #owfobj.write('{0} '.format(word))
        #olfobj.write('{0} '.format('O'))
        line_label += ' O'

    line_word = line_word.strip()
    line_label = line_label.strip()
    owfobj.write(line_word+'\n')
    olfobj.write(line_label+'\n')
    line_word = ''
    line_label = ''
    if i % 10000 == 0:
      sys.stdout.write('\r{0}'.format(str(i)))

  sys.stdout.write('\r{0}\n'.format(str(i)))

if __name__ == '__main__':

  parser = create_parser()
  args = parser.parse_args()

  # read/write files as UTF-8
  if args.input.name != '<stdin>':
    args.input = codecs.open(args.input.name, encoding='utf-8', errors='ignore')

  args.outword = codecs.open(args.outword.name, 'w', encoding='utf-8')
  args.outlabel = codecs.open(args.outlabel.name, 'w', encoding='utf-8')

  pos_postprocess(args.input, args.outword, args.outlabel)

