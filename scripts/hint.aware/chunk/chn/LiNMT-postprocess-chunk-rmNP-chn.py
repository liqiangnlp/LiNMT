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

  parser.add_argument(
    '--valid', '-v', type=bool, default=False, metavar='VALID', help='check if validation data. %(default)s')

  parser.add_argument(
    '--double', '-d', type=bool, default=False, metavar='DOUBLE', help='just double corpora. %(default)s')

  return parser



def double(ifobj, owfobj):
  i = 0
  for line in ifobj:
    i += 1
    line = line.strip()
    owfobj.write(line+'\n')
    owfobj.write(line+'\n')

    if i % 10000 == 0:
      sys.stdout.write('\r{0}'.format(str(i)))

  sys.stdout.write('\r{0}\n'.format(str(i)))


def pos_postprocess(ifobj, owfobj, olfobj, ologfobj):


  print str(args.valid)+'\n'

  line_word = ''
  line_word_rmNER = ''
  line_label = ''
  
  i = 0
  no_ner_lines = 0
  have_ner_lines = 0
  total_lines = 0
  for line in ifobj:

    i += 1
    total_lines += 1
    line = line.strip()
    words = line.split()
    match = 0
    lastNerTag = ''
    have_ner = 0
    line_empty = ''
    for word in words:
      line_empty += ' #'
      matchObj = re.match(r'<(.*)>(.*)</\1>', word)
      if matchObj and (matchObj.group(1) == 'NP'):
        line_word += ' '+matchObj.group(2)
        line_word_rmNER += ' ****'
        line_label += ' B-'+matchObj.group(1)
        match = 0
        lastNerTag = ''
        have_ner = 1
        continue
      elif matchObj:
        line_word += ' '+matchObj.group(2)
        line_word_rmNER += ' '+matchObj.group(2)
        continue

      matchObj = re.match(r'<(.*)>(.*)', word)
      if matchObj and (matchObj.group(1) == 'NP'):
        line_word += ' '+matchObj.group(2)
        line_word_rmNER += ' ****'
        line_label += ' B-'+matchObj.group(1)
        match = 1
        lastNerTag = matchObj.group(1)
        have_ner = 1
        continue
      elif matchObj:
        line_word += ' '+matchObj.group(2)
        line_word_rmNER += ' '+matchObj.group(2)
        continue
    

      matchObj = re.match(r'(.*)</(.*)>', word)
      if matchObj and (matchObj.group(2) == 'NP'):
        line_word += ' '+matchObj.group(1)
        line_word_rmNER += ' ****'
        line_label += ' I-'+matchObj.group(2)
        match = 0
        lastNerTag = ''
        have_ner = 1
        continue
      elif matchObj:
        line_word += ' '+matchObj.group(1)
        line_word_rmNER += ' '+matchObj.group(1)
        continue

      line_word += ' ' + format(word)
      line_word_rmNER += ' ' + format(word)
      if match == 1:
        line_label += ' I-'+lastNerTag
      else:
        line_label += ' O'



    line_word = "** "+line_word.strip()
    line_word_rmNER = "** "+line_word_rmNER.strip()
    line_empty = "*** "+line_empty.strip()
    line_label = line_label.strip()




    if have_ner == 1:
      have_ner_lines += 1
      owfobj.write(line_word_rmNER+'\n')
    else:
      no_ner_lines += 1
      if args.valid == True:
        owfobj.write(line_empty+'\n')
      else:
        owfobj.write(line_word+'\n')

    if args.valid == False:
      owfobj.write(line_empty+'\n')
    olfobj.write(line_label+'\n')
    line_word = ''
    line_word_rmNER = ''
    line_label = ''
    if i % 10000 == 0:
      sys.stdout.write('\r{0}'.format(str(i)))

  sys.stdout.write('\r{0}\n'.format(str(i)))
  ologfobj.write('have_ner_lines='+str(have_ner_lines)+'\n')
  ologfobj.write('no_ner_lines='+str(no_ner_lines)+'\n')
  ologfobj.write('total_lines='+str(total_lines)+'\n')

if __name__ == '__main__':

  parser = create_parser()
  args = parser.parse_args()

  # read/write files as UTF-8
  if args.input.name != '<stdin>':
    args.input = codecs.open(args.input.name, encoding='utf-8', errors='ignore')

  args.outword = codecs.open(args.outword.name, 'w', encoding='utf-8')
  args.outlabel = codecs.open(args.outlabel.name, 'w', encoding='utf-8')
  args.outlog = codecs.open(args.outword.name+'.log', 'w', encoding='utf-8')

  if args.double == True:
    double(args.input, args.outword)
  else:
    pos_postprocess(args.input, args.outword, args.outlabel, args.outlog)

