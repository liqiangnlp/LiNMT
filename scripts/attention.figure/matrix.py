import numpy
import matplotlib  
matplotlib.use('Agg') 

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import sys
import json
import argparse

# python ***.py -s [srcfile] -t [1bestfile] -i [number]
# input:
#  alignment matrix - numpy array
#  shape (target tokens + eos, number of hidden source states = source tokens +eos)
# one line correpsonds to one decoding step producing one target token
# each line has the attention model weights corresponding to that decoding step
# each float on a line is the attention model weight for a corresponding source state.
# plot: a heat map of the alignment matrix
# x axis are the source tokens (alignment is to source hidden state that roughly corresponds to a source token)
# y axis are the target tokens

# http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
def plot_head_map_save_fig(mma, target_labels, source_labels, save_pic="tmp.png"):
  fig, ax = plt.subplots()
  heatmap = ax.pcolor(mma, cmap=plt.cm.Blues)

  # put the major ticks at the middle of each cell
  ax.set_xticks(numpy.arange(mma.shape[1])+0.5, minor=False)
  ax.set_yticks(numpy.arange(mma.shape[0])+0.5, minor=False)
  
  # without this I get some extra columns rows
  # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
  ax.set_xlim(0, int(mma.shape[1]))
  ax.set_ylim(0, int(mma.shape[0]))

  # want a more natural, table-like display
  ax.invert_yaxis()
  ax.xaxis.tick_top()

  # source words -> column labels
  ax.set_xticklabels(source_labels, minor=False)
  # target words -> row labels
  ax.set_yticklabels(target_labels, minor=False)
  
  plt.xticks(rotation=45)
  #plt.tight_layout()
  plt.savefig(save_pic)
  #plt.show()

def read_plot_alignment_matrices_niutrans(src, best, pos = 1):
  mat , source_labels, target_labels = read_matrices_niutrans(src, best, pos) 
  source = source_labels.decode('utf-8').split()
  target = target_labels.decode('utf-8').split()
  plot_head_map_save_fig(mat, target, source, "%s.png" % pos)

def read_matrices_niutrans(source_file = "mt08.cn", best_file = "mt08.1best", position = 1):
    files = open(source_file)
    filet = open(best_file)
    source_labels = files.readlines()[position - 1].strip()
    best = filet.readlines()[position - 1]
    items = best.split(" |||| ")
    tgt_labels = items[0]
    raw_matrix = items[-1]
    matrix = []
    for it in raw_matrix.split(" "):
        matrix.append([float(i) for i in it.split("/")])
    return numpy.array(matrix), source_labels, tgt_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="source file")
    parser.add_argument("-t", "--target", help="1best file")
    parser.add_argument("-i", "--ith", type=int, help="the position of sentence to generate")
    args = parser.parse_args()
    read_plot_alignment_matrices_niutrans(args.source, args.target, args.ith)
