import sys
import json
import cPickle as pkl

#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        try:
            with open(filename, 'rb') as f:
                return pkl.load(f)
        except:
               with open(filename,'r') as f:
                   dict = {}
                   for l in f:
                       item = l.strip().split()
                       dict[item[0]] = item[1]
               return dict


def loadDict(dictionaries, VOCAB_SIZE):
    worddicts = load_dict(dictionaries)
    #print "In all %s words in training sentences" %  len(worddicts)

    dict1 = {}
    for k , idx in worddicts.items():
        if idx < VOCAB_SIZE:
            dict1.update({k: idx})
    return dict1

def statistic(filename, dict1):
    num = 0
    countUNK = 0
    all_word_countUNK = 0
    all_word_count = 0
    file = open(filename)
    for line in file:
        num += 1
        flag = True
        words_in = line.strip().split(' ')
        all_word_count += len(words_in)
        for w in words_in:
            if w not in dict1:
                flag = False
                all_word_countUNK += 1
        if not flag:
            countUNK += 1
    file.close()
    return num, countUNK, round(countUNK *1.0 / num,5), all_word_count, all_word_countUNK, round(all_word_countUNK *1.0 / all_word_count, 5)
           
#print "%s \tsentences,\t %s\t sentences contain UNK\tpercent: %s" % (num, countUNK, countUNK *1.0 / num)
#print "%s \ttokens,\t%s \ttokens is UNK\tpercent: %s" %  (all_word_count, all_word_countUNK, all_word_countUNK *1.0 / all_word_count)
