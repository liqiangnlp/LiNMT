from statisticCoverPercent import *
import sys




dic_filename = sys.argv[1]
train_filename = sys.argv[2]



size = [(i+1)*10000 for i in range(0,20)]


for it in size:
    dic = loadDict(dic_filename, it)
    res = statistic(train_filename, dic)
    # num, countUNK, countUNK / num, all_word_count, all_word_countUNK, all_word_countUNK  / all_word_count
    print "Dic size=[%s]\t[%s]\tsents \t[%s]\tcontainUNK\tpercent=\t[%s]" % (it, res[0], res[1], 1-res[2])
    print "Dic size=[%s]\t[%s]\ttokens\t[%s]\tUNK       \tpercent=\t[%s]" % (it, res[3], res[4], 1-res[5])
