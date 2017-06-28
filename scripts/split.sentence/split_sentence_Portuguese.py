#!/usr/bin/python
# coding:utf-8
import sys;
import re;
from itertools import dropwhile as DW;


# input
# input_sentences = 'As informações sobre o Programa, o regulamento e formulário de candidatura poderão ser consultados e descarregados na página electrónica do IC (www.icm.gov.mo) e no Website das Indústrias Culturais e Criativas de Macau (www.macaucci.gov.mo).';
# input_sentences = sys.argv[1];
intK = 30;
stopword = [',', '.', '?', '!', ';'];
max_len = 50;


# file



# def function

def countSpace(line):
    line_tmp = line.split(' ');
    return len(line_tmp);


def recover(input, log_path):
    return 0;


def Getline(input_sentences, intK, max_len):
    line_tmp = input_sentences.replace(', ', ' , ').replace('. ', ' . ').replace('? ', ' ? ').replace('! ', ' ! ').replace('; ',
                                                                                                                   ' ; ');
    # print(line_tmp)
    list_tmp = [];
    if len(set(stopword).intersection(line_tmp)) != 0:
        # tmp_list = re.split('(,|\.|\?|\!|;)|[\s]',line_tmp);
        tmp_list = line_tmp.split(' ');
        if len(tmp_list) >= intK:
            a = int(len(tmp_list) / intK);
            b = int(len(tmp_list) % intK);
            remnant = '';
            for i in range(a):
                # print('tpye1')

                tmp_element = list(sorted(set(stopword).intersection(tmp_list[i * intK:(i + 1) * intK]),key=tmp_list[i * intK:(i + 1) * intK].index));
                if len(tmp_element) != 0:
                    # print('tpye11')
                    tmp_element2 = list(sorted(set(stopword[1:]).intersection(tmp_list[i * intK:(i + 1) * intK]),key=tmp_list[i * intK:(i + 1) * intK].index));
                    if len(tmp_element2) != 0:
                        # print('tpye111')
                        index = next(DW(lambda x: tmp_list[i * intK:(i + 1) * intK][x] != tmp_element[-1],
                                        reversed(range(len(tmp_list[i * intK:(i + 1) * intK])))));
                    else:
                        # print('tpye112')
                        index = next(DW(lambda x: tmp_list[i * intK:(i + 1) * intK][x] != stopword[0],
                                        reversed(range(len(tmp_list[i * intK:(i + 1) * intK])))));
                    if i != a - 1:
                        # print('tpye1111')
                        slic1 = ' '.join(tmp_list[i * intK:(i * intK + index+1)]);
                        list_tmp.append(' '.join([remnant, slic1]));
                        remnant = ' '.join(tmp_list[(i * intK + index+1):(i+1) * intK]);
                        # print(str(list_tmp));
                    else:
                        # print('tpye1112')
                        tmp_element1 = list(sorted(set(stopword).intersection(tmp_list[a * intK:]),key=tmp_list[a * intK:].index));
                        if len(tmp_element1)!=0:
                            index1 = tmp_list[a * intK:].index(tmp_element1[0]);
                            addition = ' '.join(tmp_list[a * intK:(a * intK + index1 + 1)]);
                        else:
                            addition = ' '.join(tmp_list[a * intK:]);
                            index1 = b;
                        slic1 = ' '.join(tmp_list[i * intK:(i +1)* intK ]);
                        list_tmp.append(' '.join([remnant,slic1, addition]));

                        if b - index1 != 0:
                            list_tmp.append(' '.join(tmp_list[(a * intK + index1 + 1):]));
                        # print(str(list_tmp));
                else:
                    # print('tpye12')
                    if i!=a-1:
                        if countSpace(remnant)<=max_len:
                            remnant += ' '.join(tmp_list[i * intK:(i + 1) * intK]);
                        else:
                            tmp_llll=remnant.split(' ');
                            list_tmp.append(' '.join(tmp_llll[:max_len]));
                            remnant=' '.join(tmp_llll[max_len:])
                            remnant+=' '.join(tmp_list[i * intK:(i + 1) * intK])

                    else:
                        # print('AAA')
                        remnant += ' '.join(tmp_list[i * intK:]);
                        if countSpace(remnant)<=max_len:
                            list_tmp.append(remnant)
                        else:
                            tmp_llll = remnant.split(' ');
                            for j in range(int(countSpace(remnant)/max_len)):
                                list_tmp.append(' '.join(tmp_llll[j*max_len:(j+1)*max_len]));
                            list_tmp.append(' '.join(tmp_llll[int(countSpace(remnant)/max_len) * max_len:]));





        else:
            # print('tpye2')
            list_tmp.append(' '.join(tmp_list));

    else:
        tmp_list = line_tmp.strip().split(' ');
        a1 = int(len(tmp_list) / max_len);
        b1 = int(len(tmp_list) % max_len);

        if a1 >= 1:
            for i in range(a1):
                list_tmp.append(' '.join(tmp_list[i * max_len:(i + 1) * max_len]));
        if b1 != 0:
            list_tmp.append(' '.join(tmp_list[a1*b1:]));

    # print(str(list_tmp));
    return(list_tmp)

# Getline(input_sentences, intK, max_len)
