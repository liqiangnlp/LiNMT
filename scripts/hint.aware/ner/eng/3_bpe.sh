codes=/home/liqiang/10m-zh-en-final/0_Data/Train/all-06022017/e.input.utf8.clear.tok.lengthratio.max20.codes52k
input=e.input.utf8.4.clear.tok.out.ner.caseless.wordrmNER
output=$input.bpe52k


~/LiNMT/bin/LiNMT --bpe-segment $codes $input $output



