codes=/home/liqiang/10m-zh-en-final/0_Data/Train/all-06022017/c.input.utf8.clear.tok.lengthratio.max20.codes46k
input=c.input.utf8.clear.tok.lengthratio.max20.pos.ner.prep.wordrmNER
output=$input.bpe46k


~/LiNMT/bin/LiNMT --bpe-segment $codes $input $output




