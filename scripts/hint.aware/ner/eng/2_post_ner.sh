input=e.input.utf8.4.clear.tok.out.ner.caseless 
oword=$input.wordrmNER
olabel=$input.label


python LiNMT-postprocess-ner-rmNER-eng.py -i $input -w $oword -l $olabel -v 1 



