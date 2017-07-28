input=c.input.utf8.clear.tok.lengthratio.max20.pos.ner.prep
oword=$input.wordrmNER
olabel=$input.label


#python LiNMT-postprocess-ner-rmNER-chn.py -i $input -w $oword -l $olabel 


input=e.input.utf8.clear.tok.lengthratio.max20.bpe52k
output=$input.double
outlabel=tmp

python LiNMT-postprocess-ner-rmNER.py -d 1 -i $input -w $output -l $outlabel
rm $outlabel




