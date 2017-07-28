input=c.input.utf8.4.clear.tok.out.pos.chk.prep 
outword=$input.wordrmNP
outlabel=$input.label


python LiNMT-postprocess-chunk-rmNP-chn.py -i $input -w $outword -l $outlabel --valid 1
#python LiNMT-postprocess-chunk-rmNP-chn.py -i $input -w $outword -l $outlabel




