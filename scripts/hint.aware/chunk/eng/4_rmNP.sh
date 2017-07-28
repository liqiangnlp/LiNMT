input=e.input.utf8.4.clear.tok.out.pos.caseless.postpro.foryamcha.chunk 
word=$input.word
label=$input.label


python LiNMT-postprocess-chunk-rmNP-eng.py -i $input -w $word -l $label -v 1
#python LiNMT-postprocess-chunk-rmNP-eng.py -i $input -w $word -l $label


