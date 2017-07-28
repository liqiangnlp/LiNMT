input=c.input.utf8.clear.tok.lengthratio.max20.pos.ner
output=$input.prep

perl LiNMT-preprocess-ner-chn.pl < $input > $output



