input=c.input.utf8.4.clear.tok.out.pos.chk
output=$input.prep
perl LiNMT-preprocess-chunk-chn.pl < $input > $output
