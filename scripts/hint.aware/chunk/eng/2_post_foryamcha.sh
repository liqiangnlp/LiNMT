input=e.input.utf8.4.clear.tok.out.pos.caseless
output=$input.postpro

python LiNMT-pos-postprocess.py -i $input -o $output


input=$output
output=$input.foryamcha

python LiNMT-pos-for-yamcha.py -i $input -o $output


