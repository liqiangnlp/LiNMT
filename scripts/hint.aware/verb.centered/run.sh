input=e.input.utf8.4.clear.gen.good.pos.caseless 
python LiNMT-pos-postprocess.py -i $input -o $input.wl
perl LiNMT-only-verb.pl $input.wl $input.wl.verb


