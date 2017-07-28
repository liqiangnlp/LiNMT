model=conll2000.lc.model
input=e.input.utf8.4.clear.tok.out.pos.caseless.postpro.foryamcha
output=$input.chunk


yamcha -m $model < $input > $output


