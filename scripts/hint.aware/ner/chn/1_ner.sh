config=niuparser.config 
input=c.input.utf8.clear.tok.lengthratio.max20.pos  
output=$input.ner
thread=1
nohup ./NiuParser-v1.3.0-mt-linux --NER -c $config -in $input -out $output -t $thread &



