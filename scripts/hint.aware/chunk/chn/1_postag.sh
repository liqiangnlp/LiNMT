config=niuparser.config
input=c.input.utf8.4.clear.tok.out  
output=$input.pos
thread=1
nohup ./NiuParser-v1.3.0-mt-linux --POS -c $config -in $input -out $output -t $thread &



