config=niuparser.config
input=c.input.utf8.4.clear.tok.out.pos
output=$input.chk
thread=1
nohup ./NiuParser-v1.3.0-mt-linux --CHK -c $config -in $input -out $output -t $thread &




