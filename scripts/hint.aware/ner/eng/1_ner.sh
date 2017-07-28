curdir=`pwd`

input=$curdir/e.input.utf8.4.clear.tok.out 
echo $input
output=$input.ner.caseless
echo $output
log=$input.ner.log


tooldir=/home/liqiang/NiuParser/3_ner_eng/stanford-ner-2016-10-31
cd $tooldir
echo `pwd`
nohup ./ner.caseless.sh $input >$output 2>$log &




cd $curdir
echo `pwd`




