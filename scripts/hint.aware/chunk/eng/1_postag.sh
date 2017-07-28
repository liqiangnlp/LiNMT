curdir=`pwd`

input=$curdir/e.input.utf8.4.clear.tok.out 
echo $input
output=$input.pos.caseless
echo $output
log=$input.log


tooldir=/home/liqiang/NiuParser/5_dp_eng/stanford-parser-full-2016-10-31
cd $tooldir
echo `pwd`
nohup ./postag.sh $input >$output 2>$log &




cd $curdir
echo `pwd`




