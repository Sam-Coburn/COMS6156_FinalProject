#!/bin/sh

TIMESTAMP=`date +'%m%d%Y%H-%M'`

echo "file,T,tp,fp,tn,fn,prec,rec,acc,f1"
for i in ../*.csv; do
    f="`basename $i .csv`"
    echo "$f," `../../scripts/bestT_f1.rb "$i" "$f"`
done | tee best_t_$TIMESTAMP.csv