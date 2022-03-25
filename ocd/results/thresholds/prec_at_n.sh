#!/bin/sh
TIMESTAMP=`date +'%m%d%Y%H-%M'`

echo "file,tp,fp,prec@n"
for i in ../*.csv; do
    f="`basename $i .csv`"
    echo "$f,"`../../scripts/prec_at_n.rb "$i" "$f" $1`
done | tee prec-at-n_$TIMESTAMP.csv
