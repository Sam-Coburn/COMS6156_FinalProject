#!/bin/sh
TIMESTAMP=`date +'%m%d%Y%H-%M'`

echo "file,ARP"
for i in ../*.csv; do
    f="`basename $i .csv`"
    echo "$f,"`../../scripts/avg_r_precision.rb "$i" "$f" $1`
done | tee arp_$TIMESTAMP.csv

