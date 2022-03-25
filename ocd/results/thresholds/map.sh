#!/bin/sh

TIMESTAMP=`date +'%m%d%Y%H-%M'`

echo "file,map"
for i in ../*.csv; do
    f="`basename $i .csv`"
    echo "$f,"`../../scripts/map.rb "$i" "$f" $1`
done | tee map_$TIMESTAMP.csv
