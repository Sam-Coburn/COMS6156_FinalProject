#!/bin/sh

# Detector using diff tool to generate patch file
# and use the number of bytes to calculate similarity
#

DETECTOR=diff

# Set the directory for the results based on the current test directory.
RESULTS=`basename $PWD | sed -e s:tests:results: | sed -e s:munich:results_munich: | sed -e s:soco:results_soco: `

# Do the comparison between two tests.
#
m_compare() {
	# call diff to generate the diff file    	
	diff -i -E -b -w -B -e $1/*.java $2/*.java > ../$RESULTS/files.diff
	# get the file size of files.diff
	diffSize=`cat ../$RESULTS/files.diff | wc -c`
	fileSize=`cat $2/*.java | wc -c`

    	#echo "$diffSize/$fileSize"
	if [ $diffSize -lt $fileSize ] ; then
	    echo "100 - (($diffSize) * 100 / $fileSize)" | bc
	else
	    echo "100"
	fi
	rm -rf files.diff
}

# Main script.
#
LANG=C
#

# Set the directory for the results based on the current test directory.
RESULTS=`basename $PWD | sed -e s:tests:results: | sed -e s:munich:results_munich: | sed -e s:soco:results_soco: ` 
#echo $RESULTS
if [ "`echo "1\t2"`" = "1\t2" ]; then   
    ECHO="-e"
else
    ECHO=""
fi

# Create the table header.
LINE="-"

for p in *; do
    for i in $p/[0-9A-Za-z]*; do # $p/test_*; do
	LINE="$LINE, $i"
    done
done
echo $LINE > ../$RESULTS/$DETECTOR.csv
count=1
# Do the pairwise comparisons.
for p in *; do
    for i in $p/[0-9A-Za-z]*; do # $p/test_*; do
	LINE="$i"
	for q in *; do
	    for j in $q/[0-9A-Za-z]*; do # $q/test_*; do
		sim="`m_compare $i $j`"
		echo $ECHO "$count: diff $sim: $i $j"
		#echo $sim
		count=$(($count+1))
		LINE="$LINE, $sim"
	    done
	done
	#break
	echo $LINE >> ../$RESULTS/$DETECTOR.csv
    done
    #break
done
# Log finish.
(echo $DETECTOR; date; uname -a) > ../$RESULTS/$DETECTOR.info
# Done.
