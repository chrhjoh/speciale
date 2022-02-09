cd /Users/christianjohansen/Desktop/speciale
# Input 1 is file 
# input 2 in partition for test
mkdir data/tmp

grep "^P$2" $1 | cut -f2,3,4 > data/tmp/test.input
grep -v "^P$2" $1 | cut -f2,3,4 > data/tmp/database.input

#tools/pairlistscore_kernel -blf data/blosum/BLOSUM50 -blqij data/blosum/blosum62.qij data/tmp/test.input data/tmp/database.input >> results/baseline.res

#rm -r data/tmp