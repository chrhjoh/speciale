cd /home/projects/vaccine/people/chrhol/speciale
# Input 1 is file 
# input 2 in partition for test
mkdir data/tmp

grep "^P$2" $1 | cut -f2,3,4 > data/tmp/test.input
grep "^P$2" $1 | cut -f2,3,4 > data/tmp/database.input

../../morni/bin/pairlistscore_db_kernel data/tmp/test.input data/tmp/database.input >> results/baseline.res

rm -r data/tmp