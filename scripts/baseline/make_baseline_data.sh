cd /Users/christianjohansen/Desktop/speciale/

mkdir tmp
# Convert fasta to one line
awk '/^>/ {printf("%s%s\t",(N>0?"\n":""),$0);N++;next;} {printf("%s",$0);} END {printf("\n");}' $1 > tmp/alpha.txt

awk '/^>/ {printf("%s%s\t",(N>0?"\n":""),$0);N++;next;} {printf("%s",$0);} END {printf("\n");}' $2 > tmp/beta.txt


# If whole TCR
# alpha = $4
# beta = $6

# If CDRs
# alpha = $4$5$6
# beta = $8$9$10

paste $3 tmp/alpha.txt tmp/beta.txt | awk '{print $1 "\t" $6 "\t" $10 "\t" $2}'

rm -r tmp