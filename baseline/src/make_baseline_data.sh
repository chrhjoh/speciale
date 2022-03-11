cd /Users/christianjohansen/Desktop/speciale/

mkdir tmp
# Convert fasta to one line
awk '/^>/ {printf("%s%s\t",(N>0?"\n":""),$0);N++;next;} {printf("%s",$0);} END {printf("\n");}' $1 > tmp/alpha.txt

awk '/^>/ {printf("%s%s\t",(N>0?"\n":""),$0);N++;next;} {printf("%s",$0);} END {printf("\n");}' $2 > tmp/beta.txt

awk '/^>/ {printf("%s%s\t",(N>0?"\n":""),$0);N++;next;} {printf("%s",$0);} END {printf("\n");}' $3 > tmp/peptide.txt

# If whole TCR
# alpha = $6
# beta = $8

# If CDRs
# alpha = $6$7$8
# beta = $10$11$12

paste $4 tmp/peptide.txt tmp/alpha.txt tmp/beta.txt | awk '{print $1 "\t" $4 "\t" $8 "\t" $12 "\t" $2}'

#rm -r tmp
