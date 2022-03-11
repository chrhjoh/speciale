DIR=/home/projects/vaccine/people/chrhol/speciale/
IN_FILE=$DIR/partitioning_pipeline/partitions/train_data_all.csv
OUT_FILE=$DIR/baseline/out/baseline_all.out
TMP_DIR=$DIR/data/TMP

mkdir $TMP_DIR
module load tools
module load anaconda3/4.4.0

python3 $DIR/scripts/baseline/run_baseline.py $IN_FILE > $OUT_FILE

rm -r $TMP_DIR
