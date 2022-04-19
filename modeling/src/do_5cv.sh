CV=5
for ((test_part=1; test_part<=$CV; test_part++)); do
    for ((val_part=1; val_part<=$CV; val_part++)); do

        if [ $test_part != $val_part ]; then
            python train_lstm_attention.py $test_part $val_part
        fi
    done
done

