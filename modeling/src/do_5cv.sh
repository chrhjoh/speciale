CV=5
model_num=1
total_models=$CV*($CV-1)
for ((test_part=1; test_part<=$CV; test_part++)); do
    for ((val_part=1; val_part<=$CV; val_part++)); do
        if [ $test_part != $val_part ]; then
            python train_lstm_attention.py $test_part $val_part
            echo "done $model_num out of $total_models"
            $model_num += 1
        fi
    done

done

