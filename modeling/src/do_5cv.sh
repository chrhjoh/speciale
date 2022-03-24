CV=5
for ((part=0; part<$CV; part++)); do
    python train_net.py $part

done

