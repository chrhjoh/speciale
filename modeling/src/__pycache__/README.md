# Source code for modelling

Files in this folder contains the different files responsible for modelling of the data and performance evaluation of these.

- `attention_net.py`, `cdr_network.py`, `tcr_network` contains the acutal network classes used in this project. (`deep_learning_network.py` contains the network used for the deep learning course at DTU compute)
- `train_cnn.py`, `train_lstm_attention.py` and `train_subsample_model.py` are scripts used for training models, saving models for later use and save evaluation scores for performance evaluation. 
- `clean_cv_output.py` simply averages scores generated during nested crossvalidation
- `calc_partition_auc.py` takes the scores generated from training scripts or the cleaned crossvalidation output and calculate AUC metrics.
- R scripts uses the calculated AUCs to create plots for the thesis.
- `pretrain_model.py` contains code to pretrain a model uding only a single peptide. Used to check if pretraining using increasing amounts of data can improve predictions.
- `utils.py` contains code for the Runner and Dataset classes to use when training models.
- `do_5cv.sh` simply runs crossvalidation using a train script.