#!/bin/sh
cd src
pip install -r requirements.txt
pip install -e .
cd tlidb/examples

# If you want to train the model from scratch, use this command
# python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_train --do_eval --eval_best --model_config bert

# If you want to load the pre-trained model, use this command
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_eval --eval_best --model_config bert --saved_model_dir ./logs_and_models/PRETRAINED_Friends.emory_emotion_recognition/bert-base-uncased/

cd "$( dirname -- "$0"; )"
cd ..
mkdir -p results/D2/
scp src/tlidb/examples/logs_and_models/*/*/log.txt results/D2/
mv results/D2/log.txt results/D2/D2_scores.out
