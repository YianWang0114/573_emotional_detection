#!/bin/sh
cd src
pip install -r requirements.txt
pip install -e .
cd tlidb/examples

#lexicon methods
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_train --do_eval --eval_best --model_config lexicon
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_train --do_eval --eval_best --model bow_svm -e 1
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_train --do_eval --eval_best --model tfidf_svm -e 1

# If you want to train the model from scratch, use this command
# python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_train --do_eval --eval_best --model_config bert

#If you want to load the pre-trained model, use this command
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_eval --eval_best --model_config bert --saved_model_dir ./logs_and_models/PRETRAINED_Friends.emory_emotion_recognition/bert-base-uncased/


cd "$( dirname -- "$0"; )"
cd ..
mkdir -p results/D3/
scp src/tlidb/examples/logs_and_models/*/*/log.txt results/D3/
mv results/D3/log.txt results/D3/D3_scores.out

