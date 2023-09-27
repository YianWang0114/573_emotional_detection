#!/bin/sh
ROOTDIR=$PWD
RESULT='results'
OUTPUTS='outputs'
PRIMARY_RESULT=$RESULT'/D4/primary/'
ADAPTATION_RESULT=$RESULT'/D4/adaptation/'
PRIMARY_OUTPUT=$OUTPUTS'/D4/primary/'
ADAPTATION_OUTPUT=$OUTPUTS'/D4/adaptation/'
DEV='devtest'
EVAL='evaltest'
D4_SCORES_OUT='D4_scores.out'

mkdir -p $PRIMARY_OUTPUT
mkdir -p $ADAPTATION_OUTPUT

mkdir -p $PRIMARY_RESULT$DEV
mkdir -p $PRIMARY_RESULT$EVAL

mkdir -p $ADAPTATION_RESULT$DEV
mkdir -p $ADAPTATION_RESULT$EVAL

cd src
pip install -r requirements.txt
pip install -e .
cd tlidb/examples

#lexicon methods for emotion detection (primary task)
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_train --do_eval --eval_best --model_config lexicon
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_train --do_eval --eval_best --model bow_svm -e 1
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_train --do_eval --eval_best --model tfidf_svm -e 1

#lexicon methods for personality_detection (adaptation task)
python3 run_experiment.py --source_datasets Friends --source_tasks personality_detection --target_datasets Friends --target_tasks personality_detection --do_train --do_eval --eval_best --model_config lexicon
python3 run_experiment.py --source_datasets Friends --source_tasks personality_detection --target_datasets Friends --target_tasks personality_detection --do_train --do_eval --eval_best --model bow_svm -e 1
python3 run_experiment.py --source_datasets Friends --source_tasks personality_detection --target_datasets Friends --target_tasks personality_detection --do_train --do_eval --eval_best --model tfidf_svm -e 1

# If you want to train the model from scratch, use this command
# python3 run_experiment.py --source_datasets Friends --source_tasks personality_detection --target_datasets Friends --target_tasks personality_detection --do_train --do_eval --eval_best --model_config bert

#If you want to load the pre-trained model for emory_emotion_recognition (primary task), use this command
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_eval --eval_best --model_config bert --saved_model_dir ./logs_and_models/PRETRAINED_Friends.emory_emotion_recognition/bert-base-uncased/

# #If you want to load the pre-trained model for personality_detection (adaptation task), use this command
python3 run_experiment.py --source_datasets Friends --source_tasks personality_detection --target_datasets Friends --target_tasks personality_detection --do_eval --eval_best --model_config bert --saved_model_dir ./logs_and_models/PRETRAINED_Friends.personality_detection/bert-base-uncased/

#run ensemble on emory_emotion_recognition task
python3 ensemble.py --primary bert --one bow_svm --two lexicon --set dev --task emotion > $ROOTDIR/$PRIMARY_RESULT$DEV/$D4_SCORES_OUT # best result
scp $ROOTDIR/$OUTPUTS/D4/emory_emotion_recognition_bert_bow_svm_lexicon_dev.txt $ROOTDIR/$PRIMARY_OUTPUT/$DEV
python3 ensemble.py --primary bert --one bow_svm --two lexicon --set test --task emotion > $ROOTDIR/$PRIMARY_RESULT$EVAL/$D4_SCORES_OUT # best result
scp $ROOTDIR/$OUTPUTS/D4/emory_emotion_recognition_bert_bow_svm_lexicon_test.txt $ROOTDIR/$PRIMARY_OUTPUT/$EVAL
python3 ensemble.py --primary bert --one tfidf_svm --two bow_svm --set dev --task emotion
python3 ensemble.py --primary bert --one tfidf_svm --two lexicon --set dev --task emotion

#run ensemble on personality_detection task
python3 ensemble.py --primary bert --one bow_svm --two lexicon --set dev --task personality > $ROOTDIR/$ADAPTATION_RESULT$DEV/$D4_SCORES_OUT # best result
scp $ROOTDIR/$OUTPUTS/D4/personality_detection_bert_bow_svm_lexicon_dev.txt $ROOTDIR/$ADAPTATION_OUTPUT/$DEV
python3 ensemble.py --primary bert --one bow_svm --two lexicon --set test --task personality > $ROOTDIR/$ADAPTATION_RESULT$EVAL/$D4_SCORES_OUT # best result
scp $ROOTDIR/$OUTPUTS/D4/personality_detection_bert_bow_svm_lexicon_test.txt $ROOTDIR/$ADAPTATION_OUTPUT/$EVAL
python3 ensemble.py --primary bert --one tfidf_svm --two bow_svm --set dev --task personality
python3 ensemble.py --primary bert --one tfidf_svm --two lexicon --set dev --task personality

#run confusion matrix analysis - emory_emotion_recognition
python3 confusion.py --method bert --set test --task emotion
python3 confusion.py --method lexicon --set test --task emotion
python3 confusion.py --method bow_svm --set test --task emotion
python3 confusion.py --method tfidf_svm --set test --task emotion
python3 confusion.py --method bert --set dev --task emotion
python3 confusion.py --method lexicon --set dev --task emotion
python3 confusion.py --method bow_svm --set dev --task emotion
python3 confusion.py --method tfidf_svm --set dev --task emotion


#run confusion matrix analysis - personality_detection
python3 confusion.py --method bert --set test --task personality
python3 confusion.py --method lexicon --set test --task personality
python3 confusion.py --method bow_svm --set test --task personality
python3 confusion.py --method tfidf_svm --set test --task personality
python3 confusion.py --method bert --set dev --task personality
python3 confusion.py --method lexicon --set dev --task personality
python3 confusion.py --method bow_svm --set dev --task personality
python3 confusion.py --method tfidf_svm --set dev --task personality

