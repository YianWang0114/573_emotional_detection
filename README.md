# LING573-project
## Lexicon method
Command to run lexicon method:
```sh
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_train --do_eval --eval_best --model_config lexicon
```
## BagOfWords SVM
Command to run BagOfWords SVM:
```sh
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_train --do_eval --eval_best --model bow_svm -e 1
```
## TF-IDF SVM
Command to run TF-IDF SVM:
```sh
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_train --do_eval --eval_best --model tfidf_svm -e 1
```
## Ensemble 
Command to run ensemble:
```sh
python3 ensemble.py --primary method0 --one method1 --two method2 
where method1, method2, method3 is from [tfidf_svm, lexicon, bow_svm, bert] 
```

## BERT model
Training the model from scratch takes >5 hours on T4 GPU on google colab or >2.5hours on A100 GPU on google colab.
*Recommend: load the pre-trained model for evaluation*

### BERT (train the model from scratch)
Command to run Bert (train the model from scratch):

#### For Emotion Detion task:
```sh
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_train --do_eval --eval_best --model bert-base-uncased --learning_rate 1e-5 --optimizer AdamW --effective_batch_size 40 -e 20
```
#### For Personality Detection task:
```sh
python3 run_experiment.py --source_datasets Friends --source_tasks personality_detection --target_datasets Friends --target_tasks personality_detection --do_train --do_eval --eval_best --model bert-base-uncased --learning_rate 1e-5 --optimizer AdamW --effective_batch_size 40 -e 20
```
### BERT (load pre-trained BERT model)
Command to load pre-trained Bert model [recommended if only checking the evaluation output]:

#### For Emotion Detion task:
```sh
python3 run_experiment.py --source_datasets Friends --source_tasks emory_emotion_recognition --target_datasets Friends --target_tasks emory_emotion_recognition --do_eval --eval_best --model_config bert --saved_model_dir ./logs_and_models/PRETRAINED_Friends.emory_emotion_recognition/bert-base-uncased/
```
#### For Personality Detection task:
```sh
python3 run_experiment.py --source_datasets Friends --source_tasks personality_detection --target_datasets Friends --target_tasks personality_detection --do_eval --eval_best --model_config bert --saved_model_dir ./logs_and_models/PRETRAINED_Friends.personality_detection/bert-base-uncased/
```

#### Pre-trained Bert model checkpoint download (Emotion Detection task):
[https://drive.google.com/drive/folders/1Zbe2CWBPT-Uh2aIjWwnLnxsV8pb0YVzK](https://drive.google.com/drive/folders/1Zbe2CWBPT-Uh2aIjWwnLnxsV8pb0YVzK?usp=share_link)
- Download the `best_model.pt`
- put it into the `--saved_model_dir` stated (i.e. `./logs_and_models/PRETRAINED_Friends.emory_emotion_recognition/bert-base-uncased/`)
- run the command to load pre-trained Bert model

#### Pre-trained Bert model checkpoint download (Personality Detection task):
[https://drive.google.com/drive/folders/10xJ7ZmY_d0bJhWmA0zy1Ofn5zxfa5VIf](https://drive.google.com/drive/folders/10xJ7ZmY_d0bJhWmA0zy1Ofn5zxfa5VIf)
- Download the `best_model.pt`
- put it into the `--saved_model_dir` stated (i.e. `./logs_and_models/PRETRAINED_Friends.personality_detection/bert-base-uncased/`)
- run the command to load pre-trained Bert model

