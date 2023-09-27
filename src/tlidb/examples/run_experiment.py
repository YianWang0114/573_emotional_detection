import os
from tlidb.examples.algorithms.initializer import initialize_algorithm
from tlidb.examples.train import train,evaluate
from tlidb.examples.utils import Logger, load_datasets_split, load_algorithm, log_config, \
        set_seed, log_dataset_info, get_savepath_dir, append_to_save_path_dir, write_output
from tlidb.examples.argparser import parse_args
from tlidb.TLiDB.data_loaders.data_loaders import TLiDB_DataLoader
from tlidb.examples.utils import detach_and_clone, collate_list, concat_t_d, save_algorithm_if_needed, save_pred_if_needed
from tqdm import tqdm
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from sklearn.metrics import f1_score, accuracy_score
import pdb


def main(config):

    # if multitask, then train on both source+target tasks, and dev is target only
    if config.multitask:
        config.train_datasets = config.source_datasets+config.target_datasets
        config.train_tasks = config.source_tasks+config.target_tasks
        config.dev_datasets = config.target_datasets
        config.dev_tasks = config.target_tasks
    # if training only on source tasks, then train/dev are the same
    else:
        config.train_datasets = config.source_datasets
        config.train_tasks = config.source_tasks
        config.dev_datasets = config.source_datasets
        config.dev_tasks = config.source_tasks

    if config.target_datasets and config.target_tasks:
        # always finetune and evaluate on target tasks
        config.finetune_datasets = config.target_datasets
        config.finetune_tasks = config.target_tasks
        config.eval_datasets = config.target_datasets
        config.eval_tasks = config.target_tasks

    # create save path based only on train data
    config.save_path_dir = get_savepath_dir(config.train_datasets, config.train_tasks, config.seed, config.log_and_model_dir, config.model, config.few_shot_percent, config.multitask)

    # Initialize logs
    if os.path.exists(config.save_path_dir) and \
        (config.resume or ((config.do_finetune or config.do_eval) and (not config.do_train))):
        # if explicitly resuming, or running eval only then append to logger
        resume=True
        mode='a'
    else:
        resume=False
        mode='w'

    if not os.path.exists(config.save_path_dir):
        os.makedirs(config.save_path_dir)

    if config.debug:
        logger = Logger(mode='w')
    else:
        logger = Logger(os.path.join(config.save_path_dir, 'log.txt'), mode)

    set_seed(config.seed)
    if config.do_train:
        datasets = {}

        # load datasets for training
        datasets['train'] = load_datasets_split("train",config.train_tasks, config.train_datasets, config)
        datasets['dev'] = load_datasets_split("dev",config.dev_tasks, config.dev_datasets, config)
        if config.model_type == 'SVM' or config.model_config == 'lexicon':
            # load datasets for evaluation
            datasets['test'] = load_datasets_split("test",config.eval_tasks, config.eval_datasets, config)


        # log configuration and dataset info
        logger.write("TRAINING\n")
        log_config(config,logger)
        log_dataset_info(datasets, logger)
        if (config.model_config == 'lexicon' and config.target_tasks[0] == 'emory_emotion_recognition'):
            label_count = defaultdict(lambda: 0)
            dataloader = TLiDB_DataLoader(datasets['train'])
            pbar = tqdm(dataloader) if config.progress_bar else dataloader
            word_dict = defaultdict(lambda: defaultdict(lambda: 0))
            for batch in pbar:
                text, label, batch_metadata = batch
                for i in range(len(text)):
                    tokens = word_tokenize(text[i])
                    last_colon_index = max(index for index, item in enumerate(tokens) if item == ':')
                    tokens = tokens[last_colon_index + 1:]
                    true_label = label[i]
                    label_count[true_label] += 1

                    for j in range(len(tokens)):
                        if (tokens[j] not in '''!()-[]{};:'"\,<>./?@#$%^&*_~'''):
                            word_dict[tokens[j].lower()][true_label] += 1

            
            for split in ['dev', 'test']:
                dev_dataloader = TLiDB_DataLoader(datasets[split])
                dev_pbar = tqdm(dev_dataloader) if config.progress_bar else dev_dataloader
                prediction = []
                actual = []
                id = []
                for batch in dev_pbar:
                    text, dev_label, batch_metadata = batch
                    for i in range(len(text)):
                        tokens = word_tokenize(text[i])
                        last_colon_index = max(index for index, item in enumerate(tokens) if item == ':')
                        tokens = tokens[last_colon_index + 1:]                       
                        try:
                            true_label = dev_label[i]
                        except:
                            print('error')
                        
                        pred = defaultdict(lambda: 0)
                        for j in range(len(tokens)):              
                            if (tokens[j].lower() in word_dict):
                                total = sum(word_dict[tokens[j].lower()].values())
                                for label in word_dict[tokens[j].lower()].keys():
                                    pred[label] += word_dict[tokens[j].lower()][label]/total
                                                      
                        try:
                            pred_label = max(pred, key = pred.get)
                            # and pred_label is larger than second by a lot?
                        except:
                            pred_label = max(label_count, key = label_count.get)
                    
                        prediction.append(pred_label)
                        actual.append(true_label)
                        id.append(batch_metadata['instance_id'][i])
                assert(len(id) == len(prediction))
                write_output(config.train_tasks[0], config.model_config, split, id, prediction, logger)
                write_output(config.train_tasks[0], 'true', split, id, actual, logger)

                accuracy = accuracy_score(actual, prediction)
                micro_f1 = f1_score(actual, prediction, average='micro')
                weighted_f1 = f1_score(actual, prediction, average='weighted')
                logger.write(f"Split {split}, accuracy = {accuracy}, micro_f1 = {micro_f1}, weighted_f1 = {weighted_f1}")

            return
        
        elif (config.model_config == 'lexicon' and config.target_tasks[0] == 'personality_detection'):
            label_count = defaultdict(lambda: 0)
            dataloader = TLiDB_DataLoader(datasets['train'])
            pbar = tqdm(dataloader) if config.progress_bar else dataloader
            word_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0)))
            for batch in pbar:
                text, label, batch_metadata = batch
                for i in range(len(text)):
                    tokens = word_tokenize(text[i])
                    last_colon_index = max(index for index, item in enumerate(tokens) if item == ':')
                    tokens = tokens[last_colon_index + 1:]
                    true_label = label[i]
                    label_count[true_label] += 1
                    personality = batch_metadata['instance_id'][i].split('_')[-1]
                    for j in range(len(tokens)):
                        if (tokens[j] not in '''!()-[]{};:'"\,<>./?@#$%^&*_~'''):
                            word_dict[tokens[j].lower()][personality][true_label] += 1

            for split in ['dev', 'test']:
                dev_dataloader = TLiDB_DataLoader(datasets[split])
                dev_pbar = tqdm(dev_dataloader) if config.progress_bar else dev_dataloader
                prediction = []
                actual = []
                id = []
                per = ['aggreeable', 'conscientious', 'extroverted', 'open', 
                       'neurotic']
                for batch in dev_pbar:
                    text, dev_label, batch_metadata = batch
                    for i in range(len(text)):
                        tokens = word_tokenize(text[i])
                        # last_colon_index = max(index for index, item in enumerate(tokens) if item == ':')
                        # tokens = tokens[last_colon_index + 1:]                       
                        try:
                            true_label = dev_label[i]
                        except:
                            print('error')
                        
                        pred = defaultdict(lambda: defaultdict(lambda:0))
                        p = batch_metadata['instance_id'][i].split('_')[-1]
                        for j in range(len(tokens)):              
                            if (tokens[j].lower() in word_dict):
                                for label in word_dict[tokens[j].lower()][p].keys():
                                    try:
                                        total = sum(word_dict[tokens[j].lower()][p].values())
                                        pred[p][label] += word_dict[tokens[j].lower()][p][label]/total
                                    except:
                                        print('error')
                                        pdb.set_trace()
                        try:
                            pred_label = max(pred[p], key = pred[p].get)
                        except:
                            print('error')
                            pdb.set_trace()
                        #pdb.set_trace()
                        

                        prediction.append(pred_label)
                        actual.append(true_label)
                        id.append(batch_metadata['instance_id'][i])
                assert(len(id) == len(prediction))
                write_output(config.train_tasks[0], config.model_config, split, id, prediction, logger)
                write_output(config.train_tasks[0], 'true', split, id, actual, logger)

                accuracy = accuracy_score(actual, prediction)
                micro_f1 = f1_score(actual, prediction, average='micro')
                weighted_f1 = f1_score(actual, prediction, average='weighted')
                logger.write(f"Split {split}, accuracy = {accuracy}, micro_f1 = {micro_f1}, weighted_f1 = {weighted_f1}")

            return
 
         # initialize algorithm
        
        algorithm = initialize_algorithm(config, datasets)


        # try to resume training from a saved model
        resume_success = False
        if resume:
            if os.path.exists(os.path.join(config.save_path_dir, 'last_model.pt')):
                prev_epoch, best_val_metric = load_algorithm(algorithm, os.path.join(config.save_path_dir, 'last_model.pt'),logger)
                epoch_offset = prev_epoch + 1
                logger.write(f"Resuming training from epoch {prev_epoch} with best validation metric {best_val_metric}\n")
                resume_success = True
            else:
                logger.write("No previous model found, starting from scratch\n")

        # if not resuming, or if resuming but no previous model found, then train from scratch
        if not resume_success:
            epoch_offset=0
            best_val_metric = None

        train(algorithm, datasets, config, logger, epoch_offset, best_val_metric)

        if config.model_type == 'SVM':
            # SVM do evalution with train, skip eval
            return

    if config.do_finetune:
        assert(config.target_datasets and config.target_tasks),"Must specify target datasets and tasks to finetune"
        datasets = {}
        # get the pre-trained model path
        if config.do_train or (config.train_datasets and config.train_tasks):
            # Do nothing, this means we already have a save_dir_path and a model saved there
            pass
        elif config.saved_model_dir:
            # if user explcitily specified a pretrained model to finetune, then use that
            config.save_path_dir = config.saved_model_dir
        else:
            raise ValueError("To run fine-tuning, use:\n--saved_model_dir to specify the pre-trained model OR\
                \n--train_datasets and --train_tasks to specify the pretraining datasets and tasks")
        
        # if fine tuning, set fine-tune train, and fine-tune dev to the same tasks
        config.finetune_train_tasks = config.finetune_tasks
        config.finetune_train_datasets = config.finetune_datasets
        config.finetune_dev_tasks = config.finetune_tasks
        config.finetune_dev_datasets = config.finetune_datasets

        # load datasets for finetuning
        datasets['train'] = load_datasets_split("train",config.finetune_train_tasks,config.finetune_train_datasets,config)
        datasets['dev'] = load_datasets_split("dev", config.finetune_dev_tasks, config.finetune_dev_datasets, config)

       # initialize algorithm
        algorithm = initialize_algorithm(config, datasets)

        # always load best pretrained model
        model_path = os.path.join(config.save_path_dir, 'best_model.pt')
        is_best = True
        load_algorithm(algorithm, model_path, logger)
        epoch_offset = 0
        best_val_metric = None

        # update save path with fine-tuning details
        config.save_path_dir = append_to_save_path_dir(config.save_path_dir, config.finetune_datasets, config.finetune_tasks, config.few_shot_percent, config.seed)
        
        # note the fine-tuning in the pretrained model log
        logger.write(f"FINETUNING at {config.save_path_dir}\n")
        
        # create new logger for fine-tuning
        if not os.path.exists(config.save_path_dir):
            os.makedirs(config.save_path_dir)
        if config.debug:
            finetune_logger = Logger(mode='w')
        else:
            finetune_logger = Logger(os.path.join(config.save_path_dir, 'log.txt'), mode="w")

        # log configuration and dataset info
        finetune_logger.write("FINETUNING\n")
        finetune_logger.write(f"Loaded pretrained model from {model_path}\n")
        log_config(config,finetune_logger)
        log_dataset_info(datasets, finetune_logger)

        train(algorithm, datasets, config, finetune_logger, epoch_offset, best_val_metric)
        finetune_logger.close()

    if config.do_eval:
        assert(config.target_datasets and config.target_tasks),"Must specify target datasets and tasks to finetune"
        datasets = {}
        # If coming from training/fine-tuning, 
        #   this means we already have a save_dir_path from training/fine-tuning and a model saved there
        if config.do_finetune or config.do_train:
            pass
        elif config.saved_model_dir:
            # if user explcitily specified a model to evaluate, then use that
            config.save_path_dir = os.path.dirname(config.saved_model_dir)
        elif (config.finetune_datasets and config.finetune_tasks) and (config.train_datasets and config.train_tasks):
            # Given all the datasets and tasks, we can infer the path to the fine-tuned model
            config.save_path_dir = append_to_save_path_dir(config.save_path_dir, config.finetune_datasets, config.finetune_tasks, config.few_shot_percent, config.seed)
        else:
            raise ValueError("To run evaluation, use:\n--saved_model_dir to specify the model to evaluate OR\
                \n--finetune_datasets and --finetune_tasks and --train_datasets and --train_tasks to infer the path to the model")

        # ensure user has specified a model to evaluate
        assert(not(config.eval_last and config.eval_best)), "cannot evaluate both last and best models"
        assert(config.eval_last or config.eval_best), "must evaluate at least one model"
        
        # create logger for evaluation
        if config.debug:
            eval_logger = Logger(mode='w')
        else:
            eval_logger = Logger(os.path.join(config.save_path_dir, 'log.txt'), mode="a")
        eval_logger.write("EVALUATING\n")
        
        # load datasets for evaluation
        datasets['dev'] = load_datasets_split("dev",config.eval_tasks, config.eval_datasets, config)
        datasets['test'] = load_datasets_split("test",config.eval_tasks, config.eval_datasets, config)

        # log configuration and dataset info
        log_config(config,eval_logger)
        log_dataset_info(datasets, eval_logger)

        # initialize algorithm
        algorithm = initialize_algorithm(config, datasets)  
        
        # load evaluation model
        if config.eval_last:
            eval_model_path = os.path.join(config.save_path_dir, "last_model.pt")
            is_best = False
        else:
            eval_model_path = os.path.join(config.save_path_dir, 'best_model.pt')
            is_best = True

        epoch, best_val_metric = load_algorithm(algorithm, eval_model_path,eval_logger)
        evaluate(algorithm, datasets, config, eval_logger, epoch, is_best)

        eval_logger.close()
    logger.close()

if __name__ == "__main__":
    config = parse_args()
    nltk.download('punkt')
    main(config)
