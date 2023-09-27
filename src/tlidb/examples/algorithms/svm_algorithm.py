from collections import defaultdict
from sklearn.svm import SVC
from tlidb.TLiDB.data_loaders.data_loaders import TLiDB_DataLoader
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import f1_score, accuracy_score

from tlidb.examples.models.initializer import initialize_model
from tqdm import tqdm

INSTANCE_ID_KEY = 'instance_id'
PERSONALITIES = ['open', 'neurotic', 'extroverted', 'conscientious','aggreeable']

class SVMAlgorithm:
    def __init__(self, config, datasets):
        super().__init__()
        self.svm = dict()
        self.model = initialize_model(config, datasets)
        if (config.train_tasks[0] == 'emory_emotion_recognition'):
            self.svm['svm'] = SVC(kernel='linear')
        elif (config.train_tasks[0] == 'personality_detection'):
            for personality in PERSONALITIES:
                self.svm[personality] = SVC(kernel='rbf')
        
    def svm_fit(self, config, datasets):
        instance_ids, X_train, Y_train = self.dataloader_to_XY(config, datasets)
        X_vec_train = self.model.vectorizer.fit_transform(X_train)
        if len(self.svm) == 1:
            key = list(self.svm.keys())[0]
            self.svm[key].fit(X_vec_train, Y_train)
        else:
            X_train_dict = dict()
            Y_train_dict = dict()
            X_vec_train_dict = dict()
            for i, instance_id in enumerate(instance_ids):
                personality = instance_id.split("_")[-1]
                if personality not in X_train_dict:
                    X_train_dict[personality] = []
                    Y_train_dict[personality] = []
                X_train_dict[personality].append(X_train[i])
                Y_train_dict[personality].append(Y_train[i])
            for key in PERSONALITIES:
                X_vec_train_dict[key] = self.model.vectorizer.fit_transform(X_train_dict[key])
                self.svm[key].fit(X_vec_train_dict[key], Y_train_dict[key])


    def svm_eval(self, config, datasets):
        instance_ids, X_test, Y_test = self.dataloader_to_XY(config, datasets)
        X_vec_test = self.model.vectorizer.transform(X_test)
        if len(self.svm) == 1:
            key = list(self.svm.keys())[0]
            y_pred = self.svm[key].predict(X_vec_test)
            micro_f1 = f1_score(Y_test, y_pred, average='micro')
            weighted_f1 = f1_score(Y_test, y_pred, average='weighted')
            return (micro_f1, weighted_f1), (instance_ids, y_pred)
        else:
            y_pred = []
            for i, instance_id in enumerate(instance_ids):
                personality_to_predict = instance_id.split("_")[-1]
                cur_vector = X_vec_test[i]
                cur_y_pred = self.svm[personality_to_predict].predict(cur_vector)
                y_pred.extend(cur_y_pred)
            micro_f1 = f1_score(Y_test, y_pred, average='micro')
            weighted_f1 = f1_score(Y_test, y_pred, average='weighted')
            return (micro_f1, weighted_f1), (instance_ids, y_pred)

    def dataloader_to_XY(self, config, datasets):
        INSTANCE_IDS, X, Y = [], [], []
        dataloader = TLiDB_DataLoader(datasets)
        pbar = tqdm(dataloader) if config.progress_bar else dataloader
        for batch in pbar:
            text_list, label_list, batch_metadata = batch
            instance_ids = batch_metadata[INSTANCE_ID_KEY]
            INSTANCE_IDS.extend(instance_ids)
            Y.extend(label_list)
            for i, text in enumerate(text_list):
                tokens = word_tokenize(text)
                if (config.train_tasks[0]=='emory_emotion_recognition'):
                    last_colon_index = max(index for index, item in enumerate(tokens) if item == ':')
                    tokens = tokens[last_colon_index + 1:]
                X.append(' '.join(tokens))
        return INSTANCE_IDS, X, Y



