def initialize_model(config, datasets):
    """
    Initialize models according to the configuration
    """
    if "bert" in config.model:
        model = initialize_bert_based_model(config, datasets)
    elif "gpt" in config.model:
        model = initialize_gpt2_based_model(config)
    elif "t5" in config.model:
        model = initialize_t5_based_model(config)
    elif "bow" in config.model:
        model = initialize_bow_svm(config)
    elif "tfidf" in config.model:
        model = initialize_tfidf_svm(config)
    return model


def initialize_bert_based_model(config, datasets):
    """
    Initialize BERT based model
    """
    from tlidb.examples.models.bert import Bert
    model = Bert(config, datasets)
    return model

def initialize_gpt2_based_model(config):
    """
    Initialize GPT2 based model
    """
    from tlidb.examples.models.gpt2 import GPT2
    model = GPT2(config)
    return model

def initialize_t5_based_model(config):
    """
    Initialize t5 based model
    """
    from tlidb.examples.models.t5 import T5
    model = T5(config)
    return model

def initialize_bow_svm(config):
    """
    Initialize Bag-of-Words SVM model
    """
    from tlidb.examples.models.bow import BagOfWords
    model = BagOfWords(config)
    return model

def initialize_tfidf_svm(config):
    """
    Initialize TF-IDF SVM model
    """
    from tlidb.examples.models.tfidf import Tfidf
    model = Tfidf(config)
    return model
