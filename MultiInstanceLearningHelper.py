import sys, os, pickle
import numpy as np
import pandas as pd
from BERT_Helper import generate_BERT_embeddings

""" Helper class for loading and storing data for multi-instance learning on product reviews """
class MultiInstanceLearningHelper:

    # parse test file sentence tab sentiment
    # first column: text
    # second column: label
    def load_test_file(self, test_file):
        df = pd.read_csv(test_file)
        sentences_text = list(df["sentence"])
        y_test = list(df["label"])
        return sentences_text, y_test

    """ 
    Store the predictions to an outputFile
    Parameters:
    * document_predictions: dictionary with key=document id, value dictionary with following keys: (sentences, (predictions_binary, predictions_cont))
    * output_file: file to store predictions
    """
    def store_predictions(self, document_predictions, output_file):
        with open(output_file, "w") as fp:
            fp.write("doc_id\tsentence\tprediction\tprediction_cont\tsentence_number\tnum_sentences\n")
            for doc_id in document_predictions:
                for i in range(len(document_predictions[doc_id][0])):
                    sentence = document_predictions[doc_id][0][i]
                    prediction_cont = document_predictions[doc_id][1][1][i]
                    prediction_bin = document_predictions[doc_id][1][0][i]
                    fp.write(str(doc_id)+"\t"+sentence+"\t"+str(prediction_bin)+"\t"+str(prediction_cont)+"\t" +str(i+1)+"\t"+str(len(document_predictions[doc_id][0]))+"\n")
            fp.close()

    # load documents as stored in doc_folder with pos and neg subfolders
    def load_train_docs(self, doc_folder):
        docs = dict()
        for f in os.listdir(doc_folder+"pos"):
            doc_id = f.replace(".txt", "")
            doc_file = doc_folder+"pos/"+f
            with open(doc_file, "r") as fp:
                sentences = fp.read().splitlines()
            docs[doc_id] = dict()
            docs[doc_id]["label"] = 1
            docs[doc_id]["sentences"] = sentences
        for f in os.listdir(doc_folder+"neg"):
            doc_id = f.replace(".txt", "")
            doc_file = doc_folder+"neg/"+f
            with open(doc_file, "r") as fp:
                sentences = fp.read().splitlines()
            docs[doc_id] = dict()
            docs[doc_id]["label"] = 0
            docs[doc_id]["sentences"] = sentences
        return docs

    # load all documents in doc_folder
    def load_other_docs(self, doc_folder):
        docs = dict()
        for f in os.listdir(doc_folder):
            doc_id = f.replace(".txt", "")
            doc_file = doc_folder+f
            with open(doc_file, "r") as fp:
                sentences = fp.read().splitlines()
            docs[doc_id] = dict()
            docs[doc_id]["sentences"] = sentences

        return docs

    """
    load all data
    Returns: 
        train_docs, other_docs (dictionaries with key doc_id and entries "sentences" -> list of sentences, and label: 0 for negative, 1 for postive)
        X_test, y_test: sentences
    """
    def load_data(self, data_folder="data/"):
        # load train docs
        train_docs = self.load_train_docs(data_folder+"train/")

        # load other docs
        other_docs = self.load_other_docs(data_folder+"other/")

        # load test sentences
        sentences_test_file = data_folder+"test_sentences.csv"
        test_sentences, y_test = self.load_test_file(sentences_test_file)

        # generate embeddings only once to save time in the future
        bert_file = data_folder + "data_bert.pkl"
        if os.path.exists(bert_file):
            train_docs, other_docs, X_test, y_test = pickle.load(open(bert_file, "rb"))
        else:
            # generate BERT
            print("Generating BERT embeddings...")
            train_docs, X_test = generate_BERT_embeddings(train_docs, test_sentences)
            other_docs, _ = generate_BERT_embeddings(other_docs, test_sentences)
            pickle.dump((train_docs, other_docs, X_test, y_test), open(bert_file, "wb"))
        return train_docs, other_docs, X_test, y_test

