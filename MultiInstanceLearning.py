import sys, os, time, math, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from MultiInstanceLearningHelper import MultiInstanceLearningHelper
from Tensorflow import TensorflowMIL


""" Class for Multi Instance Learning """

class MultiInstanceLearning():
    def __init__(self, theta=None):
        self.helper = MultiInstanceLearningHelper()

    # evaluate OOS performance of classifier
    # calculate overall accuracy and accuracy on binary if model is non-neutral according to threshold

    def evaluate_oos_prediction(self, theta, X_test, y_test, c_neutral=0):
        # accuracy calculation
        y_pred, _ = self.prediction_neutral_threshold(X_test, theta, c_neutral=c_neutral)
        acc_all = accuracy_score(y_test, y_pred)

        y_pred_no_neutral = y_pred[np.where(y_test != 0)]
        y_test_no_neutral = y_test[np.where(y_test != 0)]

        y_test_no_neutral_reject = y_test_no_neutral[np.where(y_pred_no_neutral != 0)]
        y_pred_no_neutral_reject = y_pred_no_neutral[np.where(y_pred_no_neutral != 0)]
        acc_pred_no_neutral_reject = accuracy_score(y_test_no_neutral_reject, y_pred_no_neutral_reject)

        return acc_all, acc_pred_no_neutral_reject


    # the sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # evaluate accuracy at document level
    def evaluate_insample_document_accuracy(self, docs, theta):
        doc_predictions = self.predict_sentences(docs, theta)
        y_true = []
        y_pred = []

        for doc_id in doc_predictions:
            # avg of instances
            doc_label = np.mean(doc_predictions[doc_id][1])
            if doc_label >= 0.5:
                doc_label = 1
            else:
                doc_label = 0
            y_pred.append(doc_label)
            y_true.append(docs[doc_id]["label"])

        # return acc on all, binary and on neutral
        return accuracy_score(y_true, y_pred)

    # make predictions for sentence_embeddings with neutral threshold
    def prediction_neutral_threshold(self, sentence_embeddings, theta, c_neutral=0):
        s = self.sigmoid(sentence_embeddings.dot(theta)).flatten()
        
        # init zeros: neutral
        prediction = np.zeros(len(s), dtype=np.int32)

        # positive
        prediction[np.where(s > 0.5 + c_neutral)] = 1

        #negative
        prediction[np.where(s < 0.5 - c_neutral)] = -1
        
        return prediction, s

    """
    Predict all sentences in documents
    Returns:
    predictions_dict: dictionary with key doc_id and value (sentences, (predictions_binary, predictions_continuous))
    """

    def predict_sentences(self, documents, theta, feature_name="bert_embedding", c_neutral=0):
        predictions_dict = dict()
        for doc_id in documents:
            predictions_dict[doc_id] = (documents[doc_id]["sentences"], self.prediction_neutral_threshold(documents[doc_id][feature_name], theta, c_neutral=c_neutral))
        return predictions_dict

    # optimize theta with given parameters
    def learn_coefficients(self, train_docs, document_error_weight=10, batch_size=1, epochs=32, feature_name="bert_embedding", eps=0.0001, acc_callback=None, log_file=None):
        tf = TensorflowMIL(train_docs)
        theta, loss = tf.optimizeTheta(document_error_weight=document_error_weight, epochs=epochs, feature_name=feature_name, eps=eps, acc_callback=acc_callback, log_file=log_file)
        return theta, loss

if __name__ == "__main__":
    helper = MultiInstanceLearningHelper()
    mil = MultiInstanceLearning()
    feature_name = "bert_embedding"

    # neutrality threshold
    c_neutral = 0.458

    # load data
    train_docs, other_docs, X_test, y_test = helper.load_data()
    y_test = np.array(y_test)
    print("MIL data")
    num_instances = 0
    for doc_id in train_docs:
        num_instances += len(train_docs[doc_id]["sentences"])
    print("Train documents", len(train_docs), "sentences", num_instances)
    print(y_test)    
    print("Sentences test data", len(X_test))

    theta_file = "data/theta.pkl"

    # check if MIL model trained already
    # if yes: load coefficients, if no: train MIL model and store coefficients

    if os.path.exists(theta_file):
        print("Loading coefficients")
        theta = pickle.load(open(theta_file, "rb"))
    else:
        acc_callback = mil.evaluate_insample_document_accuracy
        print("Train MIL on", len(train_docs), "documents")
        # training starts here
        theta, loss = mil.learn_coefficients(train_docs, epochs=32, acc_callback=acc_callback, log_file="data/log_mil_train.tsv")
        
        print("Storing coefficients")
        pickle.dump(theta, open(theta_file, "wb"))

    # store insample and out-of-sample sentence labels
    print("Storing in-sample sentence predictions...")
    doc_predictions = mil.predict_sentences(train_docs, theta, feature_name=feature_name, c_neutral=c_neutral)

    insample_pred_file = "data/insample_doc_predictions.tsv"
    helper.store_predictions(doc_predictions, insample_pred_file)

    print("OOS Performance")
    print("Accuracy, Accuracy (non-neutral)", mil.evaluate_oos_prediction(theta, X_test, y_test, c_neutral=c_neutral))
    print("Storing out-of-sample sentence predictions...")
    oos_pred_file = "data/oos_doc_predictions.tsv"
    doc_predictions = mil.predict_sentences(other_docs, theta, feature_name=feature_name, c_neutral=c_neutral)
    helper.store_predictions(doc_predictions, oos_pred_file)

