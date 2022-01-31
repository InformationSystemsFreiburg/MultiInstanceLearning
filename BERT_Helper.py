from bert_embedding import BertEmbedding
import numpy as np

'''
Add a column of ones to a given matrix
[0,1] --> [1,0,1]
[1,2]     [1,1,2]
'''

def augment_matrix(matrix):
    return np.column_stack((np.ones(matrix.shape[0]), matrix))

'''
Generate BERT embeddings for given documents and sentences for out-of-sample prediction.

Parameters:

* documents: dictionary with the following information
             Key: document ID
             Value: "sentences": contains the ordered list of sentences in the document
             The sentence embeddings will be stored under a new value given by the parameter feature_name

* oos_sentences: list of sentences for which embeddings should also be generated

* feature_name: name under which the sentence embeddings for each document will be stored in the documents dictionary

* augment_feature: if True, add a column consisting of ones only to each embedding matrix. This should be set to True when training a logistic regression model.

Returns:

* documents: dictionary with sentence embeddings stored under the name given by parameter feature_name
* oos_sentences_embeddings: embedding matrix of sentences used for out-of-sample evaluation
'''

def generate_BERT_embeddings(documents, oos_sentences, feature_name="bert_embedding", augment_feature=True):
    # load pre-trained BERT model
    bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased')

    progress = 0
    for doc_id in documents:
        progress += 1
        if progress % 10000 == 0:
            print("Processed documents", progress)
        embeddings_group = bert_embedding(documents[doc_id]["sentences"])

        # calculate sentence embeddings as average of word embeddings
        embeddings = [np.mean(np.array(x[1]), axis=0) for x in embeddings_group]
        embeddings = np.array(embeddings)

        # add new entry to dictionary
        documents[doc_id][feature_name] = embeddings

        # add a column of ones for logistic regression bias
        if augment_feature:
            documents[doc_id][feature_name] = augment_matrix(documents[doc_id][feature_name])


    # extract embeddings for sentences used for out-of-sample evaluation
    
    oos_sentences_embeddings = bert_embedding(oos_sentences)
    oos_sentences_embeddings = np.array([np.mean(np.array(x[1]), axis=0) for x in oos_sentences_embeddings])

    # we also need to augment the sentence embeddings
    if augment_feature:
        oos_sentences_embeddings = augment_matrix(oos_sentences_embeddings)
    return documents, oos_sentences_embeddings

# test the function
if __name__ == "__main__":
    _, emb = generate_BERT_embeddings(dict(), ["This product quality is great", "I bought this product last year", "The product is good"])
    print(emb.shape)

