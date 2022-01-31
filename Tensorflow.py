import tensorflow as tf
import numpy as np

# similarity function
from sklearn.metrics.pairwise import rbf_kernel

"""
Class to train MIL model using tensorflow
"""
class TensorflowMIL:
    def __init__(self, documents):
        self.documents = documents
        doc_ids = list(self.documents.keys())
        self.firstKey = doc_ids[0]
        self.n_docs = len(documents)
        self.resetBatches()

    # create a batch from groups with given indices
    def get_next_batch(self, feature_name="bert_embedding"):
        # sample next doc id from remaining ones
        batch_index = np.random.choice(self.remaining_doc_ids, 1, replace=False)[0]

        # remove these datapoints
        self.remaining_doc_ids.remove(batch_index)
        return (self.documents[batch_index][feature_name], self.documents[batch_index]["label"])

    # reset remaining indices for optimization
    def resetBatches(self):
        self.all_doc_ids = list(self.documents.keys())
        self.remaining_doc_ids = list(self.documents.keys())

    # rbf similarity between all vectors X
    def calculate_similarity(self, X):
        return rbf_kernel(X)

    # main function
    def optimizeTheta(self, document_error_weight=10, epochs=64, feature_name="bert_embedding", eps=0.001, acc_callback=None, log_file=None):
        # reset tf graph
        tf.reset_default_graph()
        print("Building Tensorflow Graph...", end='')

        # dimension of embeddings
        feature_dim = self.documents[self.firstKey][feature_name].shape[1]
        print("Feature Size: "+str(feature_dim))

        # embeddings of sentences
        X = tf.placeholder(tf.float32, shape=(None, feature_dim))
        document_label = tf.placeholder(tf.float32)

        # theta: coefficients of logistic regression
        theta = tf.get_variable(name="theta", shape=(feature_dim,1), dtype=tf.float32, initializer=tf.random_normal_initializer, trainable=True)

        # prediction for all instances
        pred_x = tf.sigmoid(tf.matmul(X, theta))

        # document_error
        document_prediction = tf.reduce_mean(pred_x)
        document_error = tf.pow(document_label - document_prediction, 2)

        # multiply document error with lambda parameter
        document_error = tf.cast(document_error_weight, tf.float32) * document_error

        # differences between predictions
        pred_x_matrix = tf.expand_dims(pred_x, axis=1)
        pred_x_diff = tf.subtract(pred_x_matrix, tf.transpose(pred_x_matrix))
        pred_x_diff_squared = tf.pow(pred_x_diff, 2)

        pred_x_diff_squared = tf.squeeze(pred_x_diff_squared)
        XSim = tf.placeholder(tf.float32, shape=(None, None))

        # instances error
        instancesError = tf.reduce_sum(tf.multiply(pred_x_diff_squared, XSim)) / tf.cast((tf.shape(pred_x)[0] ** 2), tf.float32)
        loss = tf.add(instancesError, document_error)

        # Adam optimizer

        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(loss, var_list=[theta])

        print("done")
        init = tf.global_variables_initializer()

        if log_file is not None:
            log_file_fp = open(log_file, "w")
        with tf.Session() as session:
            session.run(init)
            theta_val = None
            old_theta_val = None
            loss_val = None
            old_loss_val = None
            highest_group_acc = 0
            best_theta = (None, None)

            print("Starting MIL optimization")
            epoch = 0
            converged = False
            while epoch < epochs and not converged:
                self.resetBatches()
                avg_loss = 0
                epoch += 1

                # go over all documents
                while len(self.remaining_doc_ids) > 0:
                    batch_instances, label = self.get_next_batch(feature_name=feature_name)
                    x_sim = self.calculate_similarity(batch_instances)
                    feed_dict = {document_label: label, X: batch_instances, XSim: x_sim}
                    # train coefficients
                    session.run(train, feed_dict=feed_dict)
                    # calculate loss
                    loss_val = session.run(loss, feed_dict=feed_dict)
                    avg_loss += loss_val

                # avg loss is used to find best theta
                avg_loss = avg_loss / self.n_docs
                theta_val = session.run(theta, feed_dict=feed_dict)
                acc_insample = acc_callback(self.documents, theta_val)
                if acc_insample > highest_group_acc:
                    highest_group_acc = acc_insample
                    best_theta = (theta_val, avg_loss)

                print("Epoch "+str(epoch)+"/"+str(epochs)+": Loss: "+str(np.round(avg_loss, 3))+ " in-sample document accuracy: " + str(acc_insample))
                if log_file is not None:
                    log_file_fp.write("{}\t{}\t{}\n".format(epoch, avg_loss, acc_insample))
                    log_file_fp.flush()

                # check for convergence
                if old_theta_val is not None:
                    if np.linalg.norm(old_theta_val - theta_val) < eps:
                        print("Reached convergence for parameter norm of theta")
                        converged = True
                        break
                else:
                    old_theta_val = theta_val

                # store theta with greatest in-sample accuracy
                if acc_insample > highest_group_acc:
                    highest_group_acc = acc
                    best_theta = (theta_val, avg_loss)

            session.close()
        if log_file is not None:
            log_file_fp.close()
        return best_theta

