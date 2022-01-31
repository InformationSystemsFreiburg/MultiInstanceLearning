# Multi-Instance Learning for Sentence Label Polarity Classification in Product Reviews

Requirements:
  
  * Tensorflow 1.8.0
  * bert-embedding 1.0.1
 
 The main file is MultiInstanceLearning.py. It uses helper functions from MultiInstanceLearningHelper.py for loading and storing data and Tensorflow.py to perform the training of the MIL model.
 
 The data needs to be organized as follows:
 
 Training reviews:<br>
  &nbsp;&nbsp;&nbsp;&nbsp; data/train/pos --> put all positive reviews for training here <br>
   &nbsp;&nbsp;&nbsp;&nbsp; data/train/neg --> put all negative reviews for training here
  
 Out-of-sample reviews used for analyses:<br>
   &nbsp;&nbsp;&nbsp;&nbsp; data/other --> put all reviews for subsequent analyses here
  
  Each review file should be stored with the name \<unique review id\>.txt and contain one sentence per line. We recommend using https://stanfordnlp.github.io/CoreNLP/ for sentence splitting.
  
  The manually-labeled dataset is stored at data/test_sentences.csv. It contains one labeled sentence and the corresponding label per line.
  
  The whole process can be started by
  python3 MultiInstanceLearning.py
  
  The script will first download the pre-trained model, generate all BERT embeddings and store the resulting data using pickle as data/data_bert.pkl. Subsequently, the MIL will be trained and the predicted insample and out-of-sample sentence labels are stored in the files data/insample_doc_predictions.tsv and data/oos_doc_predictions.tsv.
