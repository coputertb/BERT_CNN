# BERT_CNN_similarity
# Xay dung mo hinh danh gia do tuong tu giua 2 cau hoi (question1 va question2) dua tren BERT_CNN
import os
import re
import csv
import math
import codecs
import numpy
import numpy as np
import pandas as pd
import gensim
import chardet
import keras
import keras.backend as K
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
from tqdm import tqdm
from tensorflow.keras import backend as K
from string import punctuation
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Merge, Lambda
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Concatenate, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from imp import reload
from gensim.models import Word2Vec
from keras.layers import Bidirectional
import metrics
from fuzzywuzzy import fuzz
from tqdm import tqdm
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from scipy.stats import skew, kurtosis
from nltk import word_tokenize
from sklearn.preprocessing import StandardScaler
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
from keras.layers.core import Reshape, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Conv1D
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import LearningRateScheduler

stop_words = stopwords.words('english')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
import sys
reload(sys)
#sys.setdefaultencoding('utf-8')

########################################
MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 20000
batch_size = 128
epochs = 1

# Initialize session
# sess = tf.Session()
sess = tf.compat.v1.Session()
tf.compat.v1.disable_eager_execution()

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def create_tokenizer_from_hub_module(bert_path):
    """Get the vocab file and casing info from the Hub module."""
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
    )
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(text, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(text, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples

########################################
def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

######################################################################################
class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="mean",
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                "Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name="{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                "Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append("encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError("Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

##########################################################################

def main():
    # Params for bert model and tokenization
    bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    max_seq_length = 256

    train_df = pd.read_csv('D:/BERT/BERT_CNN/train100.csv', encoding='windows-1254')
    test_df = pd.read_csv('D:/BERT/BERT_CNN/test100.csv', encoding='windows-1254')
    print(train_df.head(4))

    # Create datasets (Only take up to max_seq_length words for memory)
    
    train_text_1 = train_df["question1"].tolist()
    train_text_1 = [" ".join(t.split()[0:max_seq_length]) for t in train_text_1]
    train_text_1 = np.array(train_text_1, dtype=object)[:, np.newaxis]
    
    train_text_2 = train_df["question2"].tolist()
    train_text_2 = [" ".join(t.split()[0:max_seq_length]) for t in train_text_2]
    train_text_2 = np.array(train_text_2, dtype=object)[:, np.newaxis]
    train_labels = train_df["labels"].tolist()

    test_text_1 = test_df["question1"].tolist()
    test_text_1 = [" ".join(t.split()[0:max_seq_length]) for t in test_text_1]
    test_text_1 = np.array(test_text_1, dtype=object)[:, np.newaxis]

    test_text_2 = test_df["question2"].tolist()
    test_text_2 = [" ".join(t.split()[0:max_seq_length]) for t in test_text_2]
    test_text_2 = np.array(test_text_2, dtype=object)[:, np.newaxis]
    test_labels = test_df["labels"].tolist()
    print(test_text_2.shape)

    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module(bert_path)

    # Convert data to InputExample format
    train_examples_1 = convert_text_to_examples(train_text_1, train_labels)
    train_examples_2 = convert_text_to_examples(train_text_2, train_labels)
    
    test_examples_1 = convert_text_to_examples(test_text_1, test_labels)
    test_examples_2 = convert_text_to_examples(test_text_2, test_labels)

    # Convert to features
    (
        train_input_ids_1,
        train_input_masks_1,
        train_segment_ids_1,
        train_labels,
    ) = convert_examples_to_features(
        tokenizer, train_examples_1, max_seq_length=max_seq_length
    )
    
    (
        train_input_ids_2,
        train_input_masks_2,
        train_segment_ids_2,
        train_labels,
    ) = convert_examples_to_features(
        tokenizer, train_examples_2, max_seq_length=max_seq_length
    )

    
    (
        test_input_ids_1,
        test_input_masks_1,
        test_segment_ids_1,
        test_labels,
    ) = convert_examples_to_features(
        tokenizer, test_examples_1, max_seq_length=max_seq_length
    )

    (
        test_input_ids_2,
        test_input_masks_2,
        test_segment_ids_2,
        test_labels,
    ) = convert_examples_to_features(
        tokenizer, test_examples_2, max_seq_length=max_seq_length
    )

    ## define the model structure
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
    embedded_sequences_1 = BertLayer(n_fine_tune_layers=3)(sequence_1_input)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
    embedded_sequences_2 = BertLayer(n_fine_tune_layers=3)(sequence_2_input)

    conv_layer1 = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu')
    conv_layer2 = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu')
    conv_layer3 = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu')

    con_11 = conv_layer1(embedded_sequences_1)
    con_11 = MaxPooling1D(4)(con_11)
    con_11 = Dropout(0.2)(con_11)
    con_11 = Dense(50)(con_11)

    con_12 = conv_layer2(embedded_sequences_1)
    con_12 = MaxPooling1D(4)(con_12)
    con_12 = Dropout(0.2)(con_12)
    con_12 = Dense(50)(con_12)

    con_13 = conv_layer3(embedded_sequences_1)
    con_13 = MaxPooling1D(4)(con_13)
    con_13 = Dropout(0.2)(con_13)
    con_13 = Dense(50)(con_13)

    con_21 = conv_layer1(embedded_sequences_2)
    con_21 = MaxPooling1D(4)(con_21)
    con_21 = Dropout(0.2)(con_21)
    con_21 = Dense(50)(con_21)

    con_22 = conv_layer2(embedded_sequences_2)
    con_22 = MaxPooling1D(4)(con_22)
    con_22 = Dropout(0.2)(con_22)
    con_22 = Dense(50)(con_22)

    con_23 = conv_layer3(embedded_sequences_2)
    con_23 = MaxPooling1D(4)(con_23)
    con_23 = Dropout(0.2)(con_23)
    con_23 = Dense(50)(con_23)

    merged1 = concatenate([con_11, con_12, con_13, con_21, con_22, con_23])
    merged1 = BatchNormalization()(merged1)
    merged1 = Dense(200)(merged1)
    merged1 = PReLU()(merged1)
    merged1 = Dropout(0.3)(merged1)

    con1 = conv_layer1(merged1)
    con1 = MaxPooling1D(4)(con1)
    con1 = Flatten()(con1)
    con1 = Dropout(0.2)(con1)
    con1 = Dense(50)(con1)

    con2 = conv_layer2(merged1)
    con2 = MaxPooling1D(4)(con2)
    con2 = Flatten()(con2)
    con2 = Dropout(0.2)(con2)
    con2 = Dense(50)(con2)

    con3 = conv_layer3(merged1)
    con3 = MaxPooling1D(4)(con3)
    con3 = Flatten()(con3)
    con3 = Dropout(0.2)(con3)
    con3 = Dense(50)(con3)

    merged = concatenate([con1, con2, con3])
    merged = BatchNormalization()(merged)

    merged = Dense(150)(merged)
    merged = PReLU()(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(90)(merged)
    merged = PReLU()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ## train the model
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    #sgd = SGD(lr=0.0, momentum=0.7, decay=0.0, nesterov=False)
    #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    # Instantiate variables
    #initialize_vars(sess)
    hist = model.fit([train_input_ids_1, train_input_masks_1, train_segment_ids_1],
                     [train_input_ids_2, train_input_masks_2, train_segment_ids_2], labels,
                 validation_data=([test_input_ids_1, test_input_masks_1, test_segment_ids_1],
                                  [test_input_ids_2, test_input_masks_2, test_segment_ids_2], labels_test),
                     epochs=epochs, batch_size=batch_size)
   ## make the submission
    scores = model.evaluate([test_input_ids_1, test_input_masks_1, test_segment_ids_1],
                                  [test_input_ids_2, test_input_masks_2, test_segment_ids_2], labels_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    print('Start making the submission before fine-tuning')
    preds = model.predict([test_input_ids_1, test_input_masks_1, test_segment_ids_1],
                                  [test_input_ids_2, test_input_masks_2, test_segment_ids_2], batch_size=batch_size, verbose=0)
    submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
    submission.to_csv('%.4f_'%(scores[1]*100)+'_BERT_CNN.csv', index=False)

    # Plot accuracy
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()
    #model.save('BERT_CNN.h5')

if __name__ == "__main__":
    main()
