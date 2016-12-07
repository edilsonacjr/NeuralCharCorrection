import numpy as np
import theano

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.utils.layer_utils import print_summary
# from keras.layers import Dense, Input, Flatten
# from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import *
from keras.models import Model, Sequential

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


MAX_SEQUENCE_LENGTH = 44
MAX_NB_WORDS = 44
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

NB_EPOCH = 20
BATCH_SIZE = 16
MAX_VOCAB_SIZE = 1000


def get_model():
    pass



def embed_vectorizer():
    pass


def main():


    vocab = ['teste', 'maria', 'casa'] # load texts
    # adicionar repetições baseadas em frequência


    print('Processing text dataset')
    with open('words-merged-lex.txt', 'r') as words_f:
        vocab = words_f.read().split()



    np.random.shuffle(vocab)
    vocab = vocab[:MAX_VOCAB_SIZE]

    le = LabelEncoder().fit(vocab)
    labels = le.transform(vocab)
    labels = to_categorical(np.array(labels))

    CLASS_OUT = len(labels)

    # char_index = dict(set(**))
    char_index = dict()
    for word in vocab:
        for char in word:
            if char not in char_index:
                char_index[char] = len(char_index)

    
    print(len(labels))
    print(labels.shape)
    #data = data



    # vectorizer = CountVectorizer(analyzer='char')
    # vectorizer.fit(vocab)

    # data = vectorizer.transform(vocab)
    # data = pad_sequences(data.toarray(), maxlen=MAX_SEQUENCE_LENGTH)

    # idx = np.where(data > 1)
    # data[idx] = 1

    print(len(vocab))
    MAX_WORD_SIZE = max(map(len, vocab))
    data = np.zeros((len(vocab), MAX_WORD_SIZE), dtype='int32')
    for i, word in enumerate(vocab):
        char_ids = list(map(char_index.__getitem__, word))
        data[i, :len(char_ids)] = char_ids

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    vocab = np.array(vocab)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    indices = indices[:MAX_VOCAB_SIZE] 
    # vocab = vocab[:MAX_VOCAB_SIZE]

    data = data[indices]
    labels = labels[indices]
    vocab = vocab[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    #x_train = data[:-nb_validation_samples]
    #y_train = labels[:-nb_validation_samples]
    #x_val = data[-nb_validation_samples:]
    #y_val = labels[-nb_validation_samples:]

    print('Preparing embedding matrix.')
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    
    


    print('Training model.')


    # [2, , , , , , , ]
    # [, 2, , , ,, , ,]

    sequence_input = Input(shape=(MAX_WORD_SIZE, ), dtype='int32')
    embedded_sequences = Embedding(len(char_index)+1, EMBEDDING_DIM)(sequence_input)


    # x = theano.tensor.imatrix()
    # print(theano.function(inputs=[x], outputs=x.shape)(embedded_sequences))
    # print(eval(embedded_sequences.shape))
    # print(embedded_sequences.output_shape)

    # r = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1, 1))(embedded_sequences)

    # lstml = ConvLSTM2D(nb_filter=512, nb_row=EMBEDDING_DIM, nb_col=1, 
    #     border_mode='valid', return_sequences=False)(r)


    # r2 = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(embedded_sequences)
    # print(lstml.output_shape)
    x = ZeroPadding1D(7//2)(embedded_sequences)
    x = Conv1D(256, 7, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    
    x = ZeroPadding1D(7//2)(x)
    x = Conv1D(128, 7, activation='relu')(x)
    x = MaxPooling1D(3)(x)

    #x = Conv1D(128, 4, activation='relu')(x)
    #x = MaxPooling1D(35)(x)
    
    encoded = Flatten()(x)
    x = Dense(128, activation='relu')(encoded)
    
    preds = Dense(CLASS_OUT, activation='softmax')(x)

    print('hasoidhaoiudhaoisudhasoiuudsahoiu')
    model = Model(sequence_input, preds)
    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', 'cosine_proximity'])

    model.fit(data, labels, nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE) #validation_data=(x_val, y_val)


    encoder = Model(input=sequence_input, output=encoded)
    X_encoded = encoder.predict(data)



    # new_data = []
    # new_data = vectorizer.transform(['abaixadi'])
    # new_data = pad_sequences(new_data.toarray(), maxlen=MAX_SEQUENCE_LENGTH)
    
    def process_new(word):
        z = np.zeros((1,MAX_WORD_SIZE))
        char_ids = list(map(char_index.__getitem__, word))
        z[0, :len(char_ids)] = char_ids
        return z

    new_data = process_new('abaixadi')
    # new_data = map(process_new, ['abaixadi'])
    X_test = encoder.predict(new_data)
    vec_abi = X_test[0].reshape(1, -1)

    vocab = vocab.tolist()

    # vec_aba = X_encoded[vocab.index('abaixada')].reshape(1, -1)
    # vec_abo = X_encoded[vocab.index('abaixado')].reshape(1, -1)
    # vec_abt = X_encoded[vocab.index('abacate')].reshape(1, -1)
    # vec_abj = X_encoded[vocab.index('abajur')].reshape(1, -1)
    

    # print(X_encoded.shape)
    # print(X_test.shape)
    # print(vec_abi.shape)
    # print(vec_abj.shape)


    # print('abaixado x abaixadi', cosine_similarity(vec_abo, vec_abi))
    # print('abaixada x abaixadi', cosine_similarity(vec_aba, vec_abi))
    # print('abajur x abaixadi', cosine_similarity(vec_abj, vec_abi))
    # print('abacate x abaixadi', cosine_similarity(vec_abt, vec_abi))

    sim = []
    for x in X_encoded:
        sim.append(cosine_similarity(x.reshape(1, -1), vec_abi)[0][0])

    maxs = np.argsort(sim)[-10:]
    scores = np.array(sim)[maxs]
    vets = np.array(vocab)[maxs]

    print('abaixadi\n========')
    for x, y in zip(vets, scores):
        print(x, y)
    

if __name__ == '__main__':
    main()