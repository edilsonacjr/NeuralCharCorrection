import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model, Sequential

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


MAX_SEQUENCE_LENGTH = 44
MAX_NB_WORDS = 44
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

def main():


    vocab = ['teste', 'maria', 'casa'] # load texts
    # adicionar repetições baseadas em frequência


    print('Processing text dataset')
    with open('words-merged-lex.txt', 'r') as words_f:
        vocab = words_f.read().split()

    vocab = vocab[:10000]

    vectorizer = CountVectorizer(analyzer='char')
    vectorizer.fit(vocab)


    data = vectorizer.transform(vocab)
    data = pad_sequences(data.toarray(), maxlen=MAX_SEQUENCE_LENGTH)

    le = LabelEncoder().fit(vocab)
    labels = le.transform(vocab)

    CLASS_OUT = len(labels)

    word_index = vectorizer.vocabulary_

    labels = to_categorical(np.array(labels))
    #data = data

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    vocab = np.array(vocab)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
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
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH)


    print('Training model.')

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(256, 4, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 4, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    #x = Conv1D(128, 4, activation='relu')(x)
    #x = MaxPooling1D(35)(x)
    encoded = Flatten()(x)
    x = Dense(128, activation='relu')(encoded)
    preds = Dense(CLASS_OUT, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(data, labels, nb_epoch=50, batch_size=25) #validation_data=(x_val, y_val)


    encoder = Model(input=sequence_input, output=encoded)
    X_encoded = encoder.predict(data)



    new_data = []
    new_data = vectorizer.transform(['abaixadi'])
    new_data = pad_sequences(new_data.toarray(), maxlen=MAX_SEQUENCE_LENGTH)
    X_test = encoder.predict(new_data)

    vocab = vocab.tolist()

    vec_aba = X_encoded[vocab.index('abaixada')]
    vec_abo = X_encoded[vocab.index('abaixado')]
    vec_abt = X_encoded[vocab.index('abacate')]
    vec_abj = X_encoded[vocab.index('abajur')]
    vec_abi = X_test[0]

    from sklearn.metrics.pairwise import cosine_similarity

    print('abaixado x abaixadi', cosine_similarity(vec_abo, vec_abi))
    print('abaixada x abaixadi', cosine_similarity(vec_aba, vec_abi))
    print('abajur x abaixadi', cosine_similarity(vec_abj, vec_abi))
    print('abacate x abaixadi', cosine_similarity(vec_abt, vec_abi))

    sim = []
    for x in X_encoded:
        sim.append(cosine_similarity(x, vec_abi)[0][0])

    maxs = np.argsort(sim)[-10:]

    #print(maxs)
    print('abaixadi', np.array(vocab)[maxs])

    """
    sim = []
    for x in X_encoded:
        sim.append(cosine_similarity(x, vec_abo)[0][0])

    maxs = np.argsort(sim)[-10:]

    # print(maxs)
    print('abaixado', np.array(vocab)[maxs])
    """

if __name__ == '__main__':
    main()