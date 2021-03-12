import os
import argparse
import tensorflow as tf
import tensorflow.keras.layers as klayers


def export_model(path, input_seq_len, vocab_size, emb_dim):
   
    model = tf.keras.Sequential([
        klayers.Embedding(vocab_size+1, emb_dim, input_length=input_seq_len),
        klayers.Bidirectional(klayers.LSTM(256)),
        klayers.BatchNormalization(),
        #klayers.Flatten(),
        klayers.Dense(vocab_size, 'softmax')
    ])

    model.summary()
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
    model.save(path)
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create and export the model')
    parser.add_argument('-p', '--path', type=str, required=True,
        help='dir where to save the model')
    parser.add_argument('-s', '--seqlen', type=int, required=True,
        help='input sequence length')
    parser.add_argument('-v', '--vocabsize', type=int, required=True,
        help='number of output logits, also the number of embeddings in the input layer')
    parser.add_argument('-e', '--embeddingsdim', type=int, required=True,
        help='dimension of the embeddings in the input layer')

    args = parser.parse_args()
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    export_model(args.path, args.seqlen, args.vocabsize, args.embeddingsdim) 

