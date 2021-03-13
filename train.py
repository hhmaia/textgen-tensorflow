import sys
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from dataset import create_datasets
from parsecorpus import json_to_tape 
from mlutils import create_lr_sched, export_embeddings, export_vocabulary 


def train(dataset_path,
        model_path,
        tokenizer_path,
        logs_path,
        warmup=False,
        seq_len=32, 
        batch_size=128,
        epochs=20,
        train_split = 0.8,
        val_split = 0.2):
    
    with open(tokenizer_path, 'r') as f:
        tokenizer = tokenizer_from_json(f.read())

    tape = json_to_tape(dataset_path)
    indexes_tape = tokenizer.texts_to_sequences([tape])[0]
    train_nbatches = int((len(indexes_tape)-seq_len) * train_split / batch_size)
    val_nbatches = int((len(indexes_tape)-seq_len) * val_split / batch_size)
    
    train_ds, val_ds = create_datasets(
        indexes_tape,
        train_nbatches, 
        val_nbatches,
        batch_size, 
        seq_len,
        50000) 

    model = tf.keras.models.load_model(model_path)
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
    model.summary()

    ckp_cb = tf.keras.callbacks.ModelCheckpoint(
        model_path, 'val_accuracy', save_best_only=True)

    lr_cb = tf.keras.callbacks.LearningRateScheduler(
        create_lr_sched(epochs/2, epochs, warmup=warmup), True)

    tb_cb = tf.keras.callbacks.TensorBoard(
            logs_path, 1, True, embeddings_freq=1)  

    hist = model.fit(
        train_ds, 
        batch_size=batch_size, 
        epochs=epochs,
        steps_per_epoch=train_nbatches,
        validation_data=val_ds, 
        callbacks=[ckp_cb, lr_cb, tb_cb])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for training the model.')
    parser.add_argument('-d', '--dataset', type=str, required=True,
            help='Path to the dataset to train on.')
    parser.add_argument('-m', '--model', type=str, required=True, 
            help='Path to the directory where to find the model to be loaded.')
    parser.add_argument('-t', '--tokenizer', type=str, required=True, 
            help='Path to the tokenizer JSON file.')
    parser.add_argument('-l', '--logs', type=str, required=True, 
            help='Path to the directory where to save Tensorboad logs.')
    parser.add_argument('-e', '--epochs', type=int, required=False, default='1',
            help='Number of epochs to train.')
    parser.add_argument('-b', '--batch_size', type=int, required=False, default='128',
            help='Batch size to use when creating the dataset')
    parser.add_argument('-w', '--warmup', action='store_true', 
            help='Wheter to gradualy increase learning rate at the start of the training.')
    args = parser.parse_args()

    train(args.dataset,
            args.model,
            args.tokenizer,
            args.logs,
            args.warmup,
            batch_size=args.batch_size, 
            epochs=args.epochs)

