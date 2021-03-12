import tensorflow as tf

def create_datasets(tape,
                    train_n_batches,
                    val_n_batches,
                    batch_size,
                    seq_len,
                    shuffle_buffer_size=10000,
                    seed=1):

    dataset = tf.data.Dataset.from_tensor_slices(tape)
    dataset = dataset.window(seq_len, 1, drop_remainder=True)
    dataset = dataset.flat_map(lambda x: x.batch(seq_len))
    dataset = dataset.map(lambda x: (x[:-1], x[-1]))
    dataset = dataset.shuffle(shuffle_buffer_size, seed)
    dataset = dataset.batch(batch_size)
    train_dataset = dataset.take(train_n_batches)
    dataset = dataset.skip(train_n_batches)
    val_dataset = dataset.take(val_n_batches)
    train_dataset = train_dataset.repeat()
    
    return train_dataset, val_dataset
