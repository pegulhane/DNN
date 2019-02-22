import os
import argparse
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.decomposition import PCA


def build_argparser():
    parser = argparse.ArgumentParser(description='Build Tensorboard projection')
    parser.add_argument('--vector', required=True, help='space separated vector')
    parser.add_argument('--labels', required=True, help='labels corresponding to vector file')
    parser.add_argument('--log_dir', required=False, default=None, help='log directory for TF models')
    return parser


if __name__ == '__main__':
    args = build_argparser().parse_args()

    log_dir = args.log_dir
    if args.log_dir is None:
        log_dir = os.path.join(os.getcwd(), "log_dir")

    # Load data
    data_frame = pd.read_csv(args.vector, index_col=False, header=None, sep=' ')
    metadata = args.labels

    # Generating PCA and
    pca = PCA(n_components=50, random_state=123, svd_solver='auto')
    df_pca = pd.DataFrame(pca.fit_transform(data_frame))
    df_pca = df_pca.values

    # Start tensorflow variable setup
    tf_data = tf.Variable(df_pca)

    # Start TF session
    with tf.Session() as sess:
        saver = tf.train.Saver([tf_data])
        sess.run(tf_data.initializer)
        saver.save(sess, os.path.join(log_dir, 'tf_data.ckpt'))
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = tf_data.name

        # Link this tensor to its metadata(Labels) file
        embedding.metadata_path = metadata

        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)
