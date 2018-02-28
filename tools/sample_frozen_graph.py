import argparse
import os

from tensorflow.python.platform import gfile
from skimage.io import imsave
import numpy as np
import tensorflow as tf


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Script to generate a batch sample from a trained model")
    parser.add_argument("--modelDir", required=True, type=str, help="Model directory")
    parser.add_argument("--outputImagePath", required=False, type=str, default="./export/sample",
                        help="Generated images base file path")
    parser.add_argument("--tensorboardDir", required=False, type=str, default=None,
                        help="Tensorbpard dir to view the graph")
    args = parser.parse_args()
    return args


def main():

    args = doParsing()
    print(args)

    with gfile.GFile(os.path.join(args.modelDir, "graph.pb"), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess = tf.Session()
        tf.import_graph_def(graph_def, name="")

        # Optional save of tensorboard to see tensor names
        if args.tensorboardDir is not None:
            train_writer = tf.summary.FileWriter(args.tensorboardDir)
            train_writer.add_graph(sess.graph)

        # Generator output Tanh batch_size x h x w x 3 (e.g. 32 x 128 x 128 x 3)

        # No placeholder present, random uniform batch_size x 300 generate inputs for training

        outputTensor = sess.graph.get_tensor_by_name("Tanh" + ":0")

        generatedBatch = sess.run(outputTensor)

        # Save all batch images (from batch x h x w x 3 -> 1 x h x w x 3)
        splittedImages = np.split(generatedBatch, indices_or_sections=generatedBatch.shape[0], axis=0)
        if os.path.exists(os.path.dirname(args.outputImagePath)) is False:
            os.makedirs(os.path.dirname(args.outputImagePath))
        # Squeeze first dimension to have 3D numpy array with clip to -1 and 1 in case of strange predictions
        for index, image in enumerate(splittedImages):
            filePath = args.outputImagePath + "_" + str(index+1) + ".jpg"
            imsave(filePath, np.clip(np.squeeze(image, axis=0), a_min=-1.0, a_max=1.0))
            print("Saved sample in " + filePath)


if __name__ == "__main__":
    main()