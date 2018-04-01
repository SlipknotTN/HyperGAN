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
    """
    Run the script from repository root directory
    """
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

        # Placeholder and output retrieving
        inputTensor = sess.graph.get_tensor_by_name("input:0")
        outputTensor = sess.graph.get_tensor_by_name("Tanh" + ":0")

        # Forced all zeros
        #inputValues = np.zeros(shape=inputTensor.shape, dtype=np.float32)
        # Random from -1.0 to 1.0
        inputValues = np.random.random_sample(inputTensor.shape) * 2.0 - 1.0
        generatedBatch = sess.run(outputTensor, feed_dict={inputTensor: inputValues})

        # Save all batch images (from batch x h x w x 3 -> 1 x h x w x 3)
        splittedImages = np.split(generatedBatch, indices_or_sections=generatedBatch.shape[0], axis=0)
        if os.path.exists(os.path.dirname(args.outputImagePath)) is False:
            os.makedirs(os.path.dirname(args.outputImagePath))
        # Squeeze first dimension to have 3D numpy array with clip to -1 and 1 in case of strange predictions
        for index, image in enumerate(splittedImages):
            filePath = args.outputImagePath + "_" + str(index+1) + ".jpg"
            image = np.clip(np.squeeze(image, axis=0), a_min=-1.0, a_max=1.0)
            # No normalization, only scaling to [0, 255]
            image += 1.0
            image *= 255.0/2.0
            image = image.astype(np.uint8)
            imsave(filePath, image)
            print("Saved sample in " + filePath)


if __name__ == "__main__":
    main()