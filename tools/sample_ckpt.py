import argparse
import os

from skimage.io import imsave
import numpy as np
import tensorflow as tf


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Script to generate a batch sample from a trained model")
    parser.add_argument("--modelDir", required=True, type=str, help="Checkpoint directory")
    parser.add_argument("--checkpointStep", required=False, type=int, default=None, help="Checkpoint step to load")
    parser.add_argument("--outputImagePath", required=False, type=str, default="./export/sample",
                        help="Generated images base file path")
    parser.add_argument("--tensorboardDir", required=False, type=str, default=None,
                        help="Tensorbpard dir to view the graph")
    args = parser.parse_args()
    return args


def getModelPaths(modelDir, checkpointStep):
    if checkpointStep is None:
        return os.path.join(modelDir, "model.ckpt.meta"), os.path.join(modelDir, "model.ckpt")
    else:
        return os.path.join(modelDir, "model.ckpt-" + str(checkpointStep) + ".meta"),\
               os.path.join(modelDir, "model.ckpt-" + str(checkpointStep))


def main():

    args = doParsing()
    print(args)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

        metagraphPath, checkpointPath = getModelPaths(args.modelDir, args.checkpointStep)

        print("Loading metagraph")
        saver = tf.train.import_meta_graph(metagraphPath)
        print("Restoring model")
        restored = saver.restore(sess, checkpointPath)
        print("Checkpoint loaded")

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
        # Squeeze first dimension to have 3D numpy array
        for index, image in enumerate(splittedImages):
            filePath = args.outputImagePath + "_" + str(index+1) + ".jpg"
            imsave(filePath, np.clip(np.squeeze(image, axis=0), a_min=-1.0, a_max=1.0))
            print("Saved sample in " + filePath)


if __name__ == "__main__":
    main()