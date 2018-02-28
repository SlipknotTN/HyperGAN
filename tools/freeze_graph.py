import argparse
import os

import tensorflow as tf
from custom_freeze_graph_tf import freeze_graph


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Script to generate a batch sample from a trained model")
    parser.add_argument("--modelDir", required=True, type=str, help="Checkpoint directory")
    parser.add_argument("--checkpointStep", required=False, type=int, default=None, help="Checkpoint step to load")
    parser.add_argument("--frozenModelDir", required=False, type=str, default="./export",
                        help="Frozen graph file")
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

        # Save metagraph
        tf.train.write_graph(sess.graph.as_graph_def(), "", args.frozenModelDir + "/model_graph.pb", False)
        print("Metagraph saved")

    # Freeze graph (graphdef plus parameters),
    # this includes in the graph only the layers needed to provide the output_node_names
    print("Freezing graph...")
    freeze_graph(input_graph=args.frozenModelDir + "/model_graph.pb", input_saver="", input_binary=True,
                 input_checkpoint=checkpointPath, output_node_names="Tanh",
                 restore_op_name="save/restore_all", filename_tensor_name="save/Const:0",
                 output_graph=args.frozenModelDir + "/graph.pb", clear_devices=True, initializer_nodes="")


if __name__ == "__main__":
    main()