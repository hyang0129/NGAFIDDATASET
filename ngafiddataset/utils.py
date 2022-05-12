import os
import tensorflow as tf
import subprocess

def connect_to_tpu():

    # assert (
    #     "COLAB_TPU_ADDR" in os.environ
    # ), "ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!"
    # TPU_ADDRESS = "grpc://" + os.environ["COLAB_TPU_ADDR"]
    # print("TPU address is", TPU_ADDRESS)

    # Detect hardware, return appropriate distribution strategy
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU ", tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)

    return strategy

def shell_exec(command_as_string):
    """
    Executes the command in shell script. Avoid using the yes | command pattern as that seems to cause an out of memory
    issue.

    Args:
        command_as_string: just a string, as if you were typing it into a  terminal

    Returns:
        stdout, stderr
    """
    try:

        process = subprocess.Popen(
            command_as_string.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False
        )
        stdout, stderr = process.communicate()

    except subprocess.CalledProcessError as e:
        stdout = e.stdout
        stderr = e.stderr

    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    return stdout, stderr