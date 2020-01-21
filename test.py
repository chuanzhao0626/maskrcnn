# coding:utf-8
import tensorflow as tf
from tensorflow.python.platform import gfile

tf.reset_default_graph()  # 重置计算图
output_graph_path = 'frozen.pb'
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    output_graph_def = tf.GraphDef()
    # 获得默认的图
    graph = tf.get_default_graph()
    with gfile.FastGFile(output_graph_path, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

        print("%d ops in the final graph." % len(output_graph_def.node))

        tensor_name = [tensor.name for tensor in output_graph_def.node]
        print(tensor_name)
        print('---------------------------')
        # summaryWriter = tf.summary.FileWriter('log_graph/', graph)

        for op in graph.get_operations():
            # print出tensor的name和值
            print(op.name, op.values())


