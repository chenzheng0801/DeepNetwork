import numpy as np
from functools import reduce

class Node(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0.0
        self.delta = 0.0

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        self.upstream.append(conn)

    def set_ouput(self, value):
        self.output = value

    def get_output(self):
        return self.output

    def get_delta(self):
        return self.delta

    def calc_output(self):
        node_input = reduce(lambda ret,conn: ret + conn.weight*conn.start_node.output, self.upstream, 0)
        self.output = self.__sigmoid()(node_input)

    def __sigmoid(self, weight_input):
        return 1.0/(1.0 + np.exp(- weight_input))

    def calc_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def calc_hidden_layer_delta(self):
        self.delta = reduce(lambda ret, conn: ret + conn.weight* conn.end_node.delta, self.downstream, 0.0)
        self.delta = self.delta * self.output * (1 - self.output)

    def show_info(self):
        node_str = 'layer %u, %u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + "\n\t" + str(conn), self.downstream, "")
        upstream_str = reduce(lambda ret, conn: ret + "\n\t" + str(conn), self.upstream, "")
        print(node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str)


class ConstNode(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1.0

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def cal_hidden_layer_delta(self):
        delta =0

    def calc_output(self):
        self.output = 1.0

    def show_info(self):
        node_str = "layer %u, node %u: output: 1" % (self.layer_index, self.node_index)
        downstream_str = reduce( lambda ret, conn: ret + "\n\t" + str(conn), self.downstream_str, "")
        print(node_str + '\n\tdownstream:' + downstream_str)