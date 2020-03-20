from Node import Node
from Node import ConstNode

class Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        self.node_count = node_count
        # 得先初始化完所有节点后才好连接
        for i in range(node_count-1):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count-1))

    def set_output(self, data):
        for i in range(len(data)):
            self.nodes[i].set_ouput(data[i])

    def cal_output(self):
        for node in self.nodes:
            node.calc_output()


    def show_info(self):
        for node in self.nodes:
            node.show_info()