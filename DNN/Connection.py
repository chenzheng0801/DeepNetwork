import numpy as np

class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.gradient = 0.0
        self.weight = np.random.uniform(-1.0,1.0)

    def calc_gradient(self):
        u_node = self.upstream_node
        d_node = self.downstream_node
        self.gradient = u_node.get_output() * d_node.get_delta()

    def update_weight(self, rate):
        self.weight = self.weight + rate * self.gradient

    def show_info(self):
        info_str = "\n\tup node (layer %u, index %u) \n\tdown node (layer %u, index %u)  weight%f" % ( \
                    self.upstream_node.layer_index, self.upstream_node.node_index,\
                    self.downstream_node.layer_index, self.downstream_node.layer_index, self.weight)
        print(info_str)