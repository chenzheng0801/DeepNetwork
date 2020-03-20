from Layer import Layer
from Connection import Connection

class Network(object):
    def __init__(self, layers):
        self.connections = []
        self.layers = []
        for i in range(len(layers)):
            self.layers.append(Layer(i,layers[i]))
        for i in range(len(self.layers)-1):
            Conns = [ Connection(upstream_node, downstream_node)
                     for upstream_node in self.layers[i].nodes
                     for downstream_node in self.layers[i+1].nodes]
            for conn in Conns:
                self.connections.append(conn)
                conn.downstream_node.append_downstream_connection(conn)
                conn.upstream_node.appen_downstream_connection(conn)

    def train(self, labels, data_set, rate, iteration):
        for i in range(len(iteration)):
            for d in range(len(data_set)):
                self.train_one_example(labels[d], data_set[d], rate)

    def train_one_example(self, label, sample, rate):
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self,label):
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.cal_hidden_layer_delta()

    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

