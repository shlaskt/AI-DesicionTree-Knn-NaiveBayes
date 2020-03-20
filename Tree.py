class Tree:

    def __init__(self, attribute):

        self.nodes = {}
        self.attribute = attribute
        self.f = None

    def insert(self, subtree, came_from):
        self.nodes[came_from] = subtree

    # Print the tree
    def write_tree(self, file_name):
        self.f = open(file_name, "w+")
        self.print_nodes(self.attribute, self.nodes)
        self.f.close()

    def print_nodes(self, attribute, nodes, prefix=''):
        line = '|' if attribute != self.attribute else ''
        for key in sorted(nodes):
            subtree = nodes[key]
            if len(subtree.nodes) == 0:  # leaf
                self.f.write("{0}{1}={2}:{3}\n".format(prefix+line, attribute, key, subtree.attribute))
                # print("{0}{1}={2}:{3}".format(prefix+line, attribute, key, subtree.attribute))
            else:
                self.f.write("{0}{1}={2}\n".format(prefix+line, attribute, key))
                # print("{0}{1}={2}".format(prefix+line, attribute, key))
                self.print_nodes(subtree.attribute, subtree.nodes, '\t'+prefix)
