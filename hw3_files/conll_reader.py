import sys 
from collections import defaultdict

# This class represents a singe word and its incoming dependency edge.
class DependencyEdge(object):
    """
    Represent a single dependency edge: 
    """

    def __init__(self, ident, word, pos, head, deprel): 
        self.id = ident #int - 1...
        self.word = word #string - the actual word
        self.pos = pos #POS - DT, NN, etc.
        self.head = head # int - word id of the parent word in the setnence - 0...
        self.deprel = deprel #int - dependency label on the edge pointing to this label
    
    def print_conll(self):
        return "{d.id}\t{d.word}\t_\t_\t{d.pos}\t_\t{d.head}\t{d.deprel}\t_\t_".format(d=self)
    

def parse_conll_relation(s):
    fields = s.split('\t')
    ident_s, word, lemma, upos, pos, feats, head_s, deprel, deps, misc = fields
    ident = int(ident_s)
    head = int(head_s)
    return DependencyEdge(ident, word, pos, head, deprel)

# This class represents a complete dependency tree.
class DependencyStructure(object):
    
    def __init__(self):
        self.deprels = {} # maps word ids to Dependency edge instances
        self.root = None # int - id of root node
        self.parent_to_children = defaultdict(list)
   
    # adds to the dictionary deprels{}
    def add_deprel(self, deprel):
        self.deprels[deprel.id] = deprel
        self.parent_to_children[deprel.head].append(deprel.id)
        if deprel.head == 0:
            self.root = deprel.id
    
    # prints the items in the dictionary deprels{}
    def __str__(self):
        for k,v in self.deprels.items():
            print(v)

    # prints the tree    
    def print_tree(self, parent = None):
        if not parent:
            return self.print_tree(parent = self.root)
       
        if self.deprels[parent].head == parent:
            return self.deprels[parent].word
    
        children = [self.print_tree(child) for child in self.parent_to_children[parent]]
        child_str = " ".join(children)
        return("({} {})".format(self.deprels[parent].word, child_str))
    
    def words(self):
        return [None]+[x.word for (i,x) in self.deprels.items()]
    
    def pos(self):
        return [None]+[x.pos for (i,x) in self.deprels.items()]
    
    def print_conll(self):
        deprels = [v for (k,v) in  sorted(self.deprels.items())]
        return "\n".join(deprel.print_conll() for deprel in deprels)


def conll_reader(input_file):
    current_deps = DependencyStructure() 
    while True:
        line = input_file.readline().strip()
        if not line and current_deps:
            yield current_deps
            current_deps = DependencyStructure() 
            line = input_file.readline().strip()
            if not line:  
                break
        current_deps.add_deprel(parse_conll_relation(line))



if __name__ == "__main__":
    with open(sys.argv[1],'r') as in_file:
        relations = set()
        for deps in conll_reader(in_file) :
            for deprel in deps.deprels.values(): 
                relations.add(deprel.deprel)       
            print(deps.print_tree()) 

