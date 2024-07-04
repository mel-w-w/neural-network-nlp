from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
          
            # TODO: Write the body of this loop for part 4 

            features = self.extractor.get_input_representation(words, pos, state)
            y = self.model.predict(features.reshape(1,6,))

            # create a list of possible actions(shift, right-arc, left-arc)
            possibilities = []
            for i in range(len(y[0])-1):
              possibilities.append((self.output_labels[i], y[0][i]))
            possibilities.sort(key = lambda x: x[1], reverse=True)

            # go through the possibilities and find one that should be executed
            for i in range(len(possibilities)): 
              ((best, label), prob) = possibilities[0]
              possibilities.pop(0)

              if best == "shift":
                if len(state.buffer) != 0:
                  state.shift()
                  break
              elif best == "left_arc":
                if len(state.stack) != 0 and state.stack[-1] != 0:
                  state.right_arc(label)
                  break
              elif best == "right_arc":
                if len(state.stack) != 0:
                  state.left_arc(label)
                  break
              else:
                print("No legal transition.")

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
