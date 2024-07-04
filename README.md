# neural-network-nlp

This project was completed for the course Natural Language Processing in Columbia University.

Project goal: 
- train a feed-forward neural network to predict the transitions of an arc-standard dependency parser.
- Note: The input to this network will be a representation of the current state (including words on the stack and buffer). The output will be a transition (shift, left_arc, right_arc), together with a dependency relation label.

Code I worked on can be found in the following files:
- `extract_training_data.py`
- `train_model.py`
- `decoder.py`
