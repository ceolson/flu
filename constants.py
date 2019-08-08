import numpy as np

# Convert amino acids to one-hot vectors
CATEGORIES = {
'A': [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
'R': [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'N': [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'D': [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'C': [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'Q': [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'E': [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'G': [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'H': [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'I': [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
'L': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
'K': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
'M': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
'F': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
'P': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], 
'S': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
'T': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], 
'W': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], 
'Y': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], 
'V': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
'X': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
'-': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
}

# Convert amino acids to log distributions over all amino acids
# https://www.ncbi.nlm.nih.gov/Class/FieldGuide/BLOSUM62.txt
BLOSUM = {
'A':   [ 4.,   -1.,   -2.,   -2.,    0.,   -1.,   -1.,    0.,   -2.,   -1.,   -1.,   -1.,   -1.,   -2.,   -1.,    1.,    0.,   -3.,   -2.,    0.,   -2.,   -1.,    0.,   -4],
'R':   [-1.,    5.,    0.,   -2.,   -3.,    1.,    0.,   -2.,    0.,   -3.,   -2.,    2.,   -1.,   -3.,   -2.,   -1.,   -1.,   -3.,   -2.,   -3.,   -1.,    0.,   -1.,   -4],
'N':   [-2.,    0.,    6.,    1.,   -3.,    0.,    0.,    0.,    1.,   -3.,   -3.,    0.,   -2.,   -3.,   -2.,    1.,    0.,   -4.,   -2.,   -3.,    3.,    0.,   -1.,   -4],
'D':   [-2.,   -2.,    1.,    6.,   -3.,    0.,    2.,   -1.,   -1.,   -3.,   -4.,   -1.,   -3.,   -3.,   -1.,    0.,   -1.,   -4.,   -3.,   -3.,    4.,    1.,   -1.,   -4],
'C':   [ 0.,   -3.,   -3.,   -3.,    9.,   -3.,   -4.,   -3.,   -3.,   -1.,   -1.,   -3.,   -1.,   -2.,   -3.,   -1.,   -1.,   -2.,   -2.,   -1.,   -3.,   -3.,   -2.,   -4],
'Q':   [-1.,    1.,    0.,    0.,   -3.,    5.,    2.,   -2.,    0.,   -3.,   -2.,    1.,    0.,   -3.,   -1.,    0.,   -1.,   -2.,   -1.,   -2.,    0.,    3.,   -1.,   -4],
'E':   [-1.,    0.,    0.,    2.,   -4.,    2.,    5.,   -2.,    0.,   -3.,   -3.,    1.,   -2.,   -3.,   -1.,    0.,   -1.,   -3.,   -2.,   -2.,    1.,    4.,   -1.,   -4],
'G':   [ 0.,   -2.,    0.,   -1.,   -3.,   -2.,   -2.,    6.,   -2.,   -4.,   -4.,   -2.,   -3.,   -3.,   -2.,    0.,   -2.,   -2.,   -3.,   -3.,   -1.,   -2.,   -1.,   -4],
'H':   [-2.,    0.,    1.,   -1.,   -3.,    0.,    0.,   -2.,    8.,   -3.,   -3.,   -1.,   -2.,   -1.,   -2.,   -1.,   -2.,   -2.,    2.,   -3.,    0.,    0.,   -1.,   -4],
'I':   [-1.,   -3.,   -3.,   -3.,   -1.,   -3.,   -3.,   -4.,   -3.,    4.,    2.,   -3.,    1.,    0.,   -3.,   -2.,   -1.,   -3.,   -1.,    3.,   -3.,   -3.,   -1.,   -4],
'L':   [-1.,   -2.,   -3.,   -4.,   -1.,   -2.,   -3.,   -4.,   -3.,    2.,    4.,   -2.,    2.,    0.,   -3.,   -2.,   -1.,   -2.,   -1.,    1.,   -4.,   -3.,   -1.,   -4],
'K':   [-1.,    2.,    0.,   -1.,   -3.,    1.,    1.,   -2.,   -1.,   -3.,   -2.,    5.,   -1.,   -3.,   -1.,    0.,   -1.,   -3.,   -2.,   -2.,    0.,    1.,   -1.,   -4],
'M':   [-1.,   -1.,   -2.,   -3.,   -1.,    0.,   -2.,   -3.,   -2.,    1.,    2.,   -1.,    5.,    0.,   -2.,   -1.,   -1.,   -1.,   -1.,    1.,   -3.,   -1.,   -1.,   -4],
'F':   [-2.,   -3.,   -3.,   -3.,   -2.,   -3.,   -3.,   -3.,   -1.,    0.,    0.,   -3.,    0.,    6.,   -4.,   -2.,   -2.,    1.,    3.,   -1.,   -3.,   -3.,   -1.,   -4],
'P':   [-1.,   -2.,   -2.,   -1.,   -3.,   -1.,   -1.,   -2.,   -2.,   -3.,   -3.,   -1.,   -2.,   -4.,    7.,   -1.,   -1.,   -4.,   -3.,   -2.,   -2.,   -1.,   -2.,   -4],
'S':   [ 1.,   -1.,    1.,    0.,   -1.,    0.,    0.,    0.,   -1.,   -2.,   -2.,    0.,   -1.,   -2.,   -1.,    4.,    1.,   -3.,   -2.,   -2.,    0.,    0.,    0.,   -4],
'T':   [ 0.,   -1.,    0.,   -1.,   -1.,   -1.,   -1.,   -2.,   -2.,   -1.,   -1.,   -1.,   -1.,   -2.,   -1.,    1.,    5.,   -2.,   -2.,    0.,   -1.,   -1.,    0.,   -4],
'W':   [-3.,   -3.,   -4.,   -4.,   -2.,   -2.,   -3.,   -2.,   -2.,   -3.,   -2.,   -3.,   -1.,    1.,   -4.,   -3.,   -2.,   11.,    2.,   -3.,   -4.,   -3.,   -2.,   -4],
'Y':   [-2.,   -2.,   -2.,   -3.,   -2.,   -1.,   -2.,   -3.,    2.,   -1.,   -1.,   -2.,   -1.,    3.,   -3.,   -2.,   -2.,    2.,    7.,   -1.,   -3.,   -2.,   -1.,   -4],
'V':   [ 0.,   -3.,   -3.,   -3.,   -1.,   -2.,   -2.,   -3.,   -3.,    3.,    1.,   -2.,    1.,   -1.,   -2.,   -2.,    0.,   -3.,   -1.,    4.,   -3.,   -2.,   -1.,   -4],
'B':   [-2.,   -1.,    3.,    4.,   -3.,    0.,    1.,   -1.,    0.,   -3.,   -4.,    0.,   -3.,   -3.,   -2.,    0.,   -1.,   -4.,   -3.,   -3.,    4.,    1.,   -1.,   -4],
'Z':   [-1.,    0.,    0.,    1.,   -3.,    3.,    4.,   -2.,    0.,   -3.,   -3.,    1.,   -1.,   -3.,   -1.,    0.,   -1.,   -3.,   -2.,   -2.,    1.,    4.,   -1.,   -4],
'X':   [ 0.,   -1.,   -1.,   -1.,   -2.,   -1.,   -1.,   -1.,   -1.,   -1.,   -1.,   -1.,   -1.,   -1.,   -2.,    0.,    0.,   -2.,   -1.,   -1.,   -1.,   -1.,   -1.,   -4],
'-':   [-4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,   -4.,    1]
}

# Convert subtypes to one-hot vectors
TYPES = {
1:  [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
2:  [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
3:  [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
4:  [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
5:  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
6:  [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
7:  [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
8:  [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
9:  [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
10: [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
11: [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
12: [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
13: [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
14: [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
15: [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
16: [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
17: [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
18: [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
}

# End of message and start of message characters (used by RNN)
EOM_VECTOR = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])
SOM_VECTOR = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])

# Head/stem domains
HEAD = [i for i in range(132,277)]
STEM = [i for i in range(132)] + [i for i in range(277,576)]

# Convert from vectors back to character representations of amino acids
ORDER_CATEGORICAL = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','X','-','<EOM>','<SOM>']
ORDER_BLOSUM = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','A','Q','X','-','<EOM>','<SOM>']

# Convert a matrix of amino acids encoded as vectors to a sequence string
# Doesn't matter if BLOSUM-style or one-hot vectors because just cares about argmax
def convert_to_string(prediction,ORDER):
    string = ''
    for i in range(len(prediction)):
        prediction[i,-3] = 0.      # Don't output "unknown"
        # ~ # OPTION to do this probabilistically
        # ~ scaling_factor = np.sum(prediction[i])
        # ~ probs = np.divide(prediction[i],scaling_factor)
        # ~ index = np.random.choice(len(prediction[i]),1,p=probs)
        index = np.argmax(prediction[i])
        residue = ORDER[int(index)]
        string += residue
    return string
