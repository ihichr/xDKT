#constants.py , you can execute all files in google collab , each file is one cell
Dpath = '/content/drive/xDKT/KTDataset' #you can adapt this path to your folder


datasets = {
    'assist2009' : 'assist2009',
    'assist2015' : 'assist2015',
    'assist2017' : 'assist2017',
    'static2011' : 'static2011',
    'kddcup2010' : 'kddcup2010',
    'synthetic' : 'synthetic'
}

# question number of each dataset
numbers = {
    'assist2009' : 124,
    'assist2015' : 100,
    'assist2017' : 102,
    'static2011' : 1224,
    'kddcup2010' : 661,
    'synthetic' : 50
}

DATASET = datasets['synthetic']
NUM_OF_QUESTIONS = numbers['synthetic']
# the max step of RNN model
MAX_STEP = 50
BATCH_SIZE = 64
LR = 0.0001
EPOCH = 100
#input dimension
INPUT = NUM_OF_QUESTIONS * 2
print(INPUT)
# embedding dimension
EMBED = NUM_OF_QUESTIONS
# hidden layer dimension
HIDDEN = 50
# nums of hidden layers
LAYERS = 1
# output dimension
OUTPUT = NUM_OF_QUESTIONS