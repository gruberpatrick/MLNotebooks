import matplotlib.pyplot as plt
import h5py
import pandas as pd
import numpy as np
import json

################################################################################
def plotRandom(train_images, train_data, lstm_in, nn_out, out_dict):
    
    """
    Plot a random element from the dataset.

    The function selects a random element from the given dataset and prints 
    all available information.

    Parameters
    ----------
    train_images : h5py
        image dataset
    train_data : pandas
        train dataset
    lstm_in : numpy.ndarray
        feature data used for input to NN
    nn_out : numpy.ndarray
        targets for the NN output
    out_dict : dict
        dictionary encoding the targets from index to word
        

    Returns
    -------
    int
        the index of the random element

    """

    idx = np.random.randint(0, len(train_images))
    
    print("=> ORIGINAL")
    print(train_data["index"].values[idx])
    print(train_data["question"].values[idx])
    print(train_data["encoding"].values[idx])
    print(train_data["answer"].values[idx])
    print(train_data["type"].values[idx])
    
    print("\n=> CONVERTED")
    print(lstm_in[idx])
    print(nn_out[idx])
    print(np.argmax(nn_out[idx]), out_dict[np.argmax(nn_out[idx])])
    plt.imshow(train_images[idx])
    plt.show()
    
    return idx

################################################################################
def loadData(file_name):
    
    """
    Load the data by a given file name. 

    Data comes in pairs of CSV and H5PY files. For the training of the
    model, we'll need both files per set.

    Parameters
    ----------
    file_name : string
        the name of the data pair to load
        

    Returns
    -------
    h5py array
        the h5py array to prevent loading all the images to RAM
    pandas dataframe
        the training data for each image (more information here: https://www.kaggle.com/gruberpatrick/sortofclevr)

    """
    
    # load the data;
    train_images = h5py.File("./"+file_name+".h5")[file_name]
    train_data = pd.read_csv("./"+file_name+".csv")

    # show the shapes;
    print(train_images.shape, train_data["index"].values.shape)
    
    return train_images, train_data

################################################################################
def loadQuestionEncoding(train_images, train_data):
    
    """
    Load JSON encoded catgorical question encoding.
    
    Since the categorical encoding is JSON encoded, we'll have to
    decode the information before we can use it.

    Parameters
    ----------
    file_name : string
        the name of the data pair to load
        

    Returns
    -------
    numpy.ndarray
        the simulated outputs of the LSTM
    numpy.ndarray
        each image is assigned whether it is used as a relational or non-relational example

    """
    
    print("Loading question encoding.")
    
    # load the encoding;
    lstm_in = []
    for it in range(len(train_data["encoding"].values)): 
    	lstm_in.append(json.loads(train_data["encoding"].values[it]))
        
    # convert to numpy array;
    lstm_in = np.array(lstm_in)
    
    return lstm_in

################################################################################
def getOutputs(train_data, lstm_in, nrout_tokenizer=None):
    
    """
    Encode the targets.

    We'll use a categorical representation to train the network. For this
    we need to take the current word answers, assign them indexes and
    encode the output. Previously we assigned the image a representation,
    which means that a question input is either relational or non-relational.
    Here we also load the correct relational or non-relational answer.

    Parameters
    ----------
    train_data : pandas
        train dataset
    lstm_in : numpy.ndarray
        feature data used for input to NN
    dist_encoding : numpy.ndarray
        the assignment of the dataset (relational or non-relational)
    nrout_tokenizer : dict
        the word encoding (given that we want to use the same encoding for validation and test)
        if not given, a new dict is created
        

    Returns
    -------
    numpy.ndarray
        the categorically encoded targets
    dict
        index to word dictionary
    dict
        word to index dictionary

    """
    
    nn_full = []
    for it in range(lstm_in.shape[0]): 
    	nn_full.append(train_data["answer"][it])
    
    # combine the data and tokenize the input;
    highest = 16
    if not nrout_tokenizer:
        
        word_index = {}
        index_word = {}
        idx = 0
        for it in range(len(nn_full)):
            if nn_full[it] not in word_index:
                word_index[nn_full[it]] = idx
                index_word[idx] = nn_full[it]
                idx += 1
                
    else:
        
        word_index = nrout_tokenizer
        index_word = {}
        for it in word_index:
            index_word[word_index[it]] = it
        
    nn_out = []
    for it in range(len(nn_full)):
        res = np.zeros((highest,))
        res[ word_index[nn_full[it]] ] = 1
        nn_out.append(res)

    nn_out = np.array(nn_out)
        
    return nn_out, index_word, word_index

################################################################################
train_images, train_data = loadData("data_train")
lstm_in = loadQuestionEncoding(train_images, train_data)
print(lstm_in.shape)
nn_out, train_dict, nrout_tokenizer = getOutputs(train_data, lstm_in)


for it in range(nn_out.shape[1]):
	print( it, np.sum(nn_out[:,it]) )

print(print( "type", np.sum(train_data["type"].values) ))

print(nn_out.shape)
print(train_dict)
plotRandom(train_images, train_data, lstm_in, nn_out, train_dict)