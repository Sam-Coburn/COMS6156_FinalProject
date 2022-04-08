import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

#--------------------------------------------------------------------------------------------------------------------------------
#   Function:   read_code_embeddings	
#	Inputs:     file_name: the path to the file containing the code2vec embeddings stored in a text file
#                   * the function assumes that the provided .txt file stores the code2vec embeddings in an alphabetic manner
#                     with respect to the source code file names
#	Outputs:    a dictionary of code2vec files with the file name as the key and code embedding and file tag as the value
#	Purpose:	The function reads the code2vec embeddings from the provided .txt files and creates a dictionary of them
#--------------------------------------------------------------------------------------------------------------------------------
def read_code_embeddings(file_name):
    # Open the code2vec embedding file in read mode
    file = open(file_name, "r")
    lines = file.readlines() # Retrieve all lines from the file in a list

    # Create a dictionary of code2vec embeddings
    embedding_dictionary = {}
    count = 1
    for i in range(0, len(lines)):
        if count == 11:
            count = 1

        l = lines[i].strip()

        # Delineates between source code files based on their count
        # * Assumes embeddings are in alphabetical order for proper organization within the dictionary
        if i >= 0 and i <= 9:
            embedding_dictionary["BubbleSort" + str(count)] = (l.split(" "), 1)
        elif i >= 10 and i <= 19:
            embedding_dictionary["EightQueens" + str(count)] = (l.split(" "), 2)
        elif i >= 20 and i <= 29:
            embedding_dictionary["GuessWord" + str(count)] = (l.split(" "), 3)
        elif i >= 30 and i <= 39:
            embedding_dictionary["Hanoi" + str(count)] = (l.split(" "), 4)
        elif i >= 40 and i <= 49:
            embedding_dictionary["InfixConverter" + str(count)] = (l.split(" "), 5)
        elif i >= 50 and i <= 59:
            embedding_dictionary["Kaprekar" + str(count)] = (l.split(" "), 6)
        elif i >= 60 and i <= 69:
            embedding_dictionary["MagicSquare" + str(count)] = (l.split(" "), 7)
        elif i >= 70 and i <= 79:
            embedding_dictionary["RailRoadCar" + str(count)] = (l.split(" "), 8)
        elif i >= 80 and i <= 89:
            embedding_dictionary["SLinkedList" + str(count)] = (l.split(" "), 9)
        elif i >= 90 and i <= 99:
            embedding_dictionary["SqrtAlgorithm" + str(count)] = (l.split(" "), 10)

        count += 1

    return embedding_dictionary

#--------------------------------------------------------------------------------------------------------------------------------
#   Function:   cosine_similarity	
#	Inputs:     e1: code2vec embedding #1, stored as a list of strings
#               e2: code2vec embedding #2, stored as a list of strings
#	Outputs:    cosine similarity value computed using the two embeddings
#	Purpose:	Performs the cosine similarity calculation for the two given code2vec embeddings
#                   * cosine_similarity = dot_product(e1, e2) / (|e1| * |e2|)
#--------------------------------------------------------------------------------------------------------------------------------
def cosine_similarity(e1, e2):
    # Calculate the dot product of the two embeddings
    dot_product = 0
    for i in range(0, len(e1)):
        dot_product += (float(e1[i]) * float(e2[i]))

    # Calculate the magnitude of embedding 1
    e1_magnitude = 0
    for e in e1:
        e1_magnitude += (float(e)**2)
    e1_magnitude = sqrt(e1_magnitude)

    # Calcuate the magnitude of embedding 2
    e2_magnitude = 0
    for e in e2:
        e2_magnitude += (float(e)**2)
    e2_magnitude = sqrt(e2_magnitude)

    return (dot_product) / (e1_magnitude * e2_magnitude) # Return cosine similarity value

#--------------------------------------------------------------------------------------------------------------------------------
#   Function:   threshold_grid_search	
#	Inputs:     embeddings: code2vec embedding dictionary created by calling the read_code_embeddings function
#	Outputs:    the threshold value which provides the highest observed F1 score for the given code2vec embeddings
#	Purpose:	The function calculates the F1 score, gauging the performance when determining the similarity between the given 
#               code embeddings, for varying threshold values (between 1 and 100) in order to find the highest F1 score 
#               achievable by this method of similarity comparison.
#--------------------------------------------------------------------------------------------------------------------------------
def threshold_grid_search(embeddings):
    f1_scores = []

    # Instantiation of the X and Y value lists used in the ROC Curve plot
    x = [] # False Positive Rate
    y = [] # True Positive Rate

    # Cycles through threshold values in order to find highest F1 score produced by this similarity comparison method
    for i in range(1, 101):
        threshold = i / 100

        # TP, TN, FP, and FN counts
        # TPs are defined as codes with the same tag within the embedding dictionary with a similarity score above the current threshold
        true_positive_count = 0
        true_negative_count = 0
        false_positive_count = 0
        false_negative_count = 0

        # Compare all pairwise comparisons of the code2vec embeddings by calculating their cosine similarity
        # Increment the TP, TN, FP, and FN counts appropriately
        for k1 in list(embeddings.keys()):
            for k2 in list(embeddings.keys()):
                val = cosine_similarity(embeddings[k1][0], embeddings[k2][0])
                if val >= threshold:
                    if embeddings[k1][1] == embeddings[k2][1]:
                        true_positive_count += 1
                    else:
                        false_positive_count += 1
                else:
                    if embeddings[k1][1] == embeddings[k2][1]:
                        false_negative_count += 1
                    else:
                        true_negative_count += 1
        
        # Calculate analytic metrics from the TP, TN, FP, and FN counts
        precision = true_positive_count / (true_positive_count + false_positive_count)
        recall = true_positive_count / (true_positive_count + false_negative_count)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
        
        x.append(false_positive_count / (false_positive_count + true_negative_count))
        y.append(true_positive_count / (true_positive_count + false_negative_count))

    # Retrieve max F1 found from the grid search
    print("MAX F1: ", max(f1_scores))
    print("MAX F1 T: ", (f1_scores.index(max(f1_scores)) + 1))

    # Calculate the AUC of the ROC Curve
    x_arr = np.array(x)
    y_arr = np.array(y)
    auc = np.trapz(y_arr, x = x_arr)
    print("AUC: ", auc) # Fix by sorting x (and correspondindly Y)

    # Plot ROC Curve
    '''
    plt.plot(x, y)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characterisitc (ROC) Curve")
    plt.show()
    '''

    return (f1_scores.index(max(f1_scores)) + 1) / 100 # Return the threshold value with the max observed F1 score

#--------------------------------------------------------------------------------------------------------------------------------
#   Function:   code_similarity_analysis	
#	Inputs:     embeddings: code2vec embedding dictionary of the current experiment
#               test_name: name of the current experiment (used in printing of results)
#               threshold: maximum threshold value for the given experiment
#	Outputs:    N/A
#	Purpose:	Prints the analytics values for the current experiment's code2vec embeddings at the max F1 threshold value
#                   * the code repeats the comparisons made by the threshold_grid_search function, except this function does it
#                     for one threshold value (specifically, the max threshold value) rather than all possible ones.
#--------------------------------------------------------------------------------------------------------------------------------
def code_similarity_analysis(embeddings, test_name, threshold):
    true_positive_count = 0
    true_negative_count = 0
    false_positive_count = 0
    false_negative_count = 0
    
    for k1 in list(embeddings.keys()):
        for k2 in list(embeddings.keys()):
            val = cosine_similarity(embeddings[k1][0], embeddings[k2][0])
            if val >= threshold:
                if embeddings[k1][1] == embeddings[k2][1]:
                    true_positive_count += 1
                else:
                    false_positive_count += 1
            else:
                if embeddings[k1][1] == embeddings[k2][1]:
                    false_negative_count += 1
                else:
                    true_negative_count += 1
    
    print("EMBEDDINGS " + test_name + " ANALYTICS")
    print("true positives: ", true_positive_count)
    print("true negatives: ", true_negative_count)
    print("false positives: ", false_positive_count)
    print("false negatives: ", false_negative_count)

    accuracy = (true_positive_count + true_negative_count) / ((len(k1) * len(k2)) * 100)
    precision = true_positive_count / (true_positive_count + false_positive_count)
    recall = true_positive_count / (true_positive_count + false_negative_count)
    f1 = (2 * precision * recall) / (precision + recall)
    
    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", f1)
    print("\n")

# Tests for Code Embeddings
# * separated into two experiments: embeddings trained using one model (model A) and embeddings using another, larger model (model B)
# * for each model, there are three sets of embeddings: the normal dataset, and two datasets for the original files passed through decompilers
if __name__ == "__main__":
    # EMBEDDINGS A1
    embeddings_A1 = read_code_embeddings("Code Embeddings/code_embeddings_A1.txt")
    A1_threshold = threshold_grid_search(embeddings_A1) # A1 Threshold is 0.66 (threshold with max F1 value)
    code_similarity_analysis(embeddings_A1, "A1", A1_threshold)

    # EMBEDDINGS A2
    embeddings_A2 = read_code_embeddings("Code Embeddings/code_embeddings_A2.txt")
    A2_threshold = threshold_grid_search(embeddings_A2) # A2 Threshold is  (threshold with max F1 value)
    code_similarity_analysis(embeddings_A2, "A2", A2_threshold)