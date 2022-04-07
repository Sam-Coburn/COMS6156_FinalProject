import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

# Reads and Organizes the Code Embeddings Located in the .txt File
# Outputs a dictionary with the program name as the key and its corresponding code embedding as the value
def read_code_embeddings():
    file = open("code_embeddings_A1.txt", "r")
    lines = file.readlines()

    embedding_dictionary = {}
    count = 1
    for i in range(0, len(lines)):
        if count == 11:
            count = 1

        l = lines[i].strip()
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

# Calculates the cosine similarity of two embeddings
def cosine_similarity(e1, e2):
    dot_product = 0
    for i in range(0, len(e1)):
        dot_product += (float(e1[i]) * float(e2[i]))

    e1_magnitude = 0
    for e in e1:
        e1_magnitude += (float(e)**2)
    e1_magnitude = sqrt(e1_magnitude)

    e2_magnitude = 0
    for e in e2:
        e2_magnitude += (float(e)**2)
    e2_magnitude = sqrt(e2_magnitude)

    return (dot_product) / (e1_magnitude * e2_magnitude)

# Compare the similiarity between all embeddings located in the embedding dictionary
# Positive = Same Code Origin, Negative = Otherwise
# Perform "grid search" to find the best threshold for this method which provides the best F1 score
def threshold_grid_search(embeddings):
    f1_scores = []
    x = [] # False Positive Rate
    y = [] # True Positive Rate
    for i in range(1, 101):

        threshold = i / 100

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
        
        precision = true_positive_count / (true_positive_count + false_positive_count)
        recall = true_positive_count / (true_positive_count + false_negative_count)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
        
        x.append(false_positive_count / (false_positive_count + true_negative_count))
        y.append(true_positive_count / (true_positive_count + false_negative_count))

    print("MAX F1: ", max(f1_scores))
    print("MAX F1 T: ", (f1_scores.index(max(f1_scores)) + 1))

    x_arr = np.array(x)
    y_arr = np.array(y)
    auc = np.trapz(y_arr, x = x_arr)
    print("AUC: ", auc) # Fix by sorting x (and correspondindly Y)

    plt.plot(x, y)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characterisitc (ROC) Curve")
    plt.show()

def code_similarity_analysis(embeddings):
    threshold = 0.66 # max threshold value determined by the grid search

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
    
    accuracy = (true_positive_count + true_negative_count) / ((len(k1) * len(k2)) * 100)
    precision = true_positive_count / (true_positive_count + false_positive_count)
    recall = true_positive_count / (true_positive_count + false_negative_count)
    f1 = (2 * precision * recall) / (precision + recall)
    
    print("ACCURACY: ", accuracy)
    print("PRECISION: ", precision)
    print("RECALL: ", recall)
    print("F1: ", f1)
        
embeddings = read_code_embeddings()
# threshold_grid_search(embeddings)
code_similarity_analysis(embeddings)