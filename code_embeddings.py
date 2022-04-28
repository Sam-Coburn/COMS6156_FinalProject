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
    embeddings_file = open(file_name, "r")
    lines = embeddings_file.readlines() # Retrieve all lines from the file in a list

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
    thresholds = []

    # Instantiation of the X and Y value lists used in the ROC Curve plot
    x = [] # False Positive Rate
    y = [] # True Positive Rate

    # Cycles through threshold values in order to find highest F1 score produced by this similarity comparison method
    for i in range(1, 101):
        threshold = i / 100
        thresholds.append(i)

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
    max_F1 = max(f1_scores)
    max_threshold = (f1_scores.index(max(f1_scores)) + 1)

    print("MAX F1: ", max_F1)
    print("MAX F1 T: ", max_threshold)

    # Calculate the AUC of the ROC Curve
    x_y_tuples = []
    for i in range(0, len(x)):
        x_y_tuples.append((x[i], y[i]))

    x_y_tuples.sort(key = lambda x: x[0])
    for i in range(0, len(x_y_tuples)):
        x[i] = x_y_tuples[i][0]
        y[i] = x_y_tuples[i][1]

    x_arr = np.array(x)
    y_arr = np.array(y)
    auc = np.trapz(y_arr, x = x_arr)
    print("AUC: ", auc) # Fix by sorting x (and correspondindly Y)

    '''
    # Plot F1 Curve
    horzional_line_x = []
    horzional_line_y = []
    val = 0.0
    while val < max_F1:
        horzional_line_x.append(max_threshold)
        horzional_line_y.append(val)
        val = val + 0.0001

    plt.plot(horzional_line_x, horzional_line_y, linestyle = "dashed", color = "r")
    plt.plot(thresholds, f1_scores)
    plt.text(max_threshold, max_F1 / 8, str(max_threshold), color = "r", horizontalalignment = "center", size = "smaller", bbox=dict(facecolor='white', edgecolor="white"))
    plt.text(max_threshold, max_F1 + 0.005, str(max_F1), color = "r", horizontalalignment = "center", size = "smaller")
    plt.xlabel("Threshold Value")
    plt.ylabel("F1 Score")
    plt.title("Code Similarity Performance for Various Threshold Values")
    plt.show()

    # Plot ROC Curve
    plt.plot(x, y)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characterisitc (ROC) Curve")
    plt.show()
    '''

    return (f1_scores.index(max(f1_scores)) + 1) / 100, thresholds, f1_scores # Return the threshold value with the max observed F1 score

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
    print("F1: ", f1) # return this to graph it in bar graph
    print("\n")

    return f1

#--------------------------------------------------------------------------------------------------------------------------------
#   Function:   F1_plots	
#	Inputs:     threshold_1: list of threshold values for the original dataset
#               f1_1: list of F1 scores for the original dataset at incremental threshold values
#               threshold_2: list of threshold values for the Krakatau dataset
#               f1_2: list of F1 scores for the Krakatau dataset at incremental threshold values
#               threshold_3: list of threshold values for the Procyon datasets 
#               f1_3: list of F1 scores for the Procyon dataset at incremental threshold values
#	Outputs:    N/A
#	Purpose:	Plots a graph of F1 score vs. threshold values for all three datasets on one plot
#--------------------------------------------------------------------------------------------------------------------------------
def F1_plots(threshold_1, f1_1, threshold_2, f1_2, threshold_3, f1_3):
    plt.plot(threshold_1, f1_1, color = "r", linestyle = "solid", label = "Original")
    plt.plot(threshold_2, f1_2, color = "g", linestyle = "dashed", label = "Krakatau")
    plt.plot(threshold_3, f1_3, color = "b", linestyle = "dotted", label = "Procyon")
    plt.legend()
    plt.xlabel("Threshold Value")
    plt.ylabel("F1 Score")
    plt.title("Code Similarity Performance for Various Threshold Values")
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------
#   Function:   precision_at_n	
#	Inputs:     embeddings: embedding dictionary
#               threshold: threshold value for dataset that provides highest F1 score
#	Outputs:    N/A
#	Purpose:	Creates a plot of the precision@n for the given code embeddings using the same method described in the 
#               Similarity Analyzers paper
#--------------------------------------------------------------------------------------------------------------------------------
def precision_at_n(embeddings):
    # TP = 1
    # TN = 2
    # FP = 3
    # FN = 4
    
    results = []
    for k1 in list(embeddings.keys()):
        for k2 in list(embeddings.keys()):
            val = cosine_similarity(embeddings[k1][0], embeddings[k2][0])
            if embeddings[k1][1] == embeddings[k2][1]:
                results.append((val, 1))
            else:
                results.append((val, 3))
    
    results.sort(key = lambda x: x[0])
    results.reverse()

    x = []
    y = []
    for n in range(1, 1501):
        x.append(n)

        # Calculate precision at current value of n
        # * precision = true_positive_count / (true_positive_count + false_positive_count)
        true_positive_count = 0
        false_positive_count = 0
        for i in range(0, n):
            if results[i][1] == 1:
                true_positive_count += 1
            elif results[i][1] == 3:
                false_positive_count += 1
        
        if n == 1000:
            print("Precision@N: " + str(true_positive_count / (true_positive_count + false_positive_count)))

        y.append(true_positive_count / (true_positive_count + false_positive_count))

    '''
    plt.plot(x, y)
    plt.xlabel("N")
    plt.ylabel("Precision")
    plt.title("Precision@N Curve for the Original Dataset")
    plt.show()
    '''

#--------------------------------------------------------------------------------------------------------------------------------
#   Function:   average_r_precision
#	Inputs:     embeddings: embedding dictionary
#               test_name: name of the current experiment (used in printing of results)
#               r: number of relevant results (i.e. true positives)
#	Outputs:    N/A
#	Purpose:	Calculates the ARP for a set of embeddings as a method of error measurement
#--------------------------------------------------------------------------------------------------------------------------------
def average_r_precision(embeddings, test_name, r):
    r_precisions_sum = 0

    for k1 in list(embeddings.keys()):
        # Create a ranked list of the top r similar embeddings (i.e. set of relevant files) for each query 
        top_r_results = []
        for k2 in list(embeddings.keys()):
            val = cosine_similarity(embeddings[k1][0], embeddings[k2][0])
            if (len(top_r_results) < r):
                top_r_results.append((embeddings[k1][1], val))
            else:
                min_val = (min(top_r_results, key=lambda x: x[1]))[1]
                if val < min_val:
                    top_r_results = [(tag, similarity) for (tag, similarity) in top_r_results if similarity != min_val]
                    top_r_results.append((embeddings[k1][1], val))

        top_r_results.sort(key=lambda x: x[1])

        # Count the number of true positives in each ranked list
        true_positive_count = 0
        for k in top_r_results:
            if k[0] == embeddings[k1][1]:
                true_positive_count += 1
        
        # Calculate and keep track of the r-precision for each query
        r_precisions_sum += (true_positive_count / r)
    
    # Calculate the average r-precision for the entire set of embeddings
    arp = r_precisions_sum / len(list(embeddings.keys()))

    print("EMBEDDINGS " + test_name + " ARP:", arp)

#--------------------------------------------------------------------------------------------------------------------------------
#   Function:   average_r_precision
#	Inputs:     embeddings: embedding dictionary
#               test_name: name of the current experiment (used in printing of results)
#               r: number of relevant results (i.e. true positives)
#	Outputs:    N/A
#	Purpose:	Calculates the ARP for a set of embeddings as a method of error measurement
#--------------------------------------------------------------------------------------------------------------------------------
def average_r_precision_temp(embeddings):
    average_r_total = []
    for k1 in list(embeddings.keys()):
        results = []
        for k2 in list(embeddings.keys()):
            results.append((cosine_similarity(embeddings[k1][0], embeddings[k2][0]), (embeddings[k1][1], embeddings[k2][1])))

        results.sort(key = lambda x: x[0])
        results.reverse()

        true_positives = 0
        for i in range(0, 10):
            if results[i][1][0] == results[i][1][1]:
                true_positives = true_positives + 1
        
        average_r = true_positives / 10
        average_r_total.append(average_r)
    
    print("ARP:", sum(average_r_total)/ len(average_r_total))

#--------------------------------------------------------------------------------------------------------------------------------
#   Function:   mean_average_precision
#	Inputs:     embeddings: embedding dictionary
#               test_name: name of the current experiment (used in printing of results)
#               r: number of relevant results (i.e. true positives)
#	Outputs:    N/A
#	Purpose:	Calculates the MAP for a set of embeddings as a method of error measurement
#--------------------------------------------------------------------------------------------------------------------------------
def mean_average_precision_temp(embeddings):
    average_p_total = []
    for k1 in list(embeddings.keys()):
        results = []
        for k2 in list(embeddings.keys()):
            results.append((cosine_similarity(embeddings[k1][0], embeddings[k2][0]), (embeddings[k1][1], embeddings[k2][1])))

        results.sort(key = lambda x: x[0])
        results.reverse()

        true_positives = 0
        false_positives = 0
        current_p = []
        for i in range(0, 10):
            if results[i][1][0] == results[i][1][1]:
                true_positives = true_positives + 1
                current_p.append(true_positives / (true_positives + false_positives))
            else:
                false_positives = false_positives + 1
                current_p.append(true_positives / (true_positives + false_positives))
        
        average_p = sum(current_p)/ len(current_p)
        average_p_total.append(average_p)
    
    print("MAP:", sum(average_p_total)/ len(average_p_total))

#--------------------------------------------------------------------------------------------------------------------------------
#   Function:   mean_average_precision
#	Inputs:     embeddings: embedding dictionary
#               test_name: name of the current experiment (used in printing of results)
#               r: number of relevant results (i.e. true positives)
#	Outputs:    N/A
#	Purpose:	Calculates the MAP for a set of embeddings as a method of error measurement
#--------------------------------------------------------------------------------------------------------------------------------
def mean_average_precision(embeddings, test_name, r):
    avg_precisions_sum = 0

    for k1 in list(embeddings.keys()):

        # Create a ranked list of the top r similar embeddings (i.e. set of relevant files) for each query 
        top_r_results = []
        for k2 in list(embeddings.keys()):
            val = cosine_similarity(embeddings[k1][0], embeddings[k2][0])
            if (len(top_r_results) < r):
                top_r_results.append((embeddings[k1][1], val))
            else:
                min_val = (min(top_r_results, key=lambda x: x[1]))[1]
                if val < min_val:
                    top_r_results = [(tag, similarity) for (tag, similarity) in top_r_results if similarity != min_val]
                    top_r_results.append((embeddings[k1][1], val))
        
        top_r_results.sort(key=lambda x: x[1])

        precision_at_n_sum = 0
        true_positive_count = 0
        
        # Calculate the precision@n for each query
        for i in range(1, r + 1):
            if top_r_results[i - 1][0] == embeddings[k1][1]:
                true_positive_count += 1
            precision_at_n_sum += (true_positive_count / i)
        
        # Calculate and keep track of the average precisions@n for each query
        avg_precisions_sum += (precision_at_n_sum / r)
    
    # Calculate the average precisions@n for the entire set of embeddings
    mean_precision = avg_precisions_sum / len(list(embeddings.keys()))

    print("EMBEDDINGS " + test_name + " MAP", mean_precision)
    
# Tests for Code Embeddings
# * separated into two experiments: embeddings trained using one model (model A) and embeddings using another, larger model (model B)
# * for each model, there are three sets of embeddings: the normal dataset, and two datasets for the original files passed through decompilers
if __name__ == "__main__":
    # Dataset A - Embeddings Created Using Java-14 Model
    # EMBEDDINGS A1 - Original
    embeddings_A1 = read_code_embeddings("Code Embeddings/code_embeddings_A1.txt")
    #A1_threshold, A1_thresholds, A1_F1s = threshold_grid_search(embeddings_A1) 
    #A1_F1 = code_similarity_analysis(embeddings_A1, "A1", A1_threshold) 
    precision_at_n(embeddings_A1)
    # average_r_precision(embeddings_A1, "A1", 10)
    average_r_precision_temp(embeddings_A1)
    mean_average_precision_temp(embeddings_A1)
    #mean_average_precision(embeddings_A1, "A1", 10)

    '''
    # EMBEDDINGS A2 - Krakatau 
    embeddings_A2 = read_code_embeddings("Code Embeddings/code_embeddings_A2.txt")
    A2_threshold, A2_thresholds, A2_F1s = threshold_grid_search(embeddings_A2) 
    A2_F1 = code_similarity_analysis(embeddings_A2, "A2", A2_threshold)

    # EMBEDDINGS A3 - Procyon
    embeddings_A3 = read_code_embeddings("Code Embeddings/code_embeddings_A3.txt")
    A3_threshold, A3_thresholds, A3_F1s = threshold_grid_search(embeddings_A3)
    A3_F1 = code_similarity_analysis(embeddings_A3, "A3", A3_threshold)

    F1_plots(A1_thresholds, A1_F1s, A2_thresholds, A2_F1s, A3_thresholds, A3_F1s)
    '''
    
    # Dataset B - Embeddings Created Using Java-Large Model
    # EMBEDDINGS B1 - Original
    embeddings_B1 = read_code_embeddings("Code Embeddings/code_embeddings_B1.txt")
    #B1_threshold, B1_thresholds, B1_F1s = threshold_grid_search(embeddings_B1) 
    #B1_F1 = code_similarity_analysis(embeddings_B1, "B1", B1_threshold)
    precision_at_n(embeddings_B1)
    #average_r_precision(embeddings_B1, "B1", 10)
    #mean_average_precision(embeddings_B1, "B1", 10)
    average_r_precision_temp(embeddings_B1)
    mean_average_precision_temp(embeddings_B1)

    '''
    # EMBEDDINGS B2 - Krakatau 
    embeddings_B2 = read_code_embeddings("Code Embeddings/code_embeddings_B2.txt")
    B2_threshold, B2_thresholds, B2_F1s = threshold_grid_search(embeddings_B2) 
    B2_F1 = code_similarity_analysis(embeddings_B2, "B2", B2_threshold)

    # # EMBEDDINGS B3 - Procyon
    embeddings_B3 = read_code_embeddings("Code Embeddings/code_embeddings_B3.txt")
    B3_threshold, B3_thresholds, B3_F1s = threshold_grid_search(embeddings_B3)
    B3_F1 = code_similarity_analysis(embeddings_B3, "B3", B3_threshold) 

    F1_plots(B1_thresholds, B1_F1s, B2_thresholds, B2_F1s, B3_thresholds, B3_F1s)
    '''