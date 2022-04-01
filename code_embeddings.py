from math import sqrt

def read_code_embeddings():
    file = open("code_embeddings.txt", "r")
    lines = file.readlines()

    embedding_dictionary = {}
    count = 1
    for i in range(0, len(lines)):
        if count == 11:
            count = 1

        l = lines[i].strip()
        if i >= 0 and i <= 9:
            embedding_dictionary["BubbleSort" + str(count)] = l.split(" ")
        elif i >= 10 and i <= 19:
            embedding_dictionary["EightQueens" + str(count)] = l.split(" ")
        elif i >= 20 and i <= 29:
            embedding_dictionary["GuessWord" + str(count)] = l.split(" ")
        elif i >= 30 and i <= 39:
            embedding_dictionary["Hanoi" + str(count)] = l.split(" ")
        elif i >= 40 and i <= 49:
            embedding_dictionary["InfixConverter" + str(count)] = l.split(" ")
        elif i >= 50 and i <= 59:
            embedding_dictionary["Kaprekar" + str(count)] = l.split(" ")
        elif i >= 60 and i <= 69:
            embedding_dictionary["MagicSquare" + str(count)] = l.split(" ")
        elif i >= 70 and i <= 79:
            embedding_dictionary["RailRoadCar" + str(count)] = l.split(" ")
        elif i >= 80 and i <= 89:
            embedding_dictionary["SLinkedList" + str(count)] = l.split(" ")
        elif i >= 90 and i <= 99:
            embedding_dictionary["SqrtAlgorithm" + str(count)] = l.split(" ")

        count += 1

    return embedding_dictionary

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

def code_similiarity_analysis(embeddings):
    for k1 in list(embeddings.keys()):
        print(k1)
        for k2 in list(embeddings.keys()):
            val = cosine_similarity(embeddings[k1], embeddings[k2])
            print("* " + k2 + " " + str(val))
        print("\n")
        

embeddings = read_code_embeddings()
code_similiarity_analysis(embeddings)