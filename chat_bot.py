import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import export_text
import csv
from fuzzywuzzy import process
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")



def readn(nstr):
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum += severityDictionary[item]
    if ((sum * days) / (len(exp) + 1) > 13):
        print("\nYou should take the consultation from a doctor!!")
    else:
        print("It might not be that bad but you should take precautions.")

def getDescription():
    global description_list
    description_list = {}  # Initialize the dictionary
    with open(r'MasterData\symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open(r'MasterData\Symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    precautionDictionary = {}  # Initialize the dictionary
    with open(r'MasterData\symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name?\t\t", end="->")
    name = input("")
    print("Hello,", name + "! How can I assist you today?")


def check_pattern(dis_list, inp):
    pred_list = []

    # Convert the input to lowercase for better matching
    inp_lower = inp.lower()

    # Check for exact matches
    exact_matches = [item for item in dis_list if item.lower() == inp_lower]
    if len(exact_matches) > 0:
        pred_list.append(exact_matches[0])
        return 1, pred_list

    # Check for fuzzy matches with a lower similarity threshold
    closest_match, similarity = process.extractOne(inp_lower, dis_list)
    if similarity >= 50:  # Adjust the threshold as needed (e.g., 70 instead of 80)
        pred_list.append(closest_match)
        return 1, pred_list

    print("I'm sorry, I couldn't find any matches for that symptom. Please try again.")
    return 0, []




def sec_predict(symptoms_exp):
    df = pd.read_csv(r'Data\Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    feature_names = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1

    return rf_clf.predict([input_vector])



def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        print("\nEnter the symptoms you are experiencing (comma-separated): ", end="")
        symptom_input = input("")
        symptoms_list = symptom_input.split(',')
        conf, cnf_dis = check_pattern(chk_dis, symptoms_list[0])  # You can check the pattern of the first symptom
        if conf == 1:
            break

    while True:
        try:
            num_days = int(input("\nHow many days have you been experiencing these symptoms? : "))
            break
        except ValueError:
            print("Please enter a valid number of days.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == symptoms_list[0]:  # Corrected variable name here
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("\nAre you experiencing any ")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                print(syms, "? : ", end='')
                while True:
                    inp = input("")
                    if inp == "yes" or inp == "no":
                        break
                    else:
                        print("Provide proper answers i.e. (yes/no) : ", end="")
                if inp == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if(present_disease[0]==second_prediction[0]):
                print("\nYou may have", present_disease[0])
                print(description_list[present_disease[0]])
            else:
                print("\nYou may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            precution_list=precautionDictionary[present_disease[0]]
            print("\nTake following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)

    recurse(0, 1)
  

def main():
    global le, reduced_data, severityDictionary, description_list, precautionDictionary
    description_list = {}  # Initialize the dictionary
    severityDictionary = {}  # Initialize the dictionary
    precautionDictionary = {}  # Initialize the dictionary
    training = pd.read_csv(r'Data\Training.csv')
    cols = training.columns
    cols = cols[:-1]
    x = training[cols]
    y = training['prognosis']
    y1 = y
    reduced_data = training.groupby(training['prognosis']).max()

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    clf1 = DecisionTreeClassifier()
    clf = clf1.fit(x_train, y_train)
    scores = cross_val_score(clf, x_test, y_test, cv=3)
    print("Cross-validation scores:", scores.mean())

    model = SVC()
    model.fit(x_train, y_train)
    print("SVM score:", model.score(x_test, y_test))

    getSeverityDict()
    getDescription()
    getprecautionDict()
    getInfo()
    tree_to_code(clf, cols)
    print("----------------------------------------------------------------------------------------")

if __name__ == "__main__":
    while True:
        main()
        repeat = input("Do you want to repeat the process? (yes/no): ").lower()
        if repeat == "no":
            print("Thank you for using HealthCare ChatBot! Take care and stay healthy.")
            break