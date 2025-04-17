import csv

testcasesforquestion1 = []
with open('question1.csv', 'r') as file:
    testCases = csv.reader(file)
    for row in testCases:
        testcasesforquestion1.append(row)

print(testcasesforquestion1)