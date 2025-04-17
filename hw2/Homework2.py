import csv

# Implement a stack and use this stack for question 1 and 2 
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        return len(self.items) == 0

# QUESTION 1: Valid Parentheses

def isValidParentheses(s: str) -> bool:
    if not s:
        return True  # Empty string is valid
    
    stack = Stack()
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if stack.is_empty():
                return False
            top_element = stack.pop()
            if mapping[char] != top_element:
                return False
        else:
            stack.push(char)
    
    return stack.is_empty()

# QUESTION 2: Evaluate Postfix Expression

def evaluatePostfix(exp: str) -> int:
    stack = Stack()
    tokens = exp.split()
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.push(a + b)
            elif token == '-':
                stack.push(a - b)
            elif token == '*':
                stack.push(a * b)
            elif token == '/':
                stack.push(int(a / b))  # Ensure integer division
        else:
            stack.push(int(token))
    
    return stack.pop()

#DO NOT MODIFY BELOW CODE
if __name__ == "__main__":

    # QUESTION 1: Read Test Cases
    testcasesforquestion1 = []
    try:
        with open('question1.csv', 'r') as file:
            testCases = csv.reader(file)
            for row in testCases:
                testcasesforquestion1.append(row)
    except FileNotFoundError:
        print("File Not Found: question1.csv")

    # Running Test Cases for Question 1
    print("QUESTION 1 TEST CASES")
    for i, (inputValue, expectedOutput) in enumerate(testcasesforquestion1, start=1):
        actualOutput = isValidParentheses(inputValue)
        expectedBool = expectedOutput.lower() == 'true'  # Convert string to boolean
        if expectedBool == actualOutput:
            print(f"Test Case {i} : PASSED")
        else:
            print(f"Test Case {i}: Failed (Expected : {expectedOutput}, Actual: {actualOutput})")

    #QUESTION 2
    testcasesforquestion2 = []
    try:
        with open('question2.csv','r') as file:
            testCases = csv.reader(file)
            for row in testCases:
                testcasesforquestion2.append(row)
    except FileNotFoundError:
        print("File Not Found: question2.csv") 
    
    print("QUESTION 2 TEST CASES")
    #Running Test Cases for Question 2
    for i , (inputValue,expectedOutput) in enumerate(testcasesforquestion2,start=1):
        actualOutput = evaluatePostfix(inputValue)
        if(int(expectedOutput) == actualOutput):
            print(f"Test Case {i} : PASSED") 
        else:
            print(f"Test Case {i}: Failed (Expected : {expectedOutput}, Actual: {actualOutput})")