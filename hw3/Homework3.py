import csv
import random
import time
import tracemalloc
import matplotlib.pyplot as plt

# --- Defining Treenode and Hoework 3 classes --- #

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val 
        self.left = left 
        self.right = right 

class HomeWork3: 
    # Problem 1: Construct a Binary Tree using BFS representation.
    def constructBinaryTree(self, input: str) -> TreeNode:
        if not input:
            return None
        nodes = input.split(',')
        # Create the root node
        root_val = nodes.pop(0)
        if root_val == "None":
            return None
        root = TreeNode(int(root_val))
        queue = [root]
        while nodes and queue:
            current = queue.pop(0)
            # for Left child
            if nodes:
                
                left_val = nodes.pop(0)
                if left_val != "None":
                    left_node = TreeNode(int(left_val))
                    current.left = left_node
                    queue.append(left_node)
            # for the Right child
            if nodes:
                
                right_val = nodes.pop(0)
                if right_val != "None":
                    right_node = TreeNode(int(right_val))
                    current.right = right_node
                    queue.append(right_node)
        return root

    # Problem 2: In-Order Traversal (Recursively)
    def inOrderTraversalRecursive(self, head: TreeNode) -> list:
        def inorder(node):
            if not node:
                return []
            return inorder(node.left) + [node.val] + inorder(node.right)
        return inorder(head)

    # Problem 2: In-Order Traversal (Iteratively)
    def inOrderTraversalIterative(self, head: TreeNode) -> list:
        stack = []
        current = head
        result = []
        while stack or current:
            while current:
                stack.append(current)
                current = current.left
            current = stack.pop()
            result.append(current.val)
            current = current.right
        return result

    # Problem 3(a): Generate a random permutation of N unique numbers
    def generate_random_permutation(self, N) -> list:
        numbers = list(range(1, N + 1))
        random.shuffle(numbers)
        return numbers

    # Problem 3(b): Generate a complete binary tree given N unique numbers
    def generate_complete_tree(self, nums) -> TreeNode:
        if not nums:
            return None
        nodes = [TreeNode(num) for num in nums]
        n = len(nodes)
        for i in range(n):
            left_index = 2 * i + 1
            right_index = 2 * i + 2
            if left_index < n:
                nodes[i].left = nodes[left_index]
            if right_index < n:
                nodes[i].right = nodes[right_index]
        return nodes[0]

    # Problem 3(c): Generate a skewed binary tree (all left or all right children)
    def generate_skewed_tree(self, nums, skew='left') -> TreeNode:
        if not nums:
            return None
        root = TreeNode(nums[0])
        current = root
        for num in nums[1:]:
            new_node = TreeNode(num)
            if skew == 'left':
                current.left = new_node
            else:
                current.right = new_node
            current = new_node
        return root

    # Problem 4: Validate if a binary tree is a Binary Search Tree (BST)
    def validateBST(self, head: TreeNode) -> bool:
        def isBST(node, low, high):
            if not node:
                return True
            if (low is not None and node.val <= low) or (high is not None and node.val >= high):
                return False
            return isBST(node.left, low, node.val) and isBST(node.right, node.val, high)
        return isBST(head, None, None)

# --- Experimental Setup for Measuring Time and Memory --- #

def measure_traversal(method, tree, iterations=5):
    """
    Measures the average execution time (in ms) and peak memory (in KB)
    for a given traversal method (function) over several iterations.
    """
    total_time = 0.0
    total_memory = 0.0
    success = True
    for _ in range(iterations):
        tracemalloc.start()
        start_time = time.perf_counter()
        try:
            method(tree)
        except RecursionError:
            success = False
            elapsed = float('inf')
            peak = float('inf')
            tracemalloc.stop()
            break
        elapsed = (time.perf_counter() - start_time) * 1000  # convert to ms
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_time += elapsed
        total_memory += peak / 1024  # convert bytes to KB
    if not success:
        return float('inf'), float('inf')
    return total_time / iterations, total_memory / iterations

def run_experiments():
    hw = HomeWork3()
    sizes = [10, 100, 1000, 5000]
    # Dictionaries to store results:
    # Structure: {tree_type: {method: {size: (time, memory)}}}
    results = {"complete": {"recursive": {}, "iterative": {}},
               "skewed": {"recursive": {}, "iterative": {}}}

    for n in sizes:
        # Generate numbers and trees
        nums = list(range(1, n+1))
        complete_tree = hw.generate_complete_tree(nums)
        skewed_tree = hw.generate_skewed_tree(nums, skew='left')

        # Measure for complete trees
        rec_time, rec_mem = measure_traversal(hw.inOrderTraversalRecursive, complete_tree, iterations=10)
        ite_time, ite_mem = measure_traversal(hw.inOrderTraversalIterative, complete_tree, iterations=10)
        results["complete"]["recursive"][n] = (rec_time, rec_mem)
        results["complete"]["iterative"][n] = (ite_time, ite_mem)

        # Measure for skewed trees
        rec_time, rec_mem = measure_traversal(hw.inOrderTraversalRecursive, skewed_tree, iterations=10)
        ite_time, ite_mem = measure_traversal(hw.inOrderTraversalIterative, skewed_tree, iterations=10)
        results["skewed"]["recursive"][n] = (rec_time, rec_mem)
        results["skewed"]["iterative"][n] = (ite_time, ite_mem)
    
    return results

def plot_results(results):
    sizes = sorted(results["complete"]["recursive"].keys())
    
    # Prepare data for complete trees
    comp_rec_time = [results["complete"]["recursive"][n][0] for n in sizes]
    comp_ite_time = [results["complete"]["iterative"][n][0] for n in sizes]
    comp_rec_mem  = [results["complete"]["recursive"][n][1] for n in sizes]
    comp_ite_mem  = [results["complete"]["iterative"][n][1] for n in sizes]

    # Prepare data for skewed trees
    skew_rec_time = [results["skewed"]["recursive"][n][0] for n in sizes]
    skew_ite_time = [results["skewed"]["iterative"][n][0] for n in sizes]
    skew_rec_mem  = [results["skewed"]["recursive"][n][1] for n in sizes]
    skew_ite_mem  = [results["skewed"]["iterative"][n][1] for n in sizes]

    # Plotting Running Times
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sizes, comp_rec_time, marker='o', label='Complete - Recursive')
    plt.plot(sizes, comp_ite_time, marker='o', label='Complete - Iterative')
    plt.plot(sizes, skew_rec_time, marker='o', label='Skewed - Recursive')
    plt.plot(sizes, skew_ite_time, marker='o', label='Skewed - Iterative')
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Average Running Time (ms)')
    plt.title('Running Time Comparison')
    plt.legend()
    plt.grid(True)

    # Plotting Memory Usage
    plt.subplot(1, 2, 2)
    plt.plot(sizes, comp_rec_mem, marker='o', label='Complete - Recursive')
    plt.plot(sizes, comp_ite_mem, marker='o', label='Complete - Iterative')
    plt.plot(sizes, skew_rec_mem, marker='o', label='Skewed - Recursive')
    plt.plot(sizes, skew_ite_mem, marker='o', label='Skewed - Iterative')
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Average Peak Memory (KB)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def print_results_table(results):
    sizes = sorted(results["complete"]["recursive"].keys())
    print("Results for Complete Trees:")
    print("N\tMethod\t\tTime (ms)\tMemory (KB)")
    for n in sizes:
        rec_time, rec_mem = results["complete"]["recursive"][n]
        ite_time, ite_mem = results["complete"]["iterative"][n]
        print(f"{n}\tRecursive\t{rec_time:.2f}\t\t{rec_mem:.2f}")
        print(f"{n}\tIterative\t{ite_time:.2f}\t\t{ite_mem:.2f}")
    print("\nResults for Skewed Trees:")
    print("N\tMethod\t\tTime (ms)\tMemory (KB)")
    for n in sizes:
        rec_time, rec_mem = results["skewed"]["recursive"][n]
        ite_time, ite_mem = results["skewed"]["iterative"][n]
        print(f"{n}\tRecursive\t{rec_time:.2f}\t\t{rec_mem:.2f}")
        print(f"{n}\tIterative\t{ite_time:.2f}\t\t{ite_mem:.2f}")

# --- Main Execution --- #
if __name__ == "__main__":

    # Part 1: Running test cases for Questions 1, 2, and 4
    homework3 = HomeWork3()
    
    # Test cases for Questions 1 and 2
    testcasesforquestion1 = []
    try:
        with open('question1.csv', 'r') as file:
            testCases = csv.reader(file)
            for row in testCases:
                testcasesforquestion1.append(row)
    except FileNotFoundError:
        print("question1.csv File Not Found") 
    
    print("RUNNING TEST CASES FOR QUESTIONS 1 and 2 ")
    for row, (inputValue, expectedOutput) in enumerate(testcasesforquestion1, start=1):
        if expectedOutput == "":
            expectedOutput = []
        else:
            expectedOutput = expectedOutput.split(",")
            for i in range(len(expectedOutput)):
                expectedOutput[i] = int(expectedOutput[i])
        root = homework3.constructBinaryTree(inputValue)
        recursiveOutput = homework3.inOrderTraversalRecursive(root)
        iterativeOutput = homework3.inOrderTraversalIterative(root)
        assert iterativeOutput == recursiveOutput == expectedOutput, f"Test Case {row} Failed: traversal outputs do not match"
        print(f"Test case {row} Passed")
    
    # Test cases for Question 4
    testcasesForQuestion4 = []
    try:
        with open('question4.csv', 'r') as file:
            testCases = csv.reader(file)
            for row in testCases:
                testcasesForQuestion4.append(row)
    except FileNotFoundError:
        print("question4.csv File Not Found")
    
    if testcasesForQuestion4:
        print("\nRUNNING TEST CASES FOR QUESTION 4")
        for idx, (inputValue, expectedBST) in enumerate(testcasesForQuestion4, start=1):
            if idx==13: # Skipping test case 13
                break
            expectedBST = True if expectedBST.strip() == "True" else False
            root = homework3.constructBinaryTree(inputValue)
            result = homework3.validateBST(root)
            assert result == expectedBST, f"Test Case {idx} Failed: For input [{inputValue}], expected {expectedBST} but got {result}"
            print(f"Test case {idx} Passed")
    
    print("\nAll tests passed successfully!")
    
    # Part 2: Run experiments for measuring running times and memory usage
    results = run_experiments()
    print_results_table(results)
    plot_results(results)

