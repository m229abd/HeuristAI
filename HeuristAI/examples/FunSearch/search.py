"""
Module containing seed heuristic instances and test cases for the FunSearch example.
Provides initial heuristics and corresponding test cases to evaluate their performance across multiple islands.
"""

from structures.FunSearch import HeuristicInstance
from textwrap import dedent

# Seed heuristic instances with function implementations
SEARCH_SEED_INSTANCES = [
    HeuristicInstance(
        function=dedent("""
        def bubble_sort(arr):
            n = len(arr)
            for i in range(n):
                for j in range(0, n-i-1):
                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
            return arr
        """)
    ),
    HeuristicInstance(
        function=dedent("""
        def selection_sort(arr):
            n = len(arr)
            for i in range(n):
                min_idx = i
                for j in range(i+1, n):
                    if arr[j] < arr[min_idx]:
                        min_idx = j
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
            return arr
        """)
    ),
    HeuristicInstance(
        function=dedent("""
        def insertion_sort(arr):
            for i in range(1, len(arr)):
                key = arr[i]
                j = i - 1
                while j >= 0 and key < arr[j]:
                    arr[j + 1] = arr[j]
                    j -= 1
                arr[j + 1] = key
            return arr
        """)
    )
]

# Test cases to evaluate the performance of heuristic functions
SEARCH_TEST_CASES = [
    {'test_case': [5, 2, 9, 1, 5, 6], 'answer': [1, 2, 5, 5, 6, 9]},
    {'test_case': [3, 0, 8], 'answer': [0, 3, 8]},
    {'test_case': [1, 2, 3, 4], 'answer': [1, 2, 3, 4]},
    {'test_case': [10, 9, 8, 7, 6], 'answer': [6, 7, 8, 9, 10]}
]
