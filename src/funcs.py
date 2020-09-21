import numpy as np
import pandas as pd
import re

# Takes a numpy array and folds it from left to right
def foldl(arr, bin_op, initial):
    for x in arr:
        initial = bin_op(initial,x)
    return initial

# Takes a numpy array and folds it from right to left
def foldr(arr, bin_op, initial):
    for x in np.flip(arr):
        initial = bin_op(x,initial)
    return initial

def actSceneLineConvert(acl): # string -> 3-tuple of integers
    regex = '([1-9]{1,2})[.]([1-9]{1,2})[.]([1-9]{1,3})'
    if acl != acl:
        return (0,0,0)
    m = re.search(regex, acl)
    return (int(m.group(1)),int(m.group(2)),int(m.group(3)))