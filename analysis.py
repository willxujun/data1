import readqaplib as qaplib
import os
import pandas as pd
import numpy as np
from test_qap import QAPEvaluator
import utils.index as idx
import ast


def solution_matrix(solution,n,m):
    solution = ast.literal_eval(solution)
    solution_mtx = np.zeros((n, m), dtype=np.int8)
    for i in range(1,n+1):
        for j in range(1,m+1):
            index = idx.index_1_q_to_l_1(i,j,m) - 1
            solution_mtx[i-1][j-1] = solution[index]
    return solution_mtx


def check_mtx(solution_mtx):
    size = solution_mtx.shape[0]
    test_ct1 = True
    test = np.zeros(size)
    for i in range(size):
        test += solution_mtx[i,:]
    result = test !=1
    if np.any(result):
        test_ct1 = False
    
    test_ct2 = True
    test = np.zeros(size)
    for i in range(size):
        test += solution_mtx[:,i]
    result = test != 1
    if np.any(result):
        test_ct2 = False
    
    return [test_ct1, test_ct2]

problems = {}
da_best_feasible_solutions = {}
sw_best_feasible_solutions = {}

for filename in os.listdir('.'):
    if filename.endswith('tai30a.dat'):
        problems[filename] = qaplib.readqaplib(filename)

for filename in os.listdir('.'):
    if filename.endswith('.csv'):
        components = filename.split('_')
        dataset = components[1]
        if dataset=='tai30a.dat': 
            hardware = components[0]

            df = pd.read_csv(filename)
            F,D = problems[dataset]
            size = F.shape[0]
            evaluator = QAPEvaluator(size,size,F,D)
            for config in df['problem']:
                matrix = solution_matrix(config,size,size)
                if check_mtx(matrix):
                    print("feasible solution found")
                    if hardware =='da':
                        answer = evaluator.run(matrix)
                        if dataset in da_best_feasible_solutions:
                            if answer < da_best_feasible_solutions[dataset]:
                                da_best_feasible_solutions[dataset] = answer
                        else:
                            da_best_feasible_solutions[dataset] = answer
                    else:
                        answer = evaluator.run(matrix)
                        if dataset in sw_best_feasible_solutions:
                            if answer < sw_best_feasible_solutions[dataset]:
                                sw_best_feasible_solutions[dataset] = answer
                        else:
                            sw_best_feasible_solutions[dataset] = answer
                    break

print(da_best_feasible_solutions)
print(sw_best_feasible_solutions)

with open('tai30abest.txt','r') as f:
    sol = f.read().split()
    solution_mtx = np.zeros((30,30))
    for i in range(30):
        print(sol[i])
        solution_mtx[i][int(sol[i])-1] = 1
    
    F,D = problems['tai30a.dat']
    evaluator = QAPEvaluator(30,30,F,D)
    print(evaluator.run(solution_mtx))