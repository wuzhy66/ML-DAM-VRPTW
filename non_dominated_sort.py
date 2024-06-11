# import matplotlib.pyplot as plt
import numpy as np

def non_dominated_sort(solutions):
    solen = solutions.shape[0]
    S = [[] for i in range(0, solen)]
    front = [[]]
    n=[0 for i in range(0,solen)]
    rank = [0 for i in range(0, solen)]

    for p in range(0, solen):
        S[p] = []  # solutions dominated by the $p$th solution
        n[p] = 0  # the number of solutions which dominate the $p$th solution
        for q in range(0, solen):
            if (solutions[p][0] < solutions[q][0] and solutions[p][1] < solutions[q][1]) \
                    or (solutions[p][0] <= solutions[q][0] and solutions[p][1] < solutions[q][1]) \
                    or (solutions[p][0] < solutions[q][0] and solutions[p][1] <= solutions[q][1]):
                if q not in S[p]:
                    S[p].append(q)
            elif (solutions[q][0] < solutions[p][0] and solutions[q][1] < solutions[p][1]) \
                    or (solutions[q][0] <= solutions[p][0] and solutions[q][1] < solutions[p][1]) \
                    or (solutions[q][0] < solutions[p][0] and solutions[q][1] <= solutions[p][1]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if( n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

def get_non_dominated(solutions):
    len = solutions.shape[0]
    S = [[] for i in range(0, len)]
    front = [[]]
    n = [0 for i in range(0, len)]
    rank = [0 for i in range(0, len)]

    for p in range(0, len):
        S[p] = []  # solutions dominated by the $p$th solution
        n[p] = 0  # the number of solutions which dominate the $p$th solution
        for q in range(0, len):
            if (solutions[p][0] < solutions[q][0] and solutions[p][1] < solutions[q][1]) \
                    or (solutions[p][0] <= solutions[q][0] and solutions[p][1] < solutions[q][1]) \
                    or (solutions[p][0] < solutions[q][0] and solutions[p][1] <= solutions[q][1]):
                if q not in S[p]:
                    S[p].append(q)
            elif (solutions[q][0] < solutions[p][0] and solutions[q][1] < solutions[p][1]) \
                    or (solutions[q][0] <= solutions[p][0] and solutions[q][1] < solutions[p][1]) \
                    or (solutions[q][0] < solutions[p][0] and solutions[q][1] <= solutions[p][1]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    return solutions[front[0]] #, front[0]

# if __name__ == "__main__":
#     plt.figure()
#     result1 = np.load('npy/no_meta_tsp50_teststep100.npy')
#     plt.plot(result1[:, 0], result1[:, 1], 'c+', label="DAM-100step")
#     result2 = get_non_dominated(result1)
#     plt.plot(result2[:, 0], result2[:, 1], 'r+', label="DAM-100step(non_dominated)")
#
#     plt.xlabel('f1')
#     plt.ylabel('f2')
#     plt.title('MOTSP')
#     plt.legend()
#     # leg.get_frame().set_alpha(0.5)
#     # plt.savefig('images.png')
#     plt.show()

