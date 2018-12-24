import pandas as pd
import numpy as np

def Hits(G, theta=1e-5, print_interation_count=False):
    setG = set(np.asarray(G).flatten())
    nvertex = len(setG)
    auth = np.ones(nvertex+1) # authoritative
    hub = np.ones(nvertex+1) # hub
    
    # graph relation
    pa = [[] for _ in range(nvertex+1)] # parient
    ch = [[] for _ in range(nvertex+1)] # children
    for m, n in G:
        # m -> n
        pa[n].append(m)
        ch[m].append(n)

    # power on iteration
    t = 1
    while True:
        # print('iter: ', t)
        new_auth = np.zeros(nvertex+1)
        new_hub = np.zeros(nvertex+1)
        for v in setG: # for each node in graph
            for w in pa[v]:
                new_auth[v] += hub[w]

            for w in ch[v]:
                new_hub[v] += auth[w]

        # normalization
        new_auth /= np.sum(new_auth[1:])
        new_hub /= np.sum(new_hub[1:])

        # check converage
        diff = (np.sum(np.abs(new_auth[1:] - auth[1:])) + np.sum(np.abs(new_hub[1:] - hub[1:])))
        # print('diff: ', diff)
        if (diff < theta): break
        auth = new_auth
        hub = new_hub
        t += 1
        
    if print_interation_count:
        print('converage at {} iterations.'.format(t))
        
    return (new_auth[1:], new_hub[1:])




def PageRank(G, theta =1e-5, damping_factor=0.15, print_interation_count=False):
    setG = set(np.asarray(G).flatten())
    nvertex = len(setG)

    # Page Rank initial value
    # set vertex 1 with index 1
    PR = np.ones(nvertex+1) / (nvertex) 

    # build graph relation
    pa = [[] for _ in range(nvertex+1)] # parient
    outdeg = np.zeros(nvertex+1) # out degree
    for m, n in G:
        # m -> n
        pa[n].append(m)
        outdeg[m] += 1

    t = 1
    while True:
        # print('iter: ', t)
        # power on interation
        new_PR = np.zeros(nvertex+1)
        for node in setG: # for each node in graph
            # print('node: ', node)
            for p in pa[node]: # find the parient of this node
                # print(p)
                new_PR[node] += PR[p]/outdeg[p]

            # damping factor
            new_PR[node] = (damping_factor/nvertex) + (1-damping_factor)*new_PR[node]

        # print('PR: ', new_PR[1:])
        diff = np.sum(np.abs(new_PR[1:] - PR[1:])) # calculate the differience with previous iteration
        # print('diff: ', diff)
        if (diff < theta): break
        PR = new_PR
        t += 1
        
    if print_interation_count:
        print('converage at {} iterations.'.format(t))
        
    return new_PR[1:]


def SimRank(G, C=0.5, theta=1e-5, print_interation_count=True):
    setG = set(np.asarray(G).flatten())
    nvertex = len(setG)

    # SimRank init value
    simrank = np.zeros((nvertex+1, nvertex+1)) 

    # build graph relation
    inlink = [[] for _ in range(nvertex+1)]
    for m, n in G:
        # m --> n
        inlink[n].append(m)

    # power on iteration
    t = 1
    while True:
        new_simrank = np.zeros((nvertex+1, nvertex+1))
        for a in range(1, nvertex+1):
            for b in range(1, nvertex+1):
                # print(a, b)

                # simrank(a,a)=1
                if a == b:
                    new_simrank[a][b] = 1 
                    continue

                # count inlink number
                inlinkNum_a = len(inlink[a])
                inlinkNum_b = len(inlink[b])
                if inlinkNum_a==0 or inlinkNum_b==0: continue

                tmp = 0
                for i in inlink[a]:
                    for j in inlink[b]:
                        if i == j:
                            tmp += 1
                        else:
                            tmp += simrank[i, j]

                new_simrank[a][b] = C/(inlinkNum_a*inlinkNum_b)*tmp

        diff = np.sum(np.abs(new_simrank[1:, 1:] - simrank[1:, 1:])) # l1 norm
        if diff < theta: break # check converage
        simrank = new_simrank # if not converage, keep iter
        t += 1

    if print_interation_count:
        print('converage at {} iterations.'.format(t))
        
    return(new_simrank[1:, 1:])
