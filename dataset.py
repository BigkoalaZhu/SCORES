import torch
from torch.utils import data
from scipy.io import loadmat
from random import shuffle
from random import randint
import numpy as np
from torch.autograd import Variable
import copy
from itertools import combinations

class Tree(object):
    class Node(object):
        def __init__(self, leaf=None, left=None, right=None, nType=None, sym=None, wbox=None, lab=None, symIdx=None, idx=None):
            self.tempData = False
            self.inComplete = False
            self.valid = True
            self.box = None
            self.symIdx = symIdx
            self.accNum = 1
            if symIdx is None:
                self.symIdx = -1
            if leaf is not None:
                self.childNum = 1
                self.box = leaf
                self.box_noise = leaf
                self.original_box = leaf
                self.tmpBox = leaf
                self.tmpAccBox = leaf
            self.sym = None
            self.parent = None
            if sym is not None:
                self.sym = sym
                self.sym_noise = sym
                self.original_sym = sym
                self.tmpSym = sym
                self.tmpAccSym = sym
            self.left = left
            self.right = right
            self.nType = nType
            self.vqloss = -1
            self.lrIndicator = -1
            if idx is not None:
                self.idx = idx
            if nType == 0:
                self.label = torch.FloatTensor([1,0,0]).unsqueeze(0)
                self.nlabel = torch.LongTensor([nType])
            if nType == 1:
                self.childNum = left.childNum + right.childNum
                self.label = torch.FloatTensor([0,1,0]).unsqueeze(0)
                self.nlabel = torch.LongTensor([nType])
            if nType == 2:
                self.childNum = left.childNum
                self.label = torch.FloatTensor([0,0,1]).unsqueeze(0)
                self.nlabel = torch.LongTensor([nType])
            

        def is_leaf(self):
            return self.box is not None

    def __init__(self, boxes, ops, syms):
        buffer = [b for b in torch.split(boxes, 1, 0)]
        sympara = [s for s in torch.split(syms, 1, 0)]
        opnum = ops.size()[1]
        queue = []
        self.leves = []
        self.symNodes = []
        self.allnodes = []
        buffer.reverse()
        sympara.reverse()
        nodeSize = 0
        leafSize = 0
        lossSize = 0
        symSize = 0
        for i in range(opnum):
            if ops[0, i] == 0:
                n = Tree.Node(leaf=buffer.pop(), nType=0)
                self.leves.append(n)
                queue.append(n)
                nodeSize = nodeSize + 1
                leafSize = leafSize + 1
                lossSize = lossSize + 1
            if ops[0, i] == 1:
                n_right = queue.pop()
                n_left = queue.pop()
                n = Tree.Node(left=n_left, right=n_right, nType=1)
                n_left.parent = n
                n_right.parent = n
                queue.append(n)
                nodeSize = nodeSize + 1
                self.allnodes.append(n)
            if ops[0, i] == 2:
                n_left = queue.pop()
                sym = sympara.pop()
                n = Tree.Node(left=n_left, sym=sym, nType=2, symIdx=symSize)
                n_left.parent = n
                queue.append(n)
                self.setSymForAllKids(sym, n_left, symSize)
                self.symNodes.append(n)
                self.allnodes.append(n)
                nodeSize = nodeSize + 1
                lossSize = lossSize + 1
                symSize = symSize + 1
        assert len(queue) == 1
        leftNum = self.setLrForNodes(queue[0].left, 0)
        self.setLrForNodes(queue[0].right, 1)
        self.root = queue[0]
        self.num = nodeSize
        self.boxNum = leafSize
        self.lossNum = lossSize
        self.symNum = symSize
        self.inCompleteNum = -1
        self.completeNum = -1
        self.leftNodeNum = leftNum

    def addNoise(self):
        for i in range(len(self.leves)):
            box = self.leves[i].box
            s = box.size()[0]
            noise1 = box.new(s, 3).normal_(0, 0.08)
            noise2 = box.new(s, 3).normal_(0, 0.03)
            noise3 = box.new(s, 3).normal_(0, 0.08)
            noise4 = box.new(s, 3).normal_(0, 0.03)
            noise = torch.cat((noise1, noise2, noise3, noise4), 1)
            self.leves[i].box_noise = box + noise
        
        for i in range(len(self.symNodes)):
            sym = self.symNodes[i].sym
            noise = sym.new(sym.size()).normal_(0, 0.05)
            self.symNodes[i].sym_noise = sym + noise

    def mergedTree(self, boxes):
        buffer = list(boxes)
        shuffle(buffer)
        self.leves = buffer
        self.boxNum = len(boxes)
        self.num = len(boxes)
        self.lossNum = len(boxes)
        queue = []
        for i in range(len(buffer)):
            if buffer[i].sym is not None:
                queue.append(Tree.Node(left=buffer[i], sym=buffer[i].sym, nType=2))
                self.lossNum = self.lossNum + 1
                self.num = self.num + 1
            else:
                queue.append(buffer[i])
        while len(queue) != 1:
            n_left = queue.pop()
            n_right = queue.pop()
            queue.append(Tree.Node(left=n_left, right=n_right, nType=1))
            self.num = self.num + 1
            shuffle(queue)
        self.root = queue[0]

    def mergeSampleHierForNodes(self, boxes, symSize):
        buffer = boxes
        queue = []

        if symSize > 0:
            for idx in range(symSize):
                subQueue = []
                for i in range(len(buffer)):
                    if buffer[i].sym is not None:
                        if buffer[i].symIdx == idx:
                            subQueue.append(buffer[i])
                            sym = buffer[i].sym
                if len(subQueue) == 0:
                    continue
                while len(subQueue) != 1:
                    shuffle(subQueue)
                    n_left = subQueue.pop()
                    n_right = subQueue.pop()
                    subQueue.append(Tree.Node(left=n_left, right=n_right, nType=1))
                queue.append(Tree.Node(left=subQueue[0], sym=sym, nType=2))

        for i in range(len(buffer)):
            if buffer[i].symIdx == -1:
                queue.append(buffer[i])
        
        shuffle(buffer)
        if len(queue) == 0:
            shuffle(buffer)
        while len(queue) != 1:
            n_left = queue.pop()
            n_right = queue.pop()
            queue.append(Tree.Node(left=n_left, right=n_right, nType=1))
            shuffle(queue)
        return queue[0]

    def mergeNodes(self, boxes):
        buffer = boxes
        shuffle(buffer)
        queue = []
        for i in range(len(buffer)):
            if buffer[i].sym is not None:
                queue.append(Tree.Node(left=buffer[i], sym=buffer[i].sym, nType=2))
            else:
                queue.append(buffer[i])
        while len(queue) != 1:
            n_left = queue.pop()
            n_right = queue.pop()
            queue.append(Tree.Node(left=n_left, right=n_right, nType=1))
            shuffle(queue)
        return queue[0]

    def mergedTreeWithLabel(self, boxes):
        buffer = list(boxes)
        self.leves = buffer
        self.boxNum = len(boxes)
        labelNum = 5
        groups = [[] for l in range(labelNum)]
        index = 0
        for b in boxes:
            _, l = torch.max(b.nLabel, 1)
            groups[int(l[0])].append(b)
            index = index + 1
        
        queue = []
        if len(groups[1]) != 0:
            queue.append(self.mergeNodes(groups[1]))
        
        if len(groups[2]) != 0:
            queue.append(self.mergeNodes(groups[2]))
        
        if len(groups[3]) != 0:
            queue.append(self.mergeNodes(groups[3]))
        
        if len(groups[0]) != 0:
            queue.append(self.mergeNodes(groups[0]))
        
        if len(groups[4]) != 0:
            queue.append(self.mergeNodes(groups[4]))

        queue.reverse()

        while len(queue) != 1:
            n_left = queue.pop()
            n_right = queue.pop()
            queue.append(Tree.Node(left=n_left, right=n_right, nType=1))

        self.root = queue[0]    

    def setSymForAllKids(self, symPara, node, symIdx):
        node.sym = symPara
        node.symIdx = symIdx
        buffer = [node]
        while len(buffer) != 0:
            n = buffer.pop()
            if n.left is not None:
                n.left.sym = symPara
                n.left.symIdx = symIdx
                buffer.append(n.left)
            if n.right is not None:
                n.right.sym = symPara
                n.right.symIdx = symIdx
                buffer.append(n.right)
    
    def setLrForNodes(self, node, lr):
        buffer = [node]
        count = 0
        while len(buffer) != 0:
            n = buffer.pop()
            if n.left is not None:
                if n.left.box is not None:
                    n.left.lrIndicator = lr
                    count = count + 1
                buffer.append(n.left)
            if n.right is not None:
                if n.right.box is not None:
                    n.right.lrIndicator = lr
                    count = count + 1
                buffer.append(n.right)
        return count
    
    def allValid(self):
        
        for i in range(self.boxNum):
            self.leves[i].valid = True
        for i in range(self.symNum):
            self.symNodes[i].valid = True

    def sampleLeftNodeIncompleteNoise(self):

        self.root.left.right.valid = False

    def sampleRightNodeIncompleteNoise(self):
        
        #for i in range(self.boxNum):
        #    self.leves[i].valid = True
        #for i in range(self.symNum):
        #    self.symNodes[i].valid = True

        successFlag = 1
        while successFlag is 1:
            removeIdx = randint(0, self.boxNum-1)
            if self.leves[removeIdx].lrIndicator is not 0:
                continue
            self.leves[removeIdx].valid = False
            for i in range(self.symNum):
                if self.symNodes[i].left.valid == False:
                    self.symNodes[i].valid = False
            successFlag = 0
        
    def sampleIncompleteNoise(self):
        self.inCompleteNum = -1
        self.completeNum = self.num - self.boxNum
        removeIdx = randint(0, self.boxNum-1)
        for i in range(self.boxNum):
            self.leves[i].valid = True
        for i in range(self.symNum):
            self.symNodes[i].valid = True
        self.leves[removeIdx].valid = False
        self.leves[removeIdx].inComplete = True
        while self.root.left.valid == False or self.root.right.valid == False:
            self.leves[removeIdx].valid = True
            self.leves[removeIdx].inComplete = False
            removeIdx = randint(0, self.boxNum-1)
            self.leves[removeIdx].valid = False
            self.leves[removeIdx].inComplete = True
        self.inCompleteNum = 0
        currentIncomplet = self.leves[removeIdx]
        for i in range(self.symNum):
            if self.symNodes[i].left.valid == False:
                self.symNodes[i].valid = False
                self.symNodes[i].inComplete = True
                currentIncomplet = self.symNodes[i]
                self.inCompleteNum = self.inCompleteNum + 1
        
        currentIncomplet = currentIncomplet.parent
        while currentIncomplet is not self.root:
            currentIncomplet.inComplete = True
            currentIncomplet = currentIncomplet.parent
            self.inCompleteNum = self.inCompleteNum + 1
    
    def sampleRandomPos(self):
        box = self.leves[0].box
        s = box.size()[0]
        noise1 = box.new(s, 3).normal_(0, 0.03)
        noise2 = torch.zeros(s, 3)
        noise3 = torch.zeros(s, 3)
        noise4 = torch.zeros(s, 3)
        noise = torch.cat((noise1, noise2, noise3, noise4), 1)
        for i in range(self.boxNum):
            self.leves[i].box = self.leves[i].original_box + noise


def EqualSymPara(s1, s2):
    if s1[0] is not s2[0]:
        return False
    
    if s1[0] == -1:
        dir_1 = s1[1:4]
        dir_2 = s2[1:4]
        if abs(torch.dot(dir_1, dir_2)) >0.8:
            return True
    
    if s1[0] == 1:
        dir_1 = s1[1:4]
        dir_2 = s2[1:4]
        p1 = s1[4:7]
        p2 = s2[4:7]
        if abs(torch.dot(dir_1, dir_2)) >0.8 and torch.dot(dir_1, p2-p1) < 0.1:
            return True
    
    return False

def ValidConnection(v1, v2, adj):
    for i in v1:
        for j in v2:
            if adj[i, j] == 1:
                return True
    return False

def ValidSameLabel(v1, v2, label):
    for i in v1:
        for j in v2:
            if label[i] == label[j]:
                return True
    return False

def ConnectedParts(idxs, adj):
    n = len(idxs)
    tmp = adj[idxs, :]
    subAdj = tmp[:, idxs]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if subAdj[i, j] == 1 or (subAdj[i, k] == 1 and subAdj[k, j] == 1):
                    subAdj[i, j] = 1
                else:
                    subAdj[i, j] = 0
    
    if torch.sum(subAdj) == n*n:
        return True
    return False

class TestData(object):

    def setSymForAllKids(self, node, symIdx):
        node.symIdx = symIdx
        buffer = [node]
        while len(buffer) != 0:
            n = buffer.pop()
            if n.left is not None:
                n.left.symIdx = symIdx
                buffer.append(n.left)
            if n.right is not None:
                n.right.symIdx = symIdx
                buffer.append(n.right)

    def loadLabel(self, labelA, labelB):
        self.hasLabel = True
        lA = torch.from_numpy(loadmat(root+'/labelA.mat')['labelA']).float()
        lB = torch.from_numpy(loadmat(root+'/labelB.mat')['labelB']).float()

        l = int(lA.size()[1]/30)

        labelAData = torch.chunk(lA,l,1)
        labelBData = torch.chunk(lB,l,1)

        labelAData = [l for l in torch.split(labelAData, 1, 0)]
        labelBData = [l for l in torch.split(labelBData, 1, 0)]
        
        labelDataAll = []
        for l in labelAData:
            if l == -1:
                continue
            labelDataAll.append(l)

        for l in labelBData:
            if l == -1:
                continue
            labelDataAll.append(l)
        
        self.labelDataAll = labelDataAll
        
    def __init__(self, boxA, boxB, symA, symB, adjA, adjB, idxA, idxB, adjgen, isNoise):
        self.hasLabel = False
        self.boxA = torch.t(boxA)
        self.boxB = torch.t(boxB)
        self.symA = torch.t(symA)
        self.symB = torch.t(symB)
        self.adjA = adjA
        self.adjB = adjB
        self.adjGen =  [int(b[0]) for b in torch.split(torch.t(adjgen).squeeze(0), 1, 0)]



        bufferA = [b for b in torch.split(self.boxA, 1, 0)]
        bufferB = [b for b in torch.split(self.boxB, 1, 0)]

        idsA = [b for b in torch.split(idxA, 1, 0)]
        idsB = [b for b in torch.split(idxB, 1, 0)]

        symparaA = [s for s in torch.split(self.symA, 1, 0)]
        symparaB = [s for s in torch.split(self.symB, 1, 0)]
        sympara = []

        for sym in symparaA:
            if torch.sum(sym) == -8:
                continue
            sympara.append(sym)

        for sym in symparaB:
            if torch.sum(sym) == -8:
                continue
            sympara.append(sym)
        
        self.leves = []
        queue = []
        count = -1
        for box in bufferA:
            count = count + 1
            if torch.sum(box) == 0:
                continue
            noise1 = box.new(1, 3).normal_(0, 0.04)
            noise2 = box.new(1, 3).normal_(0, 0.02)
            noise3 = box.new(1, 3).normal_(0, 0.04)
            noise4 = box.new(1, 3).normal_(0, 0.02)
            noise = torch.cat((noise1, noise2, noise3, noise4), 1)
            if isNoise is 1:
                n = Tree.Node(leaf=box+noise, nType=0, sym = symparaA[count].squeeze(0), idx=idsA[count])
            else:
                n = Tree.Node(leaf=box, nType=0, sym = symparaA[count].squeeze(0), idx=idsA[count])
            self.leves.append(n)
            queue.append(n)

        splitNum = len(queue)

        count = -1
        for box in bufferB:
            count = count + 1
            if torch.sum(box) == 0:
                continue
            noise1 = box.new(1, 3).normal_(0, 0.04)
            noise2 = box.new(1, 3).normal_(0, 0.02)
            noise3 = box.new(1, 3).normal_(0, 0.04)
            noise4 = box.new(1, 3).normal_(0, 0.02)
            noise = torch.cat((noise1, noise2, noise3, noise4), 1)
            if isNoise is 1:
                n = Tree.Node(leaf=box+noise, nType=0, sym = symparaB[count].squeeze(0), idx=idsB[count]+10000)
            else:
                n = Tree.Node(leaf=box, nType=0, sym = symparaB[count].squeeze(0), idx=idsB[count]+10000)
            self.leves.append(n)
            queue.append(n)
        
        boxNum = len(queue)
        boxFlag = [-1] * boxNum
        currentFlag = 0
        for i in range(boxNum):
            symsI = sympara[i]
            if boxFlag[i] != -1:
                continue
            else:
                boxFlag[i] = currentFlag
            for j in range(i+1, boxNum):
                symsJ = sympara[j]
                if EqualSymPara(symsI, symsJ):
                    boxFlag[j] = currentFlag
            currentFlag = currentFlag + 1
        
        treeQueue = []
        idxQueue = []
        boxidxQueue = []
        for idx in range(currentFlag):
            subQueue = []
            subIdxQueue = []
            subBoxidxQueue = []
            for i in range(len(queue)):
                if boxFlag[i] == idx:
                    subQueue.append(queue[i])
                    subIdxQueue.append(i)
                    subBoxidxQueue.append(queue[i].idx[0][0])
                    sym = sympara[i]
            idxQueue.append(subIdxQueue)
            boxidxQueue.append(subBoxidxQueue)
            if len(subQueue) < 2:
                if torch.sum(sym) < 50:
                    treeQueue.append(Tree.Node(left=subQueue[0], sym=sym, nType=2))
                    self.setSymForAllKids(subQueue[0], idx)
                    continue
                treeQueue.append(subQueue[0])
                continue
            while len(subQueue) != 1:
                shuffle(subQueue)
                n_left = subQueue.pop()
                n_right = subQueue.pop()
                subQueue.append(Tree.Node(left=n_left, right=n_right, nType=1))
            treeQueue.append(Tree.Node(left=subQueue[0], sym=sym, nType=2))
            self.setSymForAllKids(subQueue[0], idx)

        self.adjAll = torch.ones(boxNum, boxNum)
        self.adjAll[0:splitNum, 0:splitNum] = self.adjA[0:splitNum, 0:splitNum]
        self.adjAll[splitNum:boxNum, splitNum:boxNum] = self.adjB[0:boxNum-splitNum, 0:boxNum-splitNum]

        #self.adjAll[3,4] = 0
        #self.adjAll[4,3] = 0
        
        self.treeQueue = treeQueue
        self.idxQueue = idxQueue
        self.boxidxQueue = boxidxQueue

    def updateByNodes(self, queue):
        sympara = []
        count = 1
        for n in queue:
            n.idx = torch.ones(1,1)*count
            sympara.append(n.sym)
            count = count + 1

        self.leves = queue
        boxNum = len(queue)
        boxFlag = [-1] * boxNum
        currentFlag = 0
        for i in range(boxNum):
            symsI = sympara[i]
            if boxFlag[i] != -1:
                continue
            else:
                boxFlag[i] = currentFlag
            for j in range(i+1, boxNum):
                symsJ = sympara[j]
                if EqualSymPara(symsI, symsJ):
                    boxFlag[j] = currentFlag
            currentFlag = currentFlag + 1
        
        treeQueue = []
        idxQueue = []
        for idx in range(currentFlag):
            subQueue = []
            subIdxQueue = []
            for i in range(len(queue)):
                if boxFlag[i] == idx:
                    subQueue.append(queue[i])
                    subIdxQueue.append(i)
                    sym = sympara[i]
            idxQueue.append(subIdxQueue)
            if len(subQueue) < 2:
                if torch.sum(sym) < 50:
                    treeQueue.append(Tree.Node(left=subQueue[0], sym=sym, nType=2))
                    self.setSymForAllKids(subQueue[0], idx)
                    continue
                treeQueue.append(subQueue[0])
                continue
            while len(subQueue) != 1:
                shuffle(subQueue)
                n_left = subQueue.pop()
                n_right = subQueue.pop()
                subQueue.append(Tree.Node(left=n_left, right=n_right, nType=1))
            treeQueue.append(Tree.Node(left=subQueue[0], sym=sym, nType=2))
            self.setSymForAllKids(subQueue[0], idx)

        self.adjAll = torch.ones(boxNum, boxNum)
        
        self.treeQueue = treeQueue
        self.idxQueue = idxQueue
    
    def randomRemoveOne(self, idx):
        queueLen = len(self.treeQueue)
        self.treeQueue.pop(idx)
        self.idxQueue.pop(idx)
        self.boxidxQueue.pop(idx)

    def sampleBasedonLabel(self):
        treeQueue = copy.deepcopy(self.treeQueue)
        idxQueue = copy.deepcopy(self.idxQueue)
        labels = self.labelDataAll
        maxLabel = max(labels+1)

        while len(treeQueue) != 1:
            if len(treeQueue) == maxLabel:
                break
            idxs = [i for i in range(len(treeQueue))]
            shuffle(idxs)
            idx_left = idxs.pop()
            idx_right = idxs.pop()
            if idx_left < idx_right:
                tmp = idx_left
                idx_left = idx_right
                idx_right = tmp
            flag = ValidSameLabel(idxQueue[idx_left], idxQueue[idx_right], labels)
            if flag is False:
                continue
            n_left = treeQueue.pop(idx_left)
            n_right = treeQueue.pop(idx_right)
            newidx = idxQueue.pop(idx_left) + idxQueue.pop(idx_right)
            treeQueue.append(Tree.Node(left=n_left, right=n_right, nType=1))
            idxQueue.append(newidx)
        
        nodeLabel0 = None
        nodeLabel1 = None
        nodeLabel2 = None
        nodeLabel3 = None
        for i in range(len(treeQueue)):
            if labels[idxQueue[i][0]] == 0:
                nodeLabel0 = treeQueue[i]
                continue
            
            if labels[idxQueue[i][0]] == 1:
                nodeLabel1 = treeQueue[i]
                continue

            if labels[idxQueue[i][0]] == 2:
                nodeLabel2 = treeQueue[i]
                continue

            if labels[idxQueue[i][0]] == 3:
                nodeLabel3 = treeQueue[i]
                continue
        
        if nodeLabel3 is not None:
            rootNodeA = Tree.Node(left=nodeLabel0, right=nodeLabel3, nType=1)
        else:
            rootNodeA = nodeLabel0
        rootNodeB = Tree.Node(left=nodeLabel1, right=nodeLabel2, nType=1)
        
        return Tree.Node(left=rootNodeA, right=rootNodeB, nType=1)
        

    def sampleKsubstructure(self, ksize):
        nodeNum = len(self.treeQueue)
        iters = combinations(list(range(nodeNum)), ksize)
        nodes = []
        for it in iters:
            idxs = []
            for idx in it:
                idxs = idxs + self.idxQueue[idx]
            if ConnectedParts(idxs, self.adjAll) is False:
                continue
            subQueue = []
            subIdxQueue = []
            for i in it:
                subQueue.append(self.treeQueue[i])
                subIdxQueue.append(self.idxQueue[i])
            while len(subQueue) != 1:
                idxs = [i for i in range(len(subQueue))]
                shuffle(idxs)
                idx_left = idxs.pop()
                idx_right = idxs.pop()
                if idx_left < idx_right:
                    tmp = idx_left
                    idx_left = idx_right
                    idx_right = tmp
                flag = ValidConnection(subIdxQueue[idx_left], subIdxQueue[idx_right], self.adjAll)
                if flag is False:
                    continue
                n_left = subQueue.pop(idx_left)
                n_right = subQueue.pop(idx_right)
                newidx = subIdxQueue.pop(idx_left) + subIdxQueue.pop(idx_right)
                subQueue.append(Tree.Node(left=n_left, right=n_right, nType=1))
                subIdxQueue.append(newidx)
            nodes.append(subQueue[0])
        return nodes
    
    def updateBoxInformation(self, boxes):
        for i in range(len(self.leves)):
            idx = self.leves[i].idx[0][0]
            for box in boxes:
                if idx == box.idx[0][0]:
                    self.leves[i].box = box.box
                    self.leves[i].sym = box.sym
                    self.leves[i].tmpBox = box.box
                    self.leves[i].tmpSym = box.sym
        for i in range(len(self.treeQueue)):
            if self.treeQueue[i].symIdx is -1:
                continue
            symIdx = self.treeQueue[i].symIdx
            for box in boxes:
                if box.symIdx == symIdx:
                    self.treeQueue[i].sym = box.sym
                    self.treeQueue[i].tmpSym = box.sym
                    break
            
    def FirstSampleTree(self):
        treeQueue = copy.deepcopy(self.treeQueue)
        idxQueue = copy.deepcopy(self.idxQueue)
        while len(treeQueue) != 1:
            idxs = [i for i in range(len(treeQueue))]
            shuffle(idxs)
            idx_left = idxs.pop()
            idx_right = idxs.pop()
            if idx_left < idx_right:
                tmp = idx_left
                idx_left = idx_right
                idx_right = tmp
            flag = ValidConnection(idxQueue[idx_left], idxQueue[idx_right], self.adjAll)
            if flag is False:
                continue
            n_left = treeQueue.pop(idx_left)
            n_right = treeQueue.pop(idx_right)
            newidx = idxQueue.pop(idx_left) + idxQueue.pop(idx_right)
            treeQueue.append(Tree.Node(left=n_left, right=n_right, nType=1))
            idxQueue.append(newidx)
        
        return treeQueue[0]

    def SampleTree(self):
        #treeQueue = self.treeQueue
        #idxQueue = self.idxQueue
        treeQueue = copy.deepcopy(self.treeQueue)
        idxQueue = copy.deepcopy(self.idxQueue)
        while len(treeQueue) != 1:
            idxs = [i for i in range(len(treeQueue))]
            shuffle(idxs)
            idx_left = idxs.pop()
            idx_right = idxs.pop()
            if idx_left < idx_right:
                tmp = idx_left
                idx_left = idx_right
                idx_right = tmp
            flag = ValidConnection(idxQueue[idx_left], idxQueue[idx_right], self.adjAll)
            if flag is False:
                continue
            n_left = treeQueue.pop(idx_left)
            n_right = treeQueue.pop(idx_right)
            newidx = idxQueue.pop(idx_left) + idxQueue.pop(idx_right)
            treeQueue.append(Tree.Node(left=n_left, right=n_right, nType=1))
            idxQueue.append(newidx)
        
        return treeQueue[0]
    
    def SampleGenTree(self, validLeft):
        leftNode = None
        LeftNodeNum = 3

        while leftNode is None:
            treeQueue = copy.deepcopy(self.treeQueue)
            idxQueue = copy.deepcopy(self.idxQueue)
            
            while len(treeQueue) != 1:
                idxs = [i for i in range(len(treeQueue))]
                shuffle(idxs)
                idx_left = idxs.pop()
                idx_right = idxs.pop()
                if leftNode is None:
                    if idxQueue[idx_left][0] not in validLeft or idxQueue[idx_right][0] not in validLeft:
                        continue
                if idx_left < idx_right:
                    tmp = idx_left
                    idx_left = idx_right
                    idx_right = tmp
                flag = ValidConnection(idxQueue[idx_left], idxQueue[idx_right], self.adjAll)
                if flag is False and len(treeQueue) > 2:
                    continue
                n_left = treeQueue.pop(idx_left)
                n_right = treeQueue.pop(idx_right)
                newidx = idxQueue.pop(idx_left) + idxQueue.pop(idx_right)
                toAdd = Tree.Node(left=n_left, right=n_right, nType=1)
                if toAdd.childNum == LeftNodeNum and leftNode is None:
                    leftNode = toAdd
                    continue
                treeQueue.append(toAdd)
                idxQueue.append(newidx)
        
        return Tree.Node(left=leftNode, right=treeQueue[0], nType=1)

    def SampleAroundInput(self, tree):
        treeQueue = copy.deepcopy(self.treeQueue)
        idxQueue = copy.deepcopy(self.idxQueue)
        boxidxQueue = self.boxidxQueue

        def encode_node(node):
            if node.is_leaf():
                return node, 1e3
            if node.nType == 1:
                left, ll = encode_node(node.left)
                right, lr = encode_node(node.right)
                vqloss = node.vqloss
                if ll < lr:
                    tmp = left
                else:
                    tmp = right
                if vqloss < min(ll, lr) and node.childNum < 4:
                    tmp = node
                    return tmp, min(vqloss, min(ll, lr))
                return tmp, min(ll, lr)
            if node.nType == 2:
                return encode_node(node.left)
        
        def encode_node_idx(node):
            if node.is_leaf():
                return [node.idx[0][0]]
            
            if node.nType == 1:
                ll = encode_node_idx(node.left)
                rr = encode_node_idx(node.right)
                return ll + rr
            
            if node.nType == 2:
                return encode_node_idx(node.left)
        
        minNode, _ = encode_node(tree)
        minIdxs = encode_node_idx(minNode)
        invalid = []
        count = -1
        for idxs in boxidxQueue:
            count = count + 1
            inter = set.intersection(set(idxs), set(minIdxs))
            if len(inter) > 0:
                invalid.append(count)
        
        list.sort(invalid)
        invalid.reverse()
        idxtmp = []

        for idx in invalid:
            treeQueue.pop(idx)
            idxtmp = idxtmp + idxQueue.pop(idx)
        
        treeQueue.append(minNode)
        idxQueue.append(idxtmp)

        while len(treeQueue) != 1:
            idxs = [i for i in range(len(treeQueue))]
            shuffle(idxs)
            idx_left = idxs.pop()
            idx_right = idxs.pop()
            if idx_left < idx_right:
                tmp = idx_left
                idx_left = idx_right
                idx_right = tmp
            flag = ValidConnection(idxQueue[idx_left], idxQueue[idx_right], self.adjAll)
            if flag is False:
                continue
            n_left = treeQueue.pop(idx_left)
            n_right = treeQueue.pop(idx_right)
            newidx = idxQueue.pop(idx_left) + idxQueue.pop(idx_right)
            treeQueue.append(Tree.Node(left=n_left, right=n_right, nType=1))
            idxQueue.append(newidx)
        
        return treeQueue[0]

class SCORESTest(data.Dataset):
    def __init__(self, root):
        self.root = root
        
        bA = torch.from_numpy(loadmat(root+'/boxesA.mat')['boxesA']).float()
        bB = torch.from_numpy(loadmat(root+'/boxesB.mat')['boxesB']).float()
        sA = torch.from_numpy(loadmat(root+'/symsA.mat')['symsA']).float()
        sB = torch.from_numpy(loadmat(root+'/symsB.mat')['symsB']).float()
        adjA = torch.from_numpy(loadmat(root+'/adjA.mat')['adjA']).float()
        adjB = torch.from_numpy(loadmat(root+'/adjB.mat')['adjB']).float()
        idxA = torch.from_numpy(loadmat(root+'/idxA.mat')['idxA']).float()
        idxB = torch.from_numpy(loadmat(root+'/idxB.mat')['idxB']).float()
        adjGen = torch.from_numpy(loadmat(root+'/adjGen.mat')['adjGen']).float()

        l = int(bA.size()[1]/30)

        self.boxAData = torch.chunk(bA,l,1)
        self.boxBData = torch.chunk(bB,l,1)
        self.symAData = torch.chunk(sA,l,1)
        self.symBData = torch.chunk(sB,l,1)
        self.adjAData = torch.chunk(adjA,l,1)
        self.adjBData = torch.chunk(adjB,l,1)
        self.idxAData = torch.chunk(idxA,l,1)
        self.idxBData = torch.chunk(idxB,l,1)
        self.adjGenData = torch.chunk(adjGen,l,1)

    def __getitem__(self, index):
        objRoot = self.root + '/' + str(index+1)
        if index == 0:
            data = TestData(self.boxAData[index], self.boxBData[index], self.symAData[index], self.symBData[index], self.adjAData[index], self.adjBData[index], self.idxAData[index], self.idxBData[index], self.adjGenData[index], 1)
        else:
            data = TestData(self.boxAData[index], self.boxBData[index], self.symAData[index], self.symBData[index], self.adjAData[index], self.adjBData[index], self.idxAData[index], self.idxBData[index], self.adjGenData[index], 0)
        return data

    def __len__(self):
        return len(self.boxAData)