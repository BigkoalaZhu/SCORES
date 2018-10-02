import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from time import time
from dataset import Tree
from random import randint
from random import shuffle
from draw3dOBB import showGenshape
import itertools

#############################################################################
# Utility
#############################################################################

def vrrotvec2mat_cpu(rotvector):
    s = math.sin(rotvector[3])
    c = math.cos(rotvector[3])
    t = 1 - c
    x = rotvector[0]
    y = rotvector[1]
    z = rotvector[2]
    m = torch.FloatTensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x *
                                                             y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]])
    return m


def vrrotvec2mat(rotvector):
    s = math.sin(rotvector[3])
    c = math.cos(rotvector[3])
    t = 1 - c
    x = rotvector[0]
    y = rotvector[1]
    z = rotvector[2]
    m = torch.FloatTensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z,
                                                             t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]]).cuda()
    return m

def updateTreeInfo(tree):

    def updateTree(node):
        if node.is_leaf():
            node.box = node.tmpBox
            node.sym = node.tmpSym
            return 
        if node.nType == 1:
            updateTree(node.left)
            updateTree(node.right)
            return
        if node.nType == 2:
            updateTree(node.left)
            node.sym = node.tmpSym
            return
    
    updateTree(tree)

def decode_structure(model, tree):

    def encode_node(node):
        if node.is_leaf():
            return model.leafNodeTest(Variable(node.box))
        if node.nType == 1:
            if node.left.valid != True:
                return encode_node(node.right)

            if node.right.valid != True:
                return encode_node(node.left)
            left = encode_node(node.left)
            right = encode_node(node.right)
            return model.adjNodeTest(left, right)
        if node.nType == 2:
            feature = encode_node(node.left)
            sym = Variable(node.sym)
            return model.symNodeTest(feature, sym)
    
    def addExistesBoxes(node, boxes):
        if node.is_leaf():
            if node.sym is None:
                n = Tree.Node(leaf=node.box, nType=0)
            else:
                n = Tree.Node(leaf=node.box, nType=0, sym=node.sym.squeeze(0))
            boxes.append(n)
            return 
        if node.nType == 1:
            addExistesBoxes(node.left, boxes)
            addExistesBoxes(node.right, boxes)
            return
        if node.nType == 2:
            addExistesBoxes(node.left, boxes)
            return
    
    featureAll = encode_node(tree)
    featureLeft = encode_node(tree.left)
    featureRight = encode_node(tree.right)
    leftOutter = model.outterNode(featureAll, featureRight)
    leftOutterVQ = model.vqlizationWithOutLossGen(leftOutter)
    decode = model.concat(leftOutterVQ, featureLeft)

    syms = [Variable(torch.ones(8).mul(10))]
    stack = [decode]
    boxes = []
    nodeQueue = []
    n = Tree.Node()
    NodeStack = [n]

    while len(stack) > 0:
        f = stack.pop()
        label = model.IncompleteClassiferNode(f)
        l = label.data[0]
        currentNode = NodeStack.pop()
        if l == 1:
            left, right = model.IncompleteAdjNode(f)

            stack.append(left)
            stack.append(right)
            s = syms.pop()
            syms.append(s)
            syms.append(s)
            
            leftNode = Tree.Node()
            rightNode = Tree.Node()
            currentNode.left = leftNode
            currentNode.right = rightNode
            currentNode.nType = 1
            nodeQueue.append(currentNode)

            NodeStack.append(leftNode)
            NodeStack.append(rightNode)

        if l == 2:
            left, s = model.IncompleteSymNode(f)
            s = s.squeeze(0)

            stack.append(left)
            syms.pop()
            syms.append(s)

            leftNode = Tree.Node()
            currentNode.left = leftNode
            currentNode.sym = s.data
            currentNode.tmpSym = s.data
            currentNode.nType = 2
            NodeStack.append(leftNode)
            nodeQueue.append(currentNode)

        if l == 0:
            reBox = model.IncompleteBoxNode(f)
            s = syms.pop()
            node = Tree.Node(leaf=reBox.data, nType=0,
                             sym=s.data)
            boxes.append(node)

            currentNode.box = reBox.data
            currentNode.tmpBox = reBox.data
            currentNode.sym = s.data
            currentNode.tmpSym = s.data
            currentNode.nType = 0
            nodeQueue.append(currentNode)

    addExistesBoxes(tree.right, boxes)

    return boxes, nodeQueue[0]

def decode_Vq_loss(model, treeL, treeR):

    def encode_node(node):
        if node.is_leaf():
            return model.leafNodeTest(Variable(node.tmpBox))
        if node.nType == 1:

            if node.left.valid != True:
                return encode_node(node.right)

            if node.right.valid != True:
                return encode_node(node.left)

            left = encode_node(node.left)
            right = encode_node(node.right)
            f = model.adjNodeTest(left, right)
            return f
        if node.nType == 2:
            feature = encode_node(node.left)
            sym = Variable(node.tmpSym)
            f = model.symNodeTest(feature, sym)
            return f

    fL = encode_node(treeL)
    fR = encode_node(treeR)
    fRoot = model.adjNodeTest(fL, fR)
    f = model.outterNode(fRoot, fR)
    loss = model.vqlizationWithLossGen(f)

    return loss

#################################################################################
# Shape geometry related
#################################################################################

def getLeafBoxes(tree):
    
    def get_leaf_node(node, boxes):
        if node.is_leaf():
            boxes.append(node)
            return
        if node.nType == 1:
            get_leaf_node(node.left, boxes)
            get_leaf_node(node.right, boxes)
            return
        if node.nType == 2:
            get_leaf_node(node.left, boxes)
            return
    
    boxes = []
    get_leaf_node(tree, boxes)
    return boxes

def generateTranformMatrix(objShape, original, deformed):
    for b1 in deformed:
        for b2 in original:
            if b1.idx[0][0] != b2.idx[0][0]:
                continue
            reBox = b1.box
            ori = b2.original_box

            s = b1.sym
            objShape.DeformedComponent(int(b1.idx[0][0]), ori.squeeze(
                0).numpy(), reBox.squeeze(0).numpy())
            l1 = abs(s[0] + 1)
            l2 = abs(s[0])
            l3 = abs(s[0] - 1)

            if l1 < 0.16:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.squeeze(0), 1, 0)
                f1 = torch.cat([sList[1], sList[2], sList[3]])
                f1 = f1/torch.norm(f1)
                f2 = torch.cat([sList[4], sList[5], sList[6]])
                folds = round(1/s[7])
                for i in range(int(folds-1)):
                    rotvector = torch.cat(
                        [f1, sList[7].mul(2*3.1415).mul(i+1)])
                    rotm = vrrotvec2mat_cpu(rotvector)
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = rotm.matmul(center.add(-f2)).add(f2)
                    newdir1 = rotm.matmul(dir1)
                    newdir2 = rotm.matmul(dir2)
                    newbox = torch.cat([newcenter, dir0, newdir1, newdir2])
                    objShape.DeformedComponent(
                        int(b1.idx[0][0]), ori.squeeze(0).numpy(), newbox.numpy())

            if l2 < 0.16:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.squeeze(0), 1, 0)
                trans = torch.cat([sList[1], sList[2], sList[3]])
                trans_end = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                trans_length = math.sqrt(torch.sum(trans**2))
                trans_total = math.sqrt(torch.sum(trans_end.add(-center)**2))
                folds = round(trans_total/trans_length)
                for i in range(int(folds)):
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = center.add(trans.mul(i+1))
                    newbox = torch.cat([newcenter, dir0, dir1, dir2])
                    objShape.DeformedComponent(
                        int(b1.idx[0][0]), ori.squeeze(0).numpy(), newbox.numpy())

            if l3 < 0.16:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.squeeze(0), 1, 0)
                ref_normal = torch.cat([sList[1], sList[2], sList[3]])
                ref_normal = ref_normal/torch.norm(ref_normal)
                ref_point = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                dir0 = torch.cat([bList[3], bList[4], bList[5]])
                dir1 = torch.cat([bList[6], bList[7], bList[8]])
                dir2 = torch.cat([bList[9], bList[10], bList[11]])
                if ref_normal.matmul(ref_point.add(-center)) < 0:
                    ref_normal = -ref_normal
                newcenter = ref_normal.mul(
                    2*abs(torch.sum(ref_point.add(-center).mul(ref_normal)))).add(center)
                if ref_normal.matmul(dir1) < 0:
                    ref_normal = -ref_normal
                dir1 = dir1.add(ref_normal.mul(-2*ref_normal.matmul(dir1)))
                if ref_normal.matmul(dir2) < 0:
                    ref_normal = -ref_normal
                dir2 = dir2.add(ref_normal.mul(-2*ref_normal.matmul(dir2)))
                newbox = torch.cat([newcenter, dir0, dir1, dir2])
                objShape.DeformedComponent(
                    int(b1.idx[0][0]), ori.squeeze(0).numpy(), newbox.numpy())


def render_node_to_boxes(nodes):
    boxes = []
    for n in nodes:
        if n.sym is None:
            boxes.append(n.box)
        else:
            reBox = n.box
            reBoxes = [reBox]

            s = n.sym#.squeeze(0)
            l1 = abs(s[0] + 1)
            l2 = abs(s[0])
            l3 = abs(s[0] - 1)

            if l1 < 0.16:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.squeeze(0), 1, 0)
                f1 = torch.cat([sList[1], sList[2], sList[3]])
                f1 = f1/torch.norm(f1)
                f2 = torch.cat([sList[4], sList[5], sList[6]])
                folds = round(1/s[7].data.item())
                for i in range(int(folds-1)):
                    rotvector = torch.cat(
                        [f1, sList[7].mul(2*3.1415).mul(i+1)])
                    rotm = vrrotvec2mat_cpu(rotvector)
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = rotm.matmul(center.add(-f2)).add(f2)
                    newdir1 = rotm.matmul(dir1)
                    newdir2 = rotm.matmul(dir2)
                    newbox = torch.cat([newcenter, dir0, newdir1, newdir2])
                    reBoxes.append(newbox.unsqueeze(0))

            if l2 < 0.16:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.squeeze(0), 1, 0)
                trans = torch.cat([sList[1], sList[2], sList[3]])
                trans_end = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                trans_length = math.sqrt(torch.sum(trans**2))
                trans_total = math.sqrt(torch.sum(trans_end.add(-center)**2))
                folds = round(trans_total/trans_length)
                for i in range(int(folds)):
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = center.add(trans.mul(i+1))
                    newbox = torch.cat([newcenter, dir0, dir1, dir2])
                    reBoxes.append(newbox.unsqueeze(0))

            if l3 < 0.16:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.squeeze(0), 1, 0)
                ref_normal = torch.cat([sList[1], sList[2], sList[3]])
                ref_normal = ref_normal/torch.norm(ref_normal)
                ref_point = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                dir0 = torch.cat([bList[3], bList[4], bList[5]])
                dir1 = torch.cat([bList[6], bList[7], bList[8]])
                dir2 = torch.cat([bList[9], bList[10], bList[11]])
                if ref_normal.matmul(ref_point.add(-center)) < 0:
                    ref_normal = -ref_normal
                newcenter = ref_normal.mul(
                    2*abs(torch.sum(ref_point.add(-center).mul(ref_normal)))).add(center)
                if ref_normal.matmul(dir1) < 0:
                    ref_normal = -ref_normal
                dir1 = dir1.add(ref_normal.mul(-2*ref_normal.matmul(dir1)))
                if ref_normal.matmul(dir2) < 0:
                    ref_normal = -ref_normal
                dir2 = dir2.add(ref_normal.mul(-2*ref_normal.matmul(dir2)))
                newbox = torch.cat([newcenter, dir0, dir1, dir2])
                reBoxes.append(newbox.unsqueeze(0))

            boxes.extend(reBoxes)

    return boxes

def errFromGenParts(before, after):
    possibleOrder = list(itertools.permutations(range(len(after))))
    errList = []
    for i in range(len(possibleOrder)):
        err = 0
        for j in range(len(before)):
            err = err + ((before[j].box.add(-after[possibleOrder[i][j]].box))**2).sum(1)
            err = err + ((before[j].sym.add(-after[possibleOrder[i][j]].sym))**2).sum(0)
        errList.append(err[0])
    return min(errList)

#################################################################################
# Merge processing
#################################################################################

def MergeTest(model, testdata, adj):

    ########################################################
    sampleNum = 20
    trees = []
    vqLoss = []
    for i in range(sampleNum):
        trees.append(testdata.SampleGenTree(adj))
    for i in range(sampleNum):
        leftNodes = getLeafBoxes(trees[i].left)
        boxes, _ = decode_structure(model, trees[i])
        err = errFromGenParts(leftNodes, boxes)
        vqLoss.append(err)
    idx = vqLoss.index(min(vqLoss))
    ruleTree = trees[idx]
    
    boxes, tree = decode_structure(model, ruleTree)
    return boxes, tree
    
    





