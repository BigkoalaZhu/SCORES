import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from time import time
from dataset import Tree
from random import randint
from random import shuffle

#########################################################################################
# Merge testing operation
#########################################################################################

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

def repair_acc(tree):
    def updateTree(node):
        if node.is_leaf():
            node.tmpBox = node.tmpAccBox/node.accNum
            node.tmpSym = node.tmpAccSym/node.accNum
            node.tmpAccBox = node.tmpBox
            node.tmpAccSym = node.tmpSym
            node.accNum = 1
            return 
        if node.nType == 1:
            updateTree(node.left)
            updateTree(node.right)
            return
        if node.nType == 2:
            updateTree(node.left)
            node.tmpSym = node.tmpAccSym/node.accNum
            node.tmpAccSym = node.tmpSym
            node.accNum = 1
            return
    updateTree(tree)

def reset_acc(tree):
    def updateTree(node):
        if node.is_leaf():
            node.tmpBox = node.box
            node.tmpSym = node.sym
            return 
        if node.nType == 1:
            updateTree(node.left)
            updateTree(node.right)
            return
        if node.nType == 2:
            updateTree(node.left)
            node.tmpSym = node.sym
            return
    updateTree(tree)

def reverse_one_step(tree):
    def updateTree(node):
        if node.is_leaf():
            node.tmpAccSym = node.tmpAccSym - node.symCache*node.accCache
            node.tmpAccBox = node.tmpAccBox - node.boxCache*node.accCache
            node.accNum = node.accNum - node.accCache
            return 
        if node.nType == 1:
            updateTree(node.left)
            updateTree(node.right)
            return
        if node.nType == 2:
            updateTree(node.left)
            node.tmpAccSym = node.tmpAccSym - node.symCache*node.accCache
            node.accNum = node.accNum - node.accCache
            return
    updateTree(tree)

def decode_structure_acc(model, tree):

    def encode_node(node):
        if node.is_leaf():
            return model.leafNodeTest(Variable(node.box))
        if node.nType == 1:
            left = encode_node(node.left)
            right = encode_node(node.right)
            return model.adjNodeTest(left, right)
        if node.nType == 2:
            feature = encode_node(node.left)
            sym = Variable(node.sym)
            return model.symNodeTest(feature, sym)

    weight = 1
    feature = encode_node(tree)
    decode = model.decoder.desampler(feature)
    vqLoss, decode = model.vqlizationWithLoss(decode)
    vqLoss = vqLoss.sum()
    vq = []
    vq.append(vqLoss)
    syms = [Variable(torch.ones(8).mul(10))]
    stack = [decode]
    boxes = []
    nodes = [tree]

    nodeQueue = []
    n = Tree.Node()
    NodeStack = [n]

    while len(stack) > 0:
        n = nodes.pop()
        f = stack.pop()
        if n.is_leaf() == False:
            d = model.decoder.desampler(f)
            vqLoss, _ = model.vqlizationWithLoss(d)
            _, f = model.vqlizationWithLoss(f)
        label = n.label
        currentNode = NodeStack.pop()
        if label[0][1] == 1:
            vq.append(vqLoss.sum())
            left_inner = encode_node(n.left)
            right_inner = encode_node(n.right)
            left_outter = model.outterNode(f, right_inner)
            right_outter = model.outterNode(f, left_inner)

            stack.append(left_outter)
            stack.append(right_outter)
            nodes.append(n.left)
            nodes.append(n.right)
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

        if label[0][2] == 1:
            inner = encode_node(n)
            s = model.symParaNode(f, inner)
            s = s.squeeze(0)
            n.tmpAccSym = n.tmpAccSym + s.data*weight
            n.accNum = n.accNum + weight
            n.accCache = weight
            n.symCache = s.data
            stack.append(f)
            nodes.append(n.left)
            syms.pop()
            syms.append(s)

            leftNode = Tree.Node()
            currentNode.left = leftNode
            currentNode.sym = s.data
            currentNode.tmpSym = s.data
            currentNode.nType = 2
            NodeStack.append(leftNode)
            nodeQueue.append(currentNode)

        if label[0][0] == 1:
            box_inner = encode_node(n)
            reBox = model.boxNode(f, box_inner)
            s = syms.pop()
            node = Tree.Node(leaf=reBox.data, nType=0,
                             sym=s.data, idx=n.idx, symIdx=n.symIdx)
            boxes.append(node)
            n.tmpAccBox = n.tmpAccBox + reBox.data*weight
            n.tmpAccSym = n.tmpAccSym + s.data*weight
            n.accNum = n.accNum + weight
            n.accCache = weight
            n.symCache = s.data
            n.boxCache = reBox.data

            currentNode.box = reBox.data
            currentNode.tmpBox = reBox.data
            currentNode.sym = s.data
            currentNode.tmpSym = s.data
            currentNode.nType = 0
            nodeQueue.append(currentNode)
    loss = decode_Vq_loss(model, nodeQueue[0]).data[0]
    return boxes, loss

def decode_Vq_loss(model, tree):

    def encode_node(node, loss):
        if node.is_leaf():
            return model.leafNodeTest(Variable(node.tmpBox))
        if node.nType == 1:
            left = encode_node(node.left, loss)
            right = encode_node(node.right, loss)
            f = model.adjNodeTest(left, right)
            d = model.decoder.desampler(f)
            l, _ = model.vqlizationWithLoss2(d)
            #l = l/node.childNum
            #node.vqloss = l.data[0]
            loss.append(l)
            return f
        if node.nType == 2:
            feature = encode_node(node.left, loss)
            sym = Variable(node.tmpSym)
            f = model.symNodeTest(feature, sym)
            d = model.decoder.desampler(f)
            l, _ = model.vqlizationWithLoss2(d)
            #l = l/node.childNum
            #node.vqloss = l.data[0]
            # loss.append(l)
            return f

    lossList = []
    encode_node(tree, lossList)
    loss = torch.cat(lossList, 0)
    loss = loss.sum()/len(lossList)

    return loss

def render_node_to_boxes(nodes):
    boxes = []
    idxs = []
    for n in nodes:
        if n.sym is None:
            boxes.append(n.box)
        else:
            reBox = n.box
            reBoxes = [reBox]

            bid = n.idx
            subIds = [bid]

            s = n.sym
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
                    subIds.append(bid)

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
                    subIds.append(n.idx)

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
                subIds.append(n.idx)

            boxes.extend(reBoxes)

    return boxes

def oneIterMerge(model, testdata):
    patches = testdata.sampleKsubstructure(4)
    vqSum = 0
    count = 0
    for tree in patches:
        vq = decode_Vq_loss(model, tree).data[0]
        _, after = decode_structure_acc(model, tree)
        if after < vq:
            vqSum = vqSum + vq
            count = count + 1
        else:
            reverse_one_step(tree)
    if count == 0:
        return testdata.leves
    vqBefore = vqSum/count
    vqSum = 0
    for tree in patches:
        repair_acc(tree)
        vq = decode_Vq_loss(model, tree).data[0]
        vqSum = vqSum + vq
    vqLoss = vqSum/len(patches)
    if vqLoss < vqBefore:
        for tree in patches:
            updateTreeInfo(tree)
        return testdata.leves
    else:
        for tree in patches:
            reset_acc(tree)
        return testdata.leves


def iterateKMergeTest(model, testdata):
    inputBox = [render_node_to_boxes(testdata.leves)]
    for i in range(16):
        boxes= oneIterMerge(model, testdata)
        if i % 5 == 0:
            box_all = render_node_to_boxes(boxes)
            inputBox.append(box_all)

    return inputBox