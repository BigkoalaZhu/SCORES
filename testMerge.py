from dataset import SCORESTest
import torch.utils.data
import testVQContextGen
from draw3dOBB import showGenshape
import testVQContext

def class_collate(batch):
    return batch

mergeNet = torch.load('MergeNet_chair_demo_gen.pkl')
mergeNet.cpu()

mergeNetFix = torch.load('MergeNet_chair_demo_fix.pkl')
mergeNetFix = mergeNetFix.cpu()

allTestData = SCORESTest('test')
dataloader = torch.utils.data.DataLoader(allTestData, batch_size=1, shuffle=False, collate_fn=class_collate)
for i, batch in enumerate(dataloader):

    testFile = batch[0]
    
    remove = testFile.adjGen.pop(0)

    if remove != -1:
        testFile.randomRemoveOne(remove)
        originalNodes = testFile.leves
        originalNodes.pop(remove)
        boxes = testVQContextGen.render_node_to_boxes(originalNodes)
        showGenshape(torch.cat(boxes,0).numpy())
    
        nodes, newtree = testVQContextGen.MergeTest(mergeNet, testFile, testFile.adjGen)
        boxes = testVQContextGen.render_node_to_boxes(nodes)
        showGenshape(torch.cat(boxes,0).numpy())
        testFile.updateByNodes(nodes)
    else:
        originalNodes = testFile.leves
        boxes = testVQContextGen.render_node_to_boxes(originalNodes)
        showGenshape(torch.cat(boxes,0).numpy())
       
    
    allBoxes = testVQContext.iterateKMergeTest(mergeNetFix, testFile)
    boxes = allBoxes.pop()  
    showGenshape(torch.cat(boxes,0).numpy())
    

