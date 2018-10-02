import numpy as np
from numpy import linalg as LA

def abs2rel(pos, center, dir1, dir2, dir3, scale):
    tmp = pos - center
    proj1 = tmp.dot(dir1)
    v1 = proj1/scale[0]
    proj2 = tmp.dot(dir2)
    v2 = proj2/scale[1]
    proj3 = tmp.dot(dir3)
    v3 = proj3/scale[2]

    return [v1, v2, v3]

def rel2abs(pos, center, dir1, dir2, dir3, scale):
    new = scale[0]*pos[0]*dir1 + scale[1]*pos[1]*dir2 + scale[2]*pos[2]*dir3
    new = new + center
    return [new[0], new[1], new[2]]

class ShapeGeometry(object):
    class Component(object):
        def __init__(self, vertices, faces):
            self.vertices = vertices
            self.faces = faces

    def __init__(self, fileNameA, fileNameB):
        file = open(fileNameA)
        self.components = []
        currentVertices = []
        currentFaces = []
        comId = -1
        vnum = 0

        for line in file:
            lids = line.split( )
            if len(lids) > 3:
                fileType = 1
            else:
                fileType = 0
            break

        if fileType == 0:
            for line in file:
                l = line
                lids = l.split( )
                if len(lids) != 0:
                    lid = lids[0]
                else:
                    lid = ' '
                if lid == 'g':
                    if comId == -1:
                        comId = comId + 1
                        continue
                    vnum = vnum + len(currentVertices)
                    com = ShapeGeometry.Component(currentVertices, currentFaces)
                    self.components.append(com)
                    currentVertices = []
                    currentFaces = []
                    

                if lid == 'v':
                    vids = l.split( )
                    currentVertices.append([float(vids[1])/5, float(vids[2])/5, float(vids[3])/5])
                
                if lid == 'f':
                    fids = l.split( )
                    if fids[1].find('//') == -1:
                        currentFaces.append([int(fids[1]) - vnum, int(fids[2]) - vnum, int(fids[3]) - vnum])
                    else:
                        f1 = fids[1].split('//')[0]
                        f2 = fids[2].split('//')[0]
                        f3 = fids[3].split('//')[0]
                        currentFaces.append([int(f1) - vnum, int(f2) - vnum, int(f3) - vnum])
        
            com = ShapeGeometry.Component(currentVertices, currentFaces)
            self.components.append(com)

        if fileType == 1:
            flag = 1
            for line in file:
                l = line
                lids = l.split( )
                if len(lids) != 0:
                    lid = lids[0]
                else:
                    lid = ' '               

                if lid == 'v':
                    if flag == 0:
                        vnum = vnum + len(currentVertices)
                        com = ShapeGeometry.Component(currentVertices, currentFaces)
                        self.components.append(com)
                        currentVertices = []
                        currentFaces = []
                    flag = 1
                    vids = l.split( )
                    currentVertices.append([float(vids[1])/5, float(vids[2])/5, float(vids[3])/5])
                
                if lid == 'f':
                    flag = 0
                    fids = l.split( )
                    if fids[1].find('//') == -1:
                        currentFaces.append([int(fids[1]) - vnum, int(fids[2]) - vnum, int(fids[3]) - vnum])
                    else:
                        f1 = fids[1].split('//')[0]
                        f2 = fids[2].split('//')[0]
                        f3 = fids[3].split('//')[0]
                        currentFaces.append([int(f1) - vnum, int(f2) - vnum, int(f3) - vnum])
        
            com = ShapeGeometry.Component(currentVertices, currentFaces)
            self.components.append(com)
        file.close()

        self.comSplit = len(self.components)

        file = open(fileNameB)
        currentVertices = []
        currentFaces = []
        comId = -1
        vnum = 0
        for line in file:
            lids = line.split( )
            if len(lids) > 3:
                fileType = 1
            else:
                fileType = 0
            break

        if fileType == 0:
            for line in file:
                l = line
                lids = l.split( )
                if len(lids) != 0:
                    lid = lids[0]
                else:
                    lid = ' '
                if lid == 'g':
                    if comId == -1:
                        comId = comId + 1
                        continue
                    vnum = vnum + len(currentVertices)
                    com = ShapeGeometry.Component(currentVertices, currentFaces)
                    self.components.append(com)
                    currentVertices = []
                    currentFaces = []
                    

                if lid == 'v':
                    vids = l.split( )
                    currentVertices.append([float(vids[1])/5, float(vids[2])/5, float(vids[3])/5])
                
                if lid == 'f':
                    fids = l.split( )
                    if fids[1].find('//') == -1:
                        currentFaces.append([int(fids[1]) - vnum, int(fids[2]) - vnum, int(fids[3]) - vnum])
                    else:
                        f1 = fids[1].split('//')[0]
                        f2 = fids[2].split('//')[0]
                        f3 = fids[3].split('//')[0]
                        currentFaces.append([int(f1) - vnum, int(f2) - vnum, int(f3) - vnum])
        
            com = ShapeGeometry.Component(currentVertices, currentFaces)
            self.components.append(com)

        if fileType == 1:
            flag = 1
            for line in file:
                l = line
                lids = l.split( )
                if len(lids) != 0:
                    lid = lids[0]
                else:
                    lid = ' '               

                if lid == 'v':
                    if flag == 0:
                        vnum = vnum + len(currentVertices)
                        com = ShapeGeometry.Component(currentVertices, currentFaces)
                        self.components.append(com)
                        currentVertices = []
                        currentFaces = []
                    flag = 1
                    vids = l.split( )
                    currentVertices.append([float(vids[1])/5, float(vids[2])/5, float(vids[3])/5])
                
                if lid == 'f':
                    flag = 0
                    fids = l.split( )
                    if fids[1].find('//') == -1:
                        currentFaces.append([int(fids[1]) - vnum, int(fids[2]) - vnum, int(fids[3]) - vnum])
                    else:
                        f1 = fids[1].split('//')[0]
                        f2 = fids[2].split('//')[0]
                        f3 = fids[3].split('//')[0]
                        currentFaces.append([int(f1) - vnum, int(f2) - vnum, int(f3) - vnum])
        
            com = ShapeGeometry.Component(currentVertices, currentFaces)
            self.components.append(com)
        file.close()

        self.outputComponents = []

    def outPutShapeOld(self, location):
        fo = open(location, "w")
        fo.write("#Generated shape Obj \n")
        count = 0
        vcount = 1
        for cc in self.outputComponents:
            start = min(min(cc.faces)) - vcount
            end = max(max(cc.faces))
            vn = len(cc.vertices)
            lll = end - min(min(cc.faces))
            #if vn != 315:
            #    continue
            fo.write("g %s\n"%(str(count)))
            count = count + 1
            for v in cc.vertices:
                fo.write("v %s %s %s\n"%(str(v[0]*5), str(v[1]*5), str(v[2]*5)))
            for f in cc.faces:
                fo.write("f %s %s %s\n"%(str(f[0] - start), str(f[1] - start), str(f[2] - start)))
            vcount = vcount + len(cc.vertices)
        fo.write("#Finished! \n")
        fo.close()
        self.outputComponents = []

    def outPutShape(self, location):
        fo = open(location, "w")
        fo.write("#Generated shape Obj \n")
        count = 0
        vcount = 0
        for cc in self.outputComponents:
            fo.write("g %s\n"%(str(count)))
            count = count + 1
            for v in cc.vertices:
                fo.write("v %s %s %s\n"%(str(v[0]*5), str(v[1]*5), str(v[2]*5)))
            for f in cc.faces:
                fo.write("f %s %s %s\n"%(str(f[0] + vcount), str(f[1] + vcount), str(f[2] + vcount)))
            vcount = vcount + len(cc.vertices)
        fo.write("#Finished! \n")
        fo.close()
        self.outputComponents = []

    def DeformedComponent(self, idx, original, deformed):
        if idx > 10000:
            idx = idx - 10000 + self.comSplit
        
        idx = idx - 1
        currentVertices = []
        currentFaces = self.components[idx].faces

        ocenter = original[0: 3]
        olengths = original[3: 6]
        odir_1 = original[6: 9]
        odir_2 = original[9: ]

        odir_1 = odir_1/LA.norm(odir_1)
        odir_2 = odir_2/LA.norm(odir_2)
        odir_3 = np.cross(odir_1, odir_2)
        odir_3 = odir_3/LA.norm(odir_3)

        dcenter = deformed[0: 3]
        dlengths = deformed[3: 6]
        ddir_1 = deformed[6: 9]
        ddir_2 = deformed[9: ]

        ddir_1 = ddir_1/LA.norm(ddir_1)
        ddir_2 = ddir_2/LA.norm(ddir_2)
        ddir_3 = np.cross(ddir_1, ddir_2)
        ddir_3 = ddir_3/LA.norm(ddir_3)

        if odir_3.dot(ddir_3) < 0:
            ddir_3 = -ddir_3

        for i in range(len(self.components[idx].vertices)):
            rel = abs2rel(self.components[idx].vertices[i], ocenter, odir_1, odir_2, odir_3, olengths)
            newpos = rel2abs(rel, dcenter, ddir_1, ddir_2, ddir_3, dlengths)
            currentVertices.append(newpos)
            #currentVertices.append(self.components[idx].vertices[i])
        
        self.outputComponents.append(ShapeGeometry.Component(currentVertices, currentFaces))