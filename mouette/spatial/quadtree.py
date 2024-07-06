from ..mesh.datatypes import *
from ..mesh.mesh_data import DataContainer, RawMeshData
from ..geometry import Vec, AABB
import numpy as np

from dataclasses import dataclass
from enum import Enum

# class SpacePartitionTree:
    
#     class Node:
#         def __init__(self, bounding_box: AABB, depth:int = 0):
#             self.BB : AABB = bounding_box
#             self.depth = depth
#             self.id : int = None
#             self.children : list = []
#             self.parent : int = None
        
#         @property
#         def parent(self):
#             if self.id==1: return None
#             return self.id//2

#         @property
#         def children(self):
#             return 2*self.id, 2*self.id+1
        
#         def subdivide(self):
#             return

#     def __init__(self, dim:int, points: DataContainer, domain : AABB = None):
#         self.points = points
#         self.dim = dim
#         self._id_node = 0
#         self._nodes = [None]
#         bb = self.get_bounding_box() if domain is None else domain
#         _ = self.create_node(bb, 0) # root
#         self._where = points.create_attribute("node_id", int, dense=True)

#     @property
#     def root(self):
#         return self._nodes[1]

#     def create_node(self, bb:AABB, depth:int):
#         new_node = SpacePartitionTree.Node(bb, depth=depth)
#         new_node.id = self._id_node
#         self._id_node += 1
#         self._nodes.append(new_node)
#         return self._id_node

#     def get_bounding_box(self, padding:float=0.):
#         xmin,ymin,zmin = np.min(self.points._data, axis=0)
#         xmax,ymax,zmax = np.max(self.points._data, axis=0)
#         if self.dim==2:
#             pmin,pmax,pdim = Vec(xmin,ymin), Vec(xmax,ymax), Vec(padding,padding)
#         else:
#             pmin,pmax,pdim = Vec(xmin,ymin,zmin), Vec(xmax,ymax,zmax), Vec(padding,padding,padding)
#         return AABB(pmin,pmax-pmin+pdim)

#     def append(pt:Vec):
#         return

class QuadTree:
    
    MAX_PT_PER_NODE = 1

    @dataclass
    class Node:
        id : int # id of the node
        depth : int # depth of the node inside the tree
        parent : int  # id of parent node (-1 if root node)
        corners : list # indices of vertex corners of the node (should be 4 elements)
        children : list = None # id of children nodes
        points: set = None # id of contained points

    def __init__(self, domain : AABB):
        self.points = DataContainer(id="vertices")
        self._where = self.points.create_attribute("id", int)

        ## Build a treemesh as a rect
        self._tree_vert : PointCloud = PointCloud()
        v0 = Vec(domain.left, domain.bottom, 0.)
        v1 = Vec(domain.right, domain.bottom, 0.)
        v2 = Vec(domain.right, domain.top, 0.)
        v3 = Vec(domain.left, domain.top, 0.)
        self._tree_vert.vertices += [v0,v1,v2,v3]
        
        self._id_node = -1
        self._nodes = []
        self._bb = [] # bounding boxes of nodes

        ## Build a root node
        self._create_node(0, -1, [0,1,2,3])

    def _get_new_node_id(self):
        self._id_node += 1
        return self._id_node

    def _create_node(self, depth, parent, verts):
        assert len(verts)==4
        new_id = self._get_new_node_id()
        new_node = QuadTree.Node(new_id, depth, parent, verts)
        new_node.points = []
        new_node.children = []
        vmin,vmax = self.vertex(verts[0]), self.vertex(verts[2])
        self._bb.append(AABB((vmin.x, vmin.y), (vmax.x, vmax.y)))
        self._nodes.append(new_node)
        return new_id

    def vertex(self,i):
        return self._tree_vert.vertices[i]

    def _subdivide(self, node_id):
        """
        Subdivision convention
        v3 -- v6 -- v2
        |     |      |
        |     |      |
        v7 -- v8 --  v5
        |     |      |
        |     |      |
        v0 -- v4 -- v1 
        """
        node : QuadTree.Node = self._nodes[node_id]
        if len(node.children)!=0 : return # not a leaf -> cannot subdivide
        nv = len(self._tree_vert.vertices)
        v0,v1,v2,v3 = node.corners
        p0,p1,p2,p3 = (self.vertex(v) for v in node.corners)
        v4,p4 = nv,   (p0+p1)/2
        v5,p5 = nv+1, (p1+p2)/2
        v6,p6 = nv+2, (p2+p3)/2
        v7,p7 = nv+3, (p0+p3)/2
        v8,p8 = nv+4, (p0+p1+p2+p3)/4
        self._tree_vert.vertices += [p4,p5,p6,p7,p8]
        n1,n2,n3,n4 = (
            self._create_node(node.depth+1, node.id, [v0,v4,v8,v7]),
            self._create_node(node.depth+1, node.id, [v4,v1,v5,v8]),
            self._create_node(node.depth+1, node.id, [v8,v5,v2,v6]),
            self._create_node(node.depth+1, node.id, [v7,v8,v6,v3])
        )
        node.children = [n1,n2,n3,n4]
        for p in node.points:
            self._insert_point(self.points[p], p, node.id)
        node.points = []
    
    def insert_point(self, x:float, y:float):
        """Adds a point inside the Quadtree Structure
        """
        pt = Vec(x, y, 0.)
        if not self._bb[0].contains_point(pt): return
        self.points.append(pt)
        self._insert_point(pt, len(self.points)-1, 0)

    def _insert_point(self, pt, pt_id, node_id):
        node : QuadTree.Node = self._nodes[node_id]
        if len(node.children)!=0:
            # not a leaf -> recursive call to children
            for child in node.children:
                self._insert_point(pt, pt_id, child)
        else:
            BB : AABB = self._bb[node_id]
            if BB.contains_point(pt):
                self._where[pt_id] = node_id
                node.points.append(pt_id)
                if len(node.points)>QuadTree.MAX_PT_PER_NODE:
                    # trigger subdivision
                    self._subdivide(node_id)
    
    def query(self, pt:Vec):
        return
    
    def find_all_points_in(self, box:AABB) -> list:
        return
    
    def export_as_polyline(self) -> PolyLine:
        # TODO : make a version with correct connectivity
        out = RawMeshData()
        out.vertices = self._tree_vert.vertices
        for node in self._nodes:
            v0,v1,v2,v3 = node.corners
            out.edges += [(v0,v1), (v1,v2), (v2,v3), (v3,v0)]
        return PolyLine(out)