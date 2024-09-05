from ..mesh.datatypes import *
from ..mesh.mesh_data import DataContainer, RawMeshData
from ..geometry import Vec, AABB
import numpy as np

from dataclasses import dataclass
from enum import Enum
from abc import abstractmethod

# list_to_number = lambda L : reduce(lambda a,b : (a<<1) + int(b), L)

# class _SpacePartitionTree:
    
#     @dataclass
#     class Node:
#         id : int # id of the node
#         depth : int # depth of the node inside the tree
#         parent : int  # id of parent node (-1 if root node)
#         corners : list # indices of vertex corners of the node (should be 4 elements)
#         children : list = None # id of children nodes
#         points: set = None # id of contained points

#     def __init__(self, points : np.ndarray, dim :int, **kwargs):
#         self.dim = dim
#         self.max_depth : int = kwargs.get("max_depth", 8)
#         self.points = DataContainer(id="vertices")
#         self.domain = kwargs.get("domain", None)
#         if self.domain is None or not isinstance(self.domain, AABB) or self.domain.dim != dim:
#             diameter = np.amax(self.points) - np.amin(self.points)
#             self.domain = AABB.of_points(self.points, 0.01 * diameter)

#         self._where = self.points.create_attribute("id", int)

#         self._id_node = -1
#         self._nodes = []

#         ## Build a root node
#         self._create_node(0, -1, [0,1,2,3])

#     def _get_new_node_id(self):
#         self._id_node += 1
#         return self._id_node

#     def _create_node(self, depth, parent, verts):
#         assert len(verts)==4
#         new_id = self._get_new_node_id()
#         new_node = QuadTree.Node(new_id, depth, parent, verts)
#         new_node.points = []
#         new_node.children = []
#         vmin,vmax = self.vertex(verts[0]), self.vertex(verts[2])
#         self._bb.append(AABB((vmin.x, vmin.y), (vmax.x, vmax.y)))
#         self._nodes.append(new_node)
#         return new_id

#     def vertex(self,i):
#         return self._tree_vert.vertices[i]

#     def _subdivide(self, node_id):
#         """
#         Subdivision convention
#         v3 -- v6 -- v2
#         |     |      |
#         |     |      |
#         v7 -- v8 --  v5
#         |     |      |
#         |     |      |
#         v0 -- v4 -- v1 
#         """
#         node : QuadTree.Node = self._nodes[node_id]
#         if len(node.children)!=0 : return # not a leaf -> cannot subdivide
#         nv = len(self._tree_vert.vertices)
#         v0,v1,v2,v3 = node.corners
#         p0,p1,p2,p3 = (self.vertex(v) for v in node.corners)
#         v4,p4 = nv,   (p0+p1)/2
#         v5,p5 = nv+1, (p1+p2)/2
#         v6,p6 = nv+2, (p2+p3)/2
#         v7,p7 = nv+3, (p0+p3)/2
#         v8,p8 = nv+4, (p0+p1+p2+p3)/4
#         self._tree_vert.vertices += [p4,p5,p6,p7,p8]
#         n1,n2,n3,n4 = (
#             self._create_node(node.depth+1, node.id, [v0,v4,v8,v7]),
#             self._create_node(node.depth+1, node.id, [v4,v1,v5,v8]),
#             self._create_node(node.depth+1, node.id, [v8,v5,v2,v6]),
#             self._create_node(node.depth+1, node.id, [v7,v8,v6,v3])
#         )
#         node.children = [n1,n2,n3,n4]
#         for p in node.points:
#             self._insert_point(self.points[p], p, node.id)
#         node.points = []
    
#     def insert_point(self, x:float, y:float):
#         """Adds a point inside the Quadtree Structure
#         """
#         pt = Vec(x, y, 0.)
#         if not self._bb[0].contains_point(pt): return
#         self.points.append(pt)
#         self._insert_point(pt, len(self.points)-1, 0)

#     def _insert_point(self, pt, pt_id, node_id):
#         node : QuadTree.Node = self._nodes[node_id]
#         if len(node.children)!=0:
#             # not a leaf -> recursive call to children
#             for child in node.children:
#                 self._insert_point(pt, pt_id, child)
#         else:
#             BB : AABB = self._bb[node_id]
#             if BB.contains_point(pt):
#                 self._where[pt_id] = node_id
#                 node.points.append(pt_id)
#                 if len(node.points)>QuadTree.MAX_PT_PER_NODE:
#                     # trigger subdivision
#                     self._subdivide(node_id)
    
#     def query(self, pt:Vec):
#         return
    
#     def find_all_points_in(self, box:AABB) -> list:
#         return
    
#     @abstractmethod
#     def export_visualization(self):
#         return
    

# class QuadTree(_SpacePartitionTree):
#     def __init__(self, points : np.ndarray, **kwargs):
#         points = np.array(points)[:,:2]
#         super().__init__(points, 2, **kwargs)

#     def export_visualization(self) -> PolyLine:
#         out = RawMeshData()
#         out.vertices = self._tree_vert.vertices
#         for node in self._nodes:
#             v0,v1,v2,v3 = node.corners
#             out.edges += [(v0,v1), (v1,v2), (v2,v3), (v3,v0)]
#         return PolyLine(out)


# class OcTree(_SpacePartitionTree):
#     def __init__(self, points : np.ndarray, **kwargs):
#         points = np.array(points)[:,:3]
#         super().__init__(points, 3, **kwargs)

#     def export_visualization(self) -> PolyLine:
#         out = RawMeshData()
#         out.vertices = self._tree_vert.vertices
#         for node in self._nodes:
#             v0,v1,v2,v3 = node.corners
#             out.edges += [(v0,v1), (v1,v2), (v2,v3), (v3,v0)]
#         return PolyLine(out)
