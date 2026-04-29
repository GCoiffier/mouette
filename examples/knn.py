import mouette as M
import numpy as np
from collections import deque

DIM = 2
N_POINTS = 1_000_000

domain = M.geometry.AABB.unit_cube(DIM, centered=True)
points = M.sampling.sample_AABB(domain, N_POINTS)
pc = M.mesh.from_arrays(points)

print("begin build")
tree = M.spatial.KDTree(points)
print("end build")

def generate_kdtree_polyline(tree, domain : M.geometry.AABB):
    V,E = [],[]
    e = 0
    queue = deque()
    queue.append(0) # add root
    while len(queue)>0:
        node_id = queue.popleft()
        if tree.is_leaf(node_id): continue
        node = tree.nodes[node_id]
        bb = M.geometry.AABB.intersection(domain, node.bb) # intersect with the domain
        if node.split_axis == 0:
            p1 = M.Vec(node.split_value, bb.mini[1])
            p2 = M.Vec(node.split_value, bb.maxi[1])
        elif node.split_axis == 1:
            p1 = M.Vec(bb.mini[0], node.split_value)
            p2 = M.Vec(bb.maxi[0], node.split_value)
        V += (p1,p2)
        E.append((2*e,2*e+1))
        e += 1
        queue.append(node.left)
        queue.append(node.right)
    polyline = M.mesh.from_arrays(np.array(V), np.array(E))
    return polyline

if DIM==2:
    print("begin draw")
    pl = generate_kdtree_polyline(tree, domain)
    M.mesh.save(pl, "kdtree.mesh")
    print("end draw")


print("begin query ball")
near_idx = tree.query_radius(M.Vec.zeros(DIM), 0.1)
print("enq query")
nearB = pc.vertices.create_attribute("nearBall", bool)
for i in near_idx: nearB[i] = True

print("begin query K")
near_idx = tree.query(M.Vec.zeros(DIM), 200)
print("end query")
nearK = pc.vertices.create_attribute("nearK", bool)
for i in near_idx: nearK[i] = True

M.mesh.save(pc, "points.geogram_ascii")