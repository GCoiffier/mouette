import mouette as M

### Point Clouds ###

def pc_sample1():
    m = M.mesh.PointCloud()
    m.vertices += [
        M.Vec(0., 0., 0.),
        M.Vec(1., 0., 0.),
        M.Vec(0., 1., 0.),
        M.Vec(0., 0., 1.),
    ]
    return m

def pc_random10():
    m = M.mesh.PointCloud()
    m.vertices += [
        M.Vec.random(3) for _ in range(10)
    ]
    return m

def pc_grid():
    m = M.mesh.PointCloud()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                m.vertices.append(M.Vec(float(x), float(y), float(z)))
    return m

point_clouds = [
    pc_sample1(),
    pc_random10(),
    pc_grid()
]