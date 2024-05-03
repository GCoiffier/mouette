from ..mesh_data import RawMeshData
from ...geometry import Vec
from ... import config
from ...utils import keyify

def import_obj(path : str):
    """
    Parameters:
        path (str): the input file path
    """

    with open(path, 'r' ) as objf:
        output = parse_obj_data(objf.readlines())
    return output

def parse_vertex( vstr ):
    vals = vstr.split('/')
    vid = int(vals[0])-1
    tid = int(vals[1])-1 if len(vals)>1 and vals[1] else -1
    nid = int(vals[2])-1 if len(vals) > 2 else -1
    return (vid,tid,nid) 

def parse_obj_data(data):
    obj = RawMeshData()
    normals = []
    uv_coords = []
    faces = []
    for line in data:
        toks = line.split()
        if not toks: continue # empty line
        if toks[0] == 'v':
            obj.vertices.append( Vec([ float(v) for v in toks[1:4]]) ) # get only the first three components and ignore the rest
        elif toks[0] == 'vn':
            normals.append( Vec([ float(v) for v in toks[1:]]) )
        elif toks[0] == 'vt':
            uv_coords.append( Vec([float(toks[1]), float(toks[2])]) )
        elif toks[0] == 'f':
            faces.append([ parse_vertex(vstr) for vstr in toks[1:] ])
        elif toks[0] == 'l':
            v1,v2 = int(toks[1])-1, int(toks[2])-1
            e = keyify(v1,v2)
            obj.edges.append(e)

    normals_attr = obj.vertices.create_attribute("normals", float, 3)
    uv_attr = obj.face_corners.create_attribute("uv_coords", float, 2)
    ic = 0
    for iF,F in enumerate(faces):
        face = []
        for (vid,tid,nid) in F:
            face.append(vid)
            if nid!=-1:
                normals_attr[vid] = normals[nid]
            if tid!=-1:
                uv_attr[ic] = uv_coords[tid]
            ic += 1
        obj.faces.append(face)
        obj.face_corners += [(v,iF) for v in face]
    if normals_attr.empty() : obj.vertices.delete_attribute("normals")
    if uv_attr.empty(): obj.face_corners.delete_attribute("uv_coords")
    return obj

def export_obj(mesh, path):
    has_texcoords_vert = mesh.vertices.has_attribute("uv_coords") and not mesh.vertices.get_attribute("uv_coords").empty()
    has_texcoords_corners = mesh.face_corners.has_attribute("uv_coords") and not mesh.face_corners.get_attribute("uv_coords").empty()
    has_normals = mesh.vertices.has_attribute("normals") and not mesh.vertices.get_attribute("normals").empty()

    with open( path, 'w' ) as ofile:
        for vtx in mesh.vertices:
            ofile.write('v '+' '.join(['{}'.format(v) for v in vtx])+'\n')

        if has_texcoords_corners:
            texcoords = mesh.face_corners.get_attribute("uv_coords").as_array(len(mesh.face_corners))
            for tex in texcoords:
                ofile.write('vt '+' '.join([str(vt) for vt in tex])+'\n')
                    
        elif has_texcoords_vert:
            texcoords = mesh.vertices.get_attribute("uv_coords").as_array(len(mesh.vertices))
            for tex in texcoords:
                ofile.write('vt '+' '.join([str(vt) for vt in tex])+'\n')

        if has_normals :
            normals = mesh.vertices.get_attribute("normals")
            for i in mesh.id_vertices:
                nrm = normals[i]
                ofile.write('vn '+' '.join(['{}'.format(vn) for vn in nrm])+'\n')

        if config.export_edges_in_obj:
            if not config.complete_edges_from_faces:
                for a,b in mesh.edges:
                    ofile.write(f'l {a+1} {b+1}\n')
            elif mesh.edges.has_attribute("hard_edges"):
                for e in mesh.edges.get_attribute("hard_edges"):
                    a,b = mesh.edges[e]
                    ofile.write(f'l {a+1} {b+1}\n')

        cnr_id = 1
        for face in mesh.faces:
            str_face = ""
            for vid in face:
                str_id = str(vid+1)
                if has_texcoords_corners:
                    str_id+="/{}".format(cnr_id)
                elif has_texcoords_vert:
                    str_id+="/{}".format(vid+1)
                elif has_normals:
                    str_id+="/"
                if has_normals:
                    str_id+="/{}".format(vid+1)
                str_face += str_id + " "
                cnr_id += 1
            ofile.write('f ' + str_face + "\n")