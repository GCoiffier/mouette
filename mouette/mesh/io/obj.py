from ..datatypes import RawMeshData, PolyLine
from ... import config
import numpy as np

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
    out = RawMeshData()
    face_data = []
    normals = []
    uv_coords = []
    for line in data:
        toks = line.split()
        if not toks: continue # empty line
        if toks[0] == 'v':
            out.vertices.append([ float(v) for v in toks[1:4]]) # get only the first three components and ignore the rest
        elif toks[0] == 'vn':
            normals.append([ float(v) for v in toks[1:]])
        elif toks[0] == 'vt':
            uv_coords.append([float(toks[1]), float(toks[2])])
        elif toks[0] == 'f':
            face_data.append([parse_vertex(vstr) for vstr in toks[1:]])
        elif toks[0] == 'l':
            v1,v2 = int(toks[1])-1, int(toks[2])-1
            out.edges.append((v1,v2))

    ## Process f_field into faces, uv_coords and normals
    out.face_corners_attr["uv_coords"] = []
    out.face_corners_attr["normals"] = []
    for iF,F in enumerate(face_data):
        face = []
        for (vid,tid,nid) in F:
            face.append(vid)
            if nid!=-1:
                out.face_corners_attr["normals"].append(normals[nid])
            if tid!=-1:
                out.face_corners_attr["uv_coords"].append(uv_coords[tid])
        out.faces.append(face)
        out.face_corners += [[face[i], iF] for i in range(len(F))]
    return out

def export_obj(mesh, path):
    has_texcoords_vert = mesh.vertices.has_attribute("uv_coords") and not mesh.vertices.get_attribute("uv_coords").empty()
    has_texcoords_corners = mesh.face_corners.has_attribute("uv_coords") and not mesh.face_corners.get_attribute("uv_coords").empty()
    has_normals = mesh.vertices.has_attribute("normals") and not mesh.vertices.get_attribute("normals").empty()

    with open( path, 'w' ) as ofile:
        for vtx in mesh.vertices:
            ofile.write('v '+' '.join(['{}'.format(v) for v in vtx])+'\n')

        if has_texcoords_corners:
            texcoords = mesh.face_corners.get_attribute("uv_coords").as_array()
            for tex in texcoords:
                ofile.write('vt '+' '.join([str(vt) for vt in tex])+'\n')
                    
        elif has_texcoords_vert:
            texcoords = mesh.vertices.get_attribute("uv_coords").as_array()
            for tex in texcoords:
                ofile.write('vt '+' '.join([str(vt) for vt in tex])+'\n')

        if has_normals :
            normals = mesh.vertices.get_attribute("normals")
            for i in mesh.id_vertices:
                nrm = normals[i]
                ofile.write('vn '+' '.join(['{}'.format(vn) for vn in nrm])+'\n')

        if config.export_edges_in_obj:
            if not config.complete_edges_from_faces or isinstance(mesh, PolyLine):
                for a,b in mesh.edges:
                    ofile.write(f'l {a+1} {b+1}\n')
            elif mesh.edges.has_attribute("hard_edges"):
                hard_edges = mesh.edges.get_attribute("hard_edges")
                for e,(A,B) in enumerate(mesh.edges):
                    if hard_edges[e]:
                        ofile.write(f'l {A+1} {B+1}\n')

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