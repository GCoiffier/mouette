NOT_AN_ID = 4294967295 # An integer (2^32 -1) representing a default value in geogram file format

complete_edges_from_faces = True
"""
In most file formats, surface meshes are given as a set of vertices and a set of faces, since edges can be retrieved from faces.
If this option is set to true, the edge set of the mesh will be constructed from the face information.
"""


complete_faces_from_cells = True
"""
In most file formats, volume meshes are given as a set of vertices and tetrahedra (cells).
If this option is set to true, the face set of the mesh will be constructed from the cells information.
"""


export_edges_in_obj = True
"""
Whether to export edges as 'l <v> <v>' fields in .obj file format
"""

sort_neighborhoods = True
"""
If set to true, this will sort the corner connectivity arrays of a surface mesh, according to the computed half edges
"""

disable_duplicate_attribute_warning = True
"""
When creating an attribute that already exists on a mesh, mouette prints a warning in the console and returns the attribute currently carrying this name.
If this flag is set to True, no warning is printed. We then assume that you know what you are doing
"""