import numpy as np
import matplotlib.pyplot as plt
import meshio
import subprocess
import os
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def readMesh(filename, element_type):
    mesh = meshio.read(filename)
    return mesh.points[:, :2], mesh.cells_dict[element_type]
"""
def plotMesh(coords, connectivity):
    plt.axes().set_aspect('equal')
    for segment in connectivity:
        x = [coords[segment[0], 0], coords[segment[-1], 0]]
        y = [coords[segment[0], -1], coords[segment[-1], -1]]
        plt.plot(x, y, 'b--',markersize=3)
    plt.scatter(coords[:, 0], coords[:, 1], color='black')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('mesh_segment2.png')
"""

def meshGeo(filename, dim=2, order=1, element_type='line'):
    out = os.path.splitext(filename)[0] + '.msh'
    ret = subprocess.run(f"gmsh -{dim} -order {order} -o {out} {filename}", shell=True)
    if ret.returncode:
        print("Beware, gmsh could not run: mesh is not generated")
    else:
        print("Mesh generated")
        mesh = readMesh(out, element_type)
        return mesh
    return None
"""
def plotMeshs2(coords, connectivity):
    plt.axes().set_aspect('equal')
    for segment in connectivity:
        x = coords[segment, 0]
        y = coords[segment, 1]
        plt.plot(x, y, 'b--',markersize=3)
    plt.scatter(coords[:, 0], coords[:, 1], color='black')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('mesh_segment3.png')
"""
def generate_scalar_field(field, axis=None):
    if axis is None:
        field = np.linalg.norm(field, axis=1)
    else:
        field = field[:, axis]
    return field

def plotMesht3(coords, connectivity, nodal_field=None, title = None, name_file = "MeshElementTriangle.png",**kwargs):
    triangles = tri.Triangulation(coords[:, 0], coords[:, 1], connectivity)
    plt.axes().set_aspect('equal')
    if nodal_field is not None:
        nodal_field = nodal_field.reshape(coords.shape[0], nodal_field.size//coords.shape[0])
        nodal_field = generate_scalar_field(nodal_field, **kwargs)
        contour = plt.tricontourf(triangles, nodal_field)
        plt.colorbar(contour)

    plt.xlabel('X')
    plt.ylabel('Y')
    if title is not None :
        plt.title(title)

    t = plt.triplot(triangles, '--',c="black", lw=1) 
    plt.savefig(name_file)
    plt.close()

def plotMeshs(coords, connectivity, nodal_field=None, title=None, name_file="MeshSegment.png", **kwargs):
    plt.figure()
    plt.axes().set_aspect('equal')

    for segment in connectivity:
        x = coords[segment, 0]
        y = coords[segment, 1]
        plt.plot(x, y, 'b--', markersize=5)

    if nodal_field is not None:
        segments = np.array([coords[segment] for segment in connectivity])
        values = np.array([nodal_field[segment] for segment in connectivity])
        
        # Créer une collection de lignes
        lc = LineCollection(segments, cmap='viridis', array=values.mean(axis=1), linewidths=2)
        plt.gca().add_collection(lc)
        plt.colorbar(lc, fraction=0.025, pad=0.05)
    
    plt.scatter(coords[:, 0], coords[:, 1], color='black', s=10)

    plt.xlabel('X')
    plt.ylabel('Y')
    if title is not None:
        plt.title(title)
    
    plt.savefig(name_file)
    plt.close()

def plotMeshtetra(nodes, conn):
    
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    edges = []
    for tetra in conn:
        edges.extend([[nodes[tetra[0]], nodes[tetra[1]]],[nodes[tetra[0]], nodes[tetra[2]]],[nodes[tetra[0]], nodes[tetra[3]]],[nodes[tetra[1]], nodes[tetra[2]]],[nodes[tetra[1]], nodes[tetra[3]]],[nodes[tetra[2]], nodes[tetra[3]]]])
    
    edge_collection = Line3DCollection(edges, colors='k', linestyles='--', linewidths=0.8)
    ax.add_collection3d(edge_collection)
    
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color='black', s=10)
    
    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the aspect ratio to be equal
    ax.set_box_aspect([np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1]), np.ptp(nodes[:, 2])])

    # Save and show the plot
    plt.savefig('MeshElementTetrahedron.png')