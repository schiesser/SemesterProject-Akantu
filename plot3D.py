import numpy as np
import matplotlib.pyplot as plt
import meshio
import subprocess
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection



def meshGeo(filename, dim=3, order=1, element_type='line'):
    out = os.path.splitext(filename)[0] + '.msh'
    ret = subprocess.run(f"gmsh -3 -order {order} -o {out} {filename}", shell=True)
    if ret.returncode:
        print("Beware, gmsh could not run: mesh is not generated")
    else:
        print("Mesh generated")
        mesh = readMesh(out, element_type)
        return mesh
    return None

def readMesh(filename, element_type):
    mesh = meshio.read(filename)
    return mesh.points[:, :2], mesh.cells_dict[element_type]

def plotMeshtetra(nodes, conn):
    # Increase the size of the figure
    fig = plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a list to hold the edges of the tetrahedra
    edges = []
    for tetra in conn:
        # Define the edges of the tetrahedron
        edges.extend([
            [nodes[tetra[0]], nodes[tetra[1]]],
            [nodes[tetra[0]], nodes[tetra[2]]],
            [nodes[tetra[0]], nodes[tetra[3]]],
            [nodes[tetra[1]], nodes[tetra[2]]],
            [nodes[tetra[1]], nodes[tetra[3]]],
            [nodes[tetra[2]], nodes[tetra[3]]]
        ])
    
    # Create a Line3DCollection from the tetrahedron edges
    edge_collection = Line3DCollection(edges, colors='k', linestyles='--', linewidths=0.8)
    ax.add_collection3d(edge_collection)
    
    # Plot the nodes with a smaller size
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color='black', s=10)  # 's' parameter adjusts the size of the points
    
    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the aspect ratio to be equal
    ax.set_box_aspect([np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1]), np.ptp(nodes[:, 2])])

    # Save and show the plot
    plt.savefig('MeshElementTetrahedron.png')

def plotMeshHexa(nodes, conn):
    # Increase the size of the figure
    fig = plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a list to hold the edges of the hexahedra
    edges = []
    for hexa in conn:
        # Define the edges of the hexahedron
        edges.extend([
            [nodes[hexa[0]], nodes[hexa[1]]],
            [nodes[hexa[1]], nodes[hexa[2]]],
            [nodes[hexa[2]], nodes[hexa[3]]],
            [nodes[hexa[3]], nodes[hexa[0]]],
            [nodes[hexa[4]], nodes[hexa[5]]],
            [nodes[hexa[5]], nodes[hexa[6]]],
            [nodes[hexa[6]], nodes[hexa[7]]],
            [nodes[hexa[7]], nodes[hexa[4]]],
            [nodes[hexa[0]], nodes[hexa[4]]],
            [nodes[hexa[1]], nodes[hexa[5]]],
            [nodes[hexa[2]], nodes[hexa[6]]],
            [nodes[hexa[3]], nodes[hexa[7]]]
        ])
    
    # Create a Line3DCollection from the hexahedron edges
    edge_collection = Line3DCollection(edges, colors='k', linestyles='--', linewidths=0.8)
    ax.add_collection3d(edge_collection)
    
    # Plot the nodes with a smaller size
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color='black', s=10)  # 's' parameter adjusts the size of the points
    
    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the aspect ratio to be equal
    ax.set_box_aspect([np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1]), np.ptp(nodes[:, 2])])

    # Save and show the plot
    plt.savefig('MeshElementHexahedron.png')