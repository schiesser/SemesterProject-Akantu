import numpy as np
import matplotlib.pyplot as plt
import meshio
import subprocess
import os

def readMesh(filename, element_type):
    mesh = meshio.read(filename)
    return mesh.points[:, :2], mesh.cells_dict[element_type]

def plotMesh(coords, connectivity, **kwargs):
    plt.axes().set_aspect('equal')
    plt.plot(coords[:, 0], coords[:, 1], 'ko')
    for segment in connectivity:
        x = [coords[segment[0], 0], coords[segment[1], 0]]
        y = [coords[segment[0], 1], coords[segment[1], 1]]
        plt.plot(x, y, 'b-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(kwargs.get('title', 'Mesh Plot'))
    #plt.show()
    plt.savefig('mesh.png')

def meshGeo(filename, dim=2, order=1, element_type='line'):
    out = os.path.splitext(filename)[0] + '.msh'
    ret = subprocess.run(f"gmsh -2 -order {order} -o {out} {filename}", shell=True)
    if ret.returncode:
        print("Beware, gmsh could not run: mesh is not generated")
    else:
        print("Mesh generated")
        mesh = readMesh(out, element_type)
        return mesh
    return None

def plotMeshs3(points, conn):
    plt.figure(figsize=(8, 6))
    plt.plot(points[:, 0], points[:, 1], 'ko', markersize=3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Maillage')
    plt.axis('equal')
    plt.savefig('segment3mesh.png')