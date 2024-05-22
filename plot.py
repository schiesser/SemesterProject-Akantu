import numpy as np
import matplotlib.pyplot as plt
import meshio
import subprocess
import os
import matplotlib.tri as tri

def readMesh(filename, element_type):
    mesh = meshio.read(filename)
    return mesh.points[:, :2], mesh.cells_dict[element_type]

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

def plotMesht3(nodes, conn):
    triangles = tri.Triangulation(nodes[:, 0], nodes[:, 1], conn)
    plt.scatter(nodes[:, 0], nodes[:, 1], color='black')
    plt.triplot(triangles, '--', lw=.8)
    plt.savefig('MeshElementTriangle.png')