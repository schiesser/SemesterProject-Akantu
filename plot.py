#!/usr/bin/env python3

##########
import numpy as np
import matplotlib.pyplot as plt
import meshio
import subprocess
import os

def readMesh(filename, element_type='line'):
    mesh = meshio.read(filename)
    return mesh.points[:, :2], mesh.cells_dict[element_type]

def plotMesh(coords, connectivity, **kwargs):
    plt.axes().set_aspect('equal')
    plt.plot(coords[:, 0], coords[:, 1], 'ko')  # Affiche les nœuds
    for segment in connectivity:
        x = [coords[segment[0], 0], coords[segment[1], 0]]
        y = [coords[segment[0], 1], coords[segment[1], 1]]
        plt.plot(x, y, 'b-')  # Affiche les segments en bleu
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(kwargs.get('title', 'Mesh Plot'))
    plt.show()

def meshGeo(filename, dim=2, order=1):
    ret = subprocess.run(f"gmsh -2 -order 1 -o tmp.msh {filename}", shell=True)
    if ret.returncode:
        print("Beware, gmsh could not run: mesh is not generated")
    else:
        print("Mesh generated")
        mesh = readMesh('tmp.msh')
        os.remove('tmp.msh')
        return mesh
    return None
