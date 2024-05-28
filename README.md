# SemesterProject
 defining models with differential Operators in Akantu
## Files 
- plot.py : necessary to generate gmsh files and to plot mesh
- test_heat... : files that solve a heat equation with 3 types of mesh ( segment2, segment3, triangle3)
- test_comparaisonAkantu.py : do a patch test for Navier-Cauchy equation (vérifier nom!) . It compares Akantu's solution and mine.
- test_segment2_FD1.py, test_segment2_FD2.py, test_segment3_FD1.py, test_segment3_FD2.py, test_triangle3_FD1.py, test_triangle3_FD2.py : files with assertion that control the results of different integration for Navier-Cauchy equation (int(N), int(B), K)
- variationnal_operator_function.py : contains all the implementation
