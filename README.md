# SemesterProject
 Objective : defining models with differential operators in Akantu
## Files 
- variationnal_operator_function.py : contains all the implementation.

- plot.py : necessary to generate gmsh files and to plot mesh/results.
  
- test_heat.py : files that solve a heat equation with 3 types of mesh ( segment2, segment3, triangle3).
  
- patchtest_navier.py : do a patch test for Navier-Cauchy equation.

- test_CurlOperator.py : test implementation of curl operator.

- test_GenericOperator.py : test GenericOperator Class in 2 applications.
  
- test_segment2.py, test_segment3.py, test_triangle3.py : files with assertion that control the class N, matmul operation, transpose, class integrate and class assembly.

- A-glyphs.pvsm, barre_paraview.csv : file to plot the result of the "test_CurlOperator.py" on paraview.
## References
- NumPy documentation
  
- Matplotlib documentation

- Akantu documentation

- Course CIVIL-321: Numerical modelling of solids and structures
