# SemesterProject
 defining models with differential Operators in Akantu
## Files 
- variationnal_operator_function.py : contains all the implementation.

- plot.py : necessary to generate gmsh files and to plot mesh/results.
  
- test_heat.py : files that solve a heat equation with 3 types of mesh ( segment2, segment3, triangle3).
  
- patchtest_navier.py : do a patch test for Navier-Cauchy equation.

- test_CurlOperator.py : test implementation of curl operator.

- test_GenericOperator.py : test GenericOperator Class in 2 applications.

- A-glyphs.pvsm : file to plot the result of the "test_CurlOperator.py"
  
- test_segment2_FD1.py, test_segment2_FD2.py, test_segment3_FD1.py, test_segment3_FD2.py, test_triangle3_FD1.py, test_triangle3_FD2.py : files with assertion that control the results of different integration for Navier-Cauchy equation (int(N), int(B), K)

