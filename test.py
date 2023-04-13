import QP_library

Q_value = [4,2,2,6]
Q_rowstart = [0,2,4]
Q_column = [0,1,0,1]
A_value = [1,1,2,-1]
A_rowstart = [0,2,4]
A_column = [0,1,0,1]
b = [1, 0]
c = [0,0]
l = [-1e300, -1e300]
u = [1e300, 1e300]
QP_library.QP_solve(Q_value,Q_rowstart,Q_column,A_value,A_rowstart,A_column,b,c,l,u)