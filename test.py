import QP_library

Q_value = [4,2,2]
Q_rowstart = [0,1,2,3]
Q_column = [0,1,2]
A_value = [1,2,3,2,1]
A_rowstart = [0,3,5]
A_column = [0,1,2,0,1]
b = [1, 1]
c = [1,0,1]
l = [-1e300, -1, -1]
u = [1e300, 1, 1]
para = {}
para['maxADMMiter'] = 100
para['maxALMiter'] = 80
py_result = QP_library.QP_solve(Q_value,Q_rowstart,Q_column,A_value,A_rowstart,A_column,b,c,l,u,para)
print(py_result)