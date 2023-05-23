import scipy.io as spio
import os
import QP_library

PROBLEMS_FOLDER = "maros_meszaros_data"
# Get maros problems list
problems_dir = os.path.join(".", PROBLEMS_FOLDER)
# List of problems in .mat format
lst_probs = [f for f in os.listdir(problems_dir) if \
    f.endswith('.mat')]
for f in lst_probs:
    # Create example instance
    full_name = os.path.join(".", PROBLEMS_FOLDER, f)
    m = spio.loadmat(full_name)

    Q = m['Q'].tocsr()
    Q_value = Q.data.tolist()
    Q_rowstart = Q.indptr.tolist()
    Q_column = Q.indices.tolist()

    A = m['A'].tocsr()
    A_value = A.data.tolist()
    A_rowstart = A.indptr.tolist()
    A_column = A.indices.tolist()

    nRow = A.get_shape()[0]
    nCol = A.get_shape()[1]

    c = m['c'].tolist()
    b = m['rl'].tolist()

    l = m['lb'].tolist()
    u = m['ub'].tolist()

    para = {}
    para['maxADMMiter'] = 300
    para['maxALMiter'] = 20
    py_result = QP_library.QP_solve(Q_value, Q_rowstart, Q_column, A_value, A_rowstart, A_column, b, c, l, u, para)
