//
// Created by chenkaihuang on 6/7/22.
//

#include <string.h>
#include "algorithm.h"

//#define USE_CBLAS 1
#define SSN_debug 1
//#define sigma_debug 0

using namespace std;

void read_bin(const char *lpFileName, mfloat* X, int len) {
    std::ifstream in(lpFileName, std::ios::in | std::ios::binary);
    if (!in.good()) {
        std::cout << "Cannot open file: " << lpFileName << std::endl;
        return;
    }
    in.read((char *) X, sizeof(mfloat)*len);
    std::cout << in.gcount() << " bytes read" << std::endl;
    in.close();
}

void write_bin(const char *lpFileName, mfloat* X, int len) {
    std::ofstream out(lpFileName, std::ios::out | std::ios::binary);
    if (!out.good()) {
        std::cout << "Cannot open file: " << lpFileName << std::endl;
        return;
    }
    out.write((char *) X, sizeof(mfloat)*len);
//    std::cout << out.gcount() << " bytes read" << std::endl;
    out.close();
}

void write_bin(const char *lpFileName, int* X, int len) {
    std::ofstream out(lpFileName, std::ios::out | std::ios::binary);
    if (!out.good()) {
        std::cout << "Cannot open file: " << lpFileName << std::endl;
        return;
    }
    out.write((char *) X, sizeof(int)*len);
//    std::cout << out.gcount() << " bytes read" << std::endl;
    out.close();
}

void proj_l2_mat(const mfloat* input, const mfloat* weight, mfloat* output, int m, int n, mfloat* norm_input, bool* rrf, bool byRow) {
    // parallel
    if (byRow) {
        norm_R(input, norm_input, m, n, true);
        for (int i = 0; i < m; ++i) {
            rrf[i] = norm_input[i] > weight[i];
            if (!rrf[i]) {
                for (int j = 0; j < n; ++j) {
                    output[i+j*m] = input[i+j*m];
//                p += m;
                }
            } else {
                mfloat factor = weight[i]/norm_input[i];
//            int p = i;
                for (int j = 0; j < n; ++j) {
                    output[i+j*m] = input[i+j*m]*factor;
//                p += m;
                }
            }
        }
    } else {
        norm_R(input, norm_input, m, n, false);
        for (int j = 0; j < n; ++j) {
            rrf[j] = norm_input[j] > weight[j];
            if (!rrf[j]) {
                for (int i = 0; i < m; ++i) {
                    output[i+j*m] = input[i+j*m];
                }
            } else {
                mfloat factor = weight[j]/norm_input[j];
                for (int i = 0; i < m; ++i) {
                    output[i+j*m] = input[i+j*m]*factor;
                }
            }
        }
    }

}

void proj_l2(const mfloat* input, mfloat weight, mfloat* output, int len, mfloat& norm_input, bool& rrf) {
    norm_input = norm2(input, len);
    rrf = norm_input > weight;
    if (!rrf) {
        vMemcpy(input, len, output);
    } else {
        mfloat factor = weight/norm_input;
        for (int i = 0; i < len; ++i) {
            output[i] = input[i]*factor;
        }
    }
}

void proj(const mfloat* input, mfloat* output, int len, const mfloat* l, const mfloat* u, bool* idx) {
    if (!idx) {
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(len, input, output, l ,u)
        for (int i = 0; i <= len; ++i) {
            if (input[i] < l[i] + 1e-14)
                output[i] = l[i]+ 1e-14;
            else if (input[i] >= u[i] - 1e-14)
                output[i] = u[i]- 1e-14;
            else
                output[i] = input[i];
        }
    } else {
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(len, input, output, l ,u, idx)
        for (int i = 0; i < len; ++i) {
            if (input[i] <= l[i] + 1e-14) {
                output[i] = l[i] + 1e-14;
                idx[i] = false;
            }
            else if (input[i] >= u[i] - 1e-14) {
                output[i] = u[i] - 1e-14;
                idx[i] = false;
            } else {
                output[i] = input[i];
                idx[i] = true;
            }

        }
    }
}

mfloat norm2(const mfloat* x, int len) {
    mfloat sum2 = 0.0;
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) shared(len, x) reduction(+:sum2) default(none)
    for (int i = 0; i < len; ++i) {
        sum2 += x[i] * x[i];
    }
    return sqrt(sum2);
}

mfloat distance(const mfloat* x, const mfloat* yf, int len) {
    mfloat sum2 = 0.0;
//#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) shared(len, x, yf) reduction(+:sum2) default(none)
    for (int i = 0; i < len; ++i) {
        sum2 += pow(x[i]-yf[i], 2);
    }
    return sqrt(sum2);
}

mfloat norm2_mat(const mfloat* x, int m, int n) {
    mfloat res = 0.0;
    mfloat *tmp = new mfloat [n];
    int *p = new int [n];
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) shared(m, p, tmp, n, x) default(none)
    for (int i = 0; i < n; ++i) {
        tmp[i] = 0; p[i] = i*m;
        for (int j = 0; j < m; ++j) {
            tmp[i] += x[p[i]] * x[p[i]];
            p[i]++;
        }
    }

    for (int i = 0; i < n; ++i)
        res += tmp[i];

    delete [] p;
    delete [] tmp;
    return sqrt(res);
}

void axpy(const mfloat a, const mfloat* x, const mfloat* y, mfloat* z, const int len){
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(len, y, z, a, x)
        for (int i = 0; i < len; ++i)
            z[i] = y[i] + a*x[i];
}


void partial_axpy(const mfloat a, const mfloat* x, const mfloat* y, mfloat* z, const int len, bool *idx){
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(len, y, z, a, x, idx)
    for (int i = 0; i < len; ++i) {
        if (idx[i]) {
            z[i] = y[i] + a*x[i];
        }
    }

}

void axpby(const mfloat a, const mfloat* x, mfloat b, const mfloat* y, mfloat* z, const int len){
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(len, y, z, a, x, b)
    for (int i = 0; i < len; ++i)
        z[i] = b*y[i] + a*x[i];
}

void partial_axpby(const mfloat a, const mfloat* x, mfloat b, const mfloat* y, mfloat* z, const int len, bool *idx){
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(len, y, z, a, x, b, idx)
    for (int i = 0; i < len; ++i) {
        if (idx[i]) {
            z[i] = b*y[i] + a*x[i];
        }
    }
}

void ax(mfloat a, const mfloat* x, mfloat* z, int len) {
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(len, z, a, x)
    for (int i = 0; i < len; ++i)
        z[i] = a*x[i];
}

void axpyT(const mfloat a, const mfloat* x, const mfloat* y, mfloat* z, const int m, const int n){
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(m, n, z, y, a, x)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            z[i+j*m] = y[i+j*m] + a*x[j+i*m];
}

void axpy_mat(const mfloat a, const mfloat* x, mfloat* yf, int m, int n){
    int *p = new int [n];
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(n,p,m,yf,a,x)
    for (int i = 0; i < n; ++i) {
        p[i] = i * m;
        for (int j = 0; j < m; ++j) {
            yf[p[i]] = yf[p[i]] + a*x[p[i]];
            p[i]++;
        }
    }
    delete [] p;
}

void axpy_R(const mfloat* a, const mfloat* x, const mfloat* yf, mfloat* zf, int m, int n, bool byRow) {
    if (byRow) {
        int *p = new int [m];
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) shared(m,n,zf,a,x,yf,p) default(none)
        for (int i = 0; i < m; ++i) {
            p[i] = i;
            for (int j = 0; j < n; ++j) {
                zf[p[i]] = a[i]*x[p[i]] + yf[p[i]];
                p[i] += m;
            }
        }
        delete [] p;
    } else {
        int *p = new int [n];
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) shared(m,n,zf,a,x,yf,p) default(none)
        for (int i = 0; i < n; ++i) {
            p[i] = i*m;
            for (int j = 0; j < m; ++j) {
                zf[p[i]] = a[i]*x[p[i]] + yf[p[i]];
                p[i]++;
            }
        }
        delete [] p;
    }

}

void xdoty(const mfloat* x, const mfloat* y, mfloat* z, int len, bool prod) {
    if (prod) {
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(len,z,x,y)
        for (int i = 0; i < len; ++i)
            z[i] = x[i] * y[i];
    } else {
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(len,z,x,y)
        for (int i = 0; i < len; ++i)
            z[i] = x[i] / y[i];
    }
}

mfloat* xdoty_mat(const mfloat* x, const mfloat* yf, int m, int n) {
    mfloat* zf = new mfloat [m*n];
    int *p = new int [n];
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(n,p,m,zf,x,yf)
    for (int i = 0; i < n; ++i) {
        p[i] = i * m;
        for (int j = 0; j < m; ++j) {
            zf[p[i]] = x[p[i]] * yf[p[i]];
            p[i]++;
        }
    }
    delete [] p;
    return zf;
}

mfloat xTy(const mfloat* x, const mfloat* yf, int len) {
    mfloat res = 0.0;
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) reduction(+:res) shared(len,x,yf)
    for (int i = 0; i < len; ++i)
        res += x[i] * yf[i];
    return res;
}

mfloat sum(const mfloat* x, int len) {
    mfloat res = 0.0;
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) reduction(+:res) shared(x, len) default(none)
    for (int i = 0; i < len; ++i)
        res += x[i];
    return res;
}

void knnsearch(const mfloat*x, int k, int m, int n, int* k_nearest, mfloat* dist) {
//    mfloat* xmy = new mfloat[m];
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(n, k_nearest)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            k_nearest[i*n+j] = j;

#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(n, k_nearest,x,m,dist)
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            dist[i*n+j] = distance(x+j*m, x+i*m, m);
            dist[j*n+i] = dist[i*n+j];
        }

    }
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(n, k_nearest,dist)
    for (int i = 0; i < n; ++i) {
        mSort(dist+i*n, dist+i*n+n, k_nearest+i*n);
    }
}


//void spMV_C(spC A, Mat x, int m, int n, mfloat* Ax) {
//    mfloat* value = A.value;
//    int* colStart = A.colStart;
//    int* row = A.row;
//    mfloat* xvec = x.vec;
//    assert(A.n)
//    for (int i = 0; i < n; ++i) {
//        mfloat temp = 0;
//        for (int j = colStart[i]; j < colStart[i+1]; ++j) {
//            temp += value[j]*xvec[row[j]];
//        }
//        Ax[i] = temp;
//    }
//}

void get_spR(const int* NodeArcMatrix, sparseRowMatrix * BT, int num) {
    BT->value = new mfloat [2*num];
    BT->rowStart = new int [num+1];
    BT->column = new int [2*num];
    mfloat* value = BT->value;
    int* colStart = BT->rowStart;
    int* row = BT->column;
    for (int i = 0; i < num+1; ++i) {
        colStart[i] = 2*i;
    }
    for (int i = 0; i < num; ++i) {
        row[i*2] = NodeArcMatrix[i*2];
        row[i*2+1] = NodeArcMatrix[i*2+1];
        value[i*2] = 1;
        value[i*2+1] = -1;
    }
}


sparseRowMatrix get_BT(sparseRowMatrix BT) {
    // somewhere wrong
    sparseRowMatrix Bf;
//    int* rowIndex = new int [2*num];
    int nnz = BT.rowStart[BT.nRow];
    Bf.value = new mfloat [nnz];
    // correct: BT.nCol -> BT.nCol + 2
    Bf.rowStart = new int [BT.nCol+2];
//    BT->rowLength = new int [m];
    Bf.column = new int [nnz];
    mfloat* newValue = Bf.value;
    int *rowStart = Bf.rowStart;
//    int *rowLength = BT->rowLength;
    int *column = Bf.column;
    mfloat* oldValue = BT.value;
    int *colStart = BT.rowStart;
    int *row = BT.column;
    for (int i = 0; i < BT.nCol+2; ++i) {
        rowStart[i] = 0;
    }
    // count per row
    for (int i = 0; i < nnz; ++i) {
        ++rowStart[row[i]+2];
    }

    // generate rowStart
    for (int i = 2; i < BT.nCol+2; ++i) {
        rowStart[i] += rowStart[i-1];
    }

    // main part
    for (int i = 0; i < BT.nRow; ++i) {
        for (int j = colStart[i]; j < colStart[i+1]; ++j) {
            int newIndex = rowStart[row[j] + 1]++;
            newValue[newIndex] = oldValue[j];
            column[newIndex] = i;
        }
    }

    Bf.nCol = BT.nRow;
    Bf.nRow = BT.nCol;
    return Bf;
    // shift
//    for (int i = 0; i < m+1; ++i)
//        rowStart[i] = rowStart[i+1];
}

void spMV_R(sparseRowMatrix A, const mfloat* x, int nRow, int nCol, mfloat* Ax) {
    mfloat* value = A.value;
    int* rowStart = A.rowStart;
    int* column = A.column;
    assert(A.nCol == nCol);
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) shared(nRow, Ax, rowStart, value, x, column, nCol) default(none)
    for (int i = 0; i < nRow; ++i) {
        Ax[i] = 0;
        for (int j = rowStart[i]; j < rowStart[i+1]; ++j) {
            assert(column[j] < nCol);
            Ax[i] += value[j]*x[column[j]];
        }
    }
}

void partial_spMV_2(sparseRowMatrix A, const mfloat* x, int m, int n, mfloat* Ax, spVec_int nzidx) {
    mfloat* value = A.value;
    int* rowStart = A.rowStart;
    int* column = A.column;
    assert(A.nCol == n);
//    assert(nzidx.number == m);
    int* nzidxv = nzidx.vec;
    for (int i = 0; i < n; ++i) Ax[i] = 0;

//#pragma omp parallel for
    for (int idx = 0; idx < nzidx.number; ++idx) {
//        int i = nzidxv[idx];
//        mfloat temp = x[idx];
        for (int j = rowStart[nzidxv[idx]]; j < rowStart[nzidxv[idx]+1]; ++j) {
            assert(column[j] < n);
            Ax[column[j]] += value[j]*x[idx];
        }
    }
}


void sum(const mfloat* X, mfloat* output, int m, int n) {
    for (int i = 0; i < n; ++i){
//        mfloat temp = 0.0;
//        int im = i*m;
//        for (int j = 0; j < m; ++j)
//            temp += X[im+j];
        output[i] = sum(X+i*m, n);
    }
}

void norm_R(const mfloat* X, mfloat* output, int m, int n, bool byRow) {
//#pragma omp parallel for
    if (byRow) {
        for (int i = 0; i < m; ++i) {
//        int p = i;
            mfloat temp = 0;
            for (int j = 0; j < n; ++j) {
                mfloat temp2 = X[i+j*m];
                temp += temp2*temp2;
//            p += m;
            }
            output[i] = sqrt(temp);
        }
    } else {
        for (int j = 0; j < n; ++j) {
            mfloat temp = 0;
            for (int i = 0; i < m; ++i) {
                mfloat temp2 = X[i+j*m];
                temp += temp2*temp2;
            }
            output[j] = sqrt(temp);
        }
    }

}

mfloat* xdoty_R(const mfloat* X, const mfloat* Y, int m, int n, bool byRow) {
    if (byRow) {
        mfloat* output = new mfloat [m];
        int *p = new int [m];
#pragma omp parallel for num_threads(1) shared(m,n,X,Y,p,output) default(none)
        for (int i = 0; i < m; ++i) {
            p[i] = i; output[i] = 0;
            for (int j = 0; j < n; ++j) {
                output[i] += X[p[i]]*Y[p[i]];
                p[i] += m;
            }
        }
        delete [] p;
        return output;
    } else {
        mfloat* output = new mfloat [n];
        int *p = new int [n];
#pragma omp parallel for num_threads(1) shared(m,n,X,Y,p,output) default(none)
        for (int i = 0; i < n; ++i) {
            p[i] = i*m; output[i] = 0;
            for (int j = 0; j < m; ++j) {
                output[i] += X[p[i]]*Y[p[i]];
                p[i]++;
            }
        }
        delete [] p;
        return output;
    }
}

Mat get_transpose(Mat X) {
    Mat XT = creat_Mat(X.nCol, X.nRow);
    mfloat* newVec = XT.vec;
    mfloat* oldVec = X.vec;
    for (int i = 0; i < X.nCol; ++i) {
        int im = i*X.nRow;
        for (int j = 0; j < X.nRow; ++j) {
            newVec[j*X.nCol+i] = oldVec[im+j];
        }
    }
    return XT;
}

void get_transpose(const mfloat* X, mfloat* output, int m, int n) {
    for (int i = 0; i < n; ++i) {
        int im = i*m;
        for (int j = 0; j < m; ++j) {
            output[j*n+i] = X[im+j];
        }
    }
}

void create_My_CG() {
}

void My_CG() {
}

void create_PCG() {
}

void BiCG_test() {
    QP_struct *QP_space = new QP_struct;
    sparseRowMatrix Q;
    Q.value = new mfloat [4];
    Q.nRow = 2; Q.nCol = 2;
    Q.value[0] = 4; Q.value[1] = 2; Q.value[2] = 2; Q.value[3] = 6;
    Q.rowStart = new int [3];
    Q.rowStart[0] = 0; Q.rowStart[1] = 2; Q.rowStart[2] = 4;
    Q.column = new int [4];
    Q.column[0] = 0; Q.column[1] = 1; Q.column[2] = 0; Q.column[3] = 1;

    BiCG_struct *BiCG_space = new BiCG_struct;
    QP_space->Q = Q;

    sparseRowMatrix A;
    A.value = new mfloat [4];
    A.nRow = 2; A.nCol = 2;
    A.value[0] = 1; A.value[1] = 1; A.value[2] = 2; A.value[3] = -1;
    A.rowStart = new int [3];
    A.rowStart[0] = 0; A.rowStart[1] = 2; A.rowStart[2] = 4;
    A.column = new int [4];
    A.column[0] = 0; A.column[1] = 1; A.column[2] = 0; A.column[3] = 1;
    QP_space->A = A;
    QP_space->AT = get_BT(A);
    QP_space->m = 2; QP_space->n = 2;
    QP_space->Qx = new mfloat [2];
    QP_space->ATy = new mfloat [2];
//    sGSADMM_QP_init(QP_space);
    BiCG_space->dim_n = 2;
    BiCG_space->dim_m = 2;
    BiCG_init(BiCG_space);
    mfloat *x0 = new mfloat [2+2];
    x0[0] = 0; x0[1] = 0;
    x0[2] = 0; x0[3] = 0;
    mfloat *rhs = new mfloat [4];
    rhs[0] = 1; rhs[1] = 1; rhs[2] = 1; rhs[3] = 1;
    mfloat *output = new mfloat [4];
    QP_space->tau = 1;
    QP_space->sigma = 1;
    spVec_int U_idx;
    U_idx.number = 2;
    U_idx.vec = new int [2];
    U_idx.vec[0] = 0; U_idx.vec[1] = 1;
    QP_space->U_idx = U_idx;
    BiCG(x0, rhs, output, BiCG_space, QP_space);
    cout << output[0] << ' ' << output[1] << ' ' << output[2] << ' ' << output[3] << endl;
}

void simple_QP_test() {
    QP_struct *QP_space = new QP_struct;
    sparseRowMatrix Q;
    Q.value = new mfloat [4];
    Q.nRow = 2; Q.nCol = 2;
    Q.value[0] = 4; Q.value[1] = 2; Q.value[2] = 2; Q.value[3] = 6;
    Q.rowStart = new int [3];
    Q.rowStart[0] = 0; Q.rowStart[1] = 2; Q.rowStart[2] = 4;
    Q.column = new int [4];
    Q.column[0] = 0; Q.column[1] = 1; Q.column[2] = 0; Q.column[3] = 1;

    QP_space->Q = Q;
    QP_space->c = new mfloat [2];
    QP_space->c[0] = 0; QP_space->c[1] = 0;

    sparseRowMatrix A;
    A.value = new mfloat [4];
    A.nRow = 2; A.nCol = 2;
    A.value[0] = 1; A.value[1] = 1; A.value[2] = 2; A.value[3] = -1;
    A.rowStart = new int [3];
    A.rowStart[0] = 0; A.rowStart[1] = 2; A.rowStart[2] = 4;
    A.column = new int [4];
    A.column[0] = 0; A.column[1] = 1; A.column[2] = 0; A.column[3] = 1;
    QP_space->A = A;

    QP_space->b = new mfloat [2];
    QP_space->b[0] = 1; QP_space->b[1] = 0;

    QP_space->l = new mfloat [2];
    QP_space->l[0] = 0; QP_space->l[1] = 0;
//    QP_space->l[0] = 0; QP_space->l[1] = 0;

    QP_space->u = new mfloat [2];
    QP_space->u[0] = 1; QP_space->u[1] = 1;
//    QP_space->u[0] = 1e300; QP_space->u[1] = 1e300;

    QP_space->m = 2; QP_space->n = 2;

    input_parameters para;
    output_parameters *para_out = new output_parameters;
    para.max_ALM_iter = 10;
    para.max_ADMM_iter = 300;
    para.kappa = 1e-4;
    QP_solve(Q, A, QP_space->b, QP_space->c, QP_space->l, QP_space->u, QP_space->m, QP_space->n, para, para_out);
}


void simple_ruiz_test() {
    sparseRowMatrix A;
    A.value = new mfloat [4];
    A.nRow = 2; A.nCol = 2;
    A.value[0] = 1; A.value[1] = 1; A.value[2] = 2; A.value[3] = -1;
    A.rowStart = new int [3];
    A.rowStart[0] = 0; A.rowStart[1] = 2; A.rowStart[2] = 4;
    A.column = new int [4];
    A.column[0] = 0; A.column[1] = 1; A.column[2] = 0; A.column[3] = 1;

    sparseRowMatrix AT = get_BT(A);

//    sparseRowMatrix *A_new = new sparseRowMatrix;
//    sparseRowMatrix *AT_new = new sparseRowMatrix;

    mfloat *D = new mfloat [2];
    mfloat *E = new mfloat [2];
    ruiz_equilibration(A, &A, AT, &AT, D, E);

    print_spM(A);
    cout << D[0] << ' ' << D[1] << endl;
    cout << E[0] << ' ' << E[1] << endl;
}

void QP_solve(sparseRowMatrix Q, sparseRowMatrix A, mfloat* b, mfloat* c, mfloat* l, mfloat* u, int m, int n, input_parameters para, output_parameters *para_out) {
    QP_struct *QP_space = new QP_struct;
    QP_space->Q = Q;
    QP_space->A = A;
    QP_space->b = b;
    QP_space->c = c;
    QP_space->l = l;
    QP_space->u = u;
    QP_space->m = m;
    QP_space->n = n;
//    QP_space->input_para = para;

    load_para(QP_space, para);
    preprocessing(QP_space);
    sGSADMM_QP_init(QP_space);

    QP_space->time_solve_start = time_now();

    QP_space->iter = 1;
    if (QP_space->max_ADMM_iter > 0)
        sGSADMM_QP(QP_space);

    para_out->iter_ADMM = QP_space->iter;
    para_out->time_ADMM = time_since(QP_space->time_solve_start);

    QP_space->iter = 1;
//    for (int i = 0; i < n; ++i) {
//        QP_space->w_old[i] = QP_space->w[i];
//        QP_space->w0[i] = QP_space->w[i];
//    }
//    for (int i = 0; i < m; ++i) {
//        QP_space->y_old[i] = QP_space->y[i];
//        QP_space->y0[i] = QP_space->y[i];
//    }
    if (QP_space->max_ALM_iter > 0)
        pALM_SSN_QP(QP_space);

    para_out->iter_pALM = QP_space->iter;
    para_out->time_pALM = time_since(QP_space->time_solve_start) - para_out->time_ADMM;
}

void sGSADMM_QP(QP_struct *QP_space) {
    sGSADMM_QP_compute_status(QP_space);

    while (QP_space->iter <= QP_space->max_ADMM_iter) {
        sGSADMM_QP_print_status(QP_space);
        if (QP_space->sGSADMM_finish) break;

        // rescaling
        rescaling(QP_space);

        // reset gamma
        if (((QP_space->iter % 1000 == 1) || (QP_space->iter == QP_space->gamma_reset_start)) && (QP_space->gamma_test)) {
            QP_space->gamma = 1.95;
        }

        // old iterates
        if (QP_space->gamma_test) {
            save_solution(QP_space);
        }

        sGSADMM_QP_update_y_first(QP_space);
//        cout << norm2(QP_space->y, QP_space->m) << endl;
//        cout << norm2(QP_space->Rd, QP_space->n) << endl;
        sGSADMM_QP_update_w_first(QP_space);
//        cout << norm2(QP_space->w, QP_space->n) << endl;
        sGSADMM_QP_update_z(QP_space);
//        cout << norm2(QP_space->z, QP_space->n) << endl;
        sGSADMM_QP_update_w_second(QP_space);
//        cout << norm2(QP_space->w, QP_space->n) << endl;
        sGSADMM_QP_update_y_second(QP_space);
//        cout << norm2(QP_space->y, QP_space->m) << endl;
        sGSADMM_QP_update_x(QP_space);
//        cout << norm2(QP_space->x, QP_space->n) << endl;

        // update workspace
//        if ((QP_space->iter % sGSADMM_sigma_update_iter(QP_space->iter) == 1) || (QP_space->iter % sGSADMM_print_iter(QP_space->iter) == 1))
        sGSADMM_QP_compute_status(QP_space);

        // update gamma
        update_gamma(QP_space);
        // update sigma
        update_sigma(QP_space);

        QP_space->iter++;
    }
}

void sGSADMM_QP_init(QP_struct *QP_space) {
    int n = QP_space->n, m = QP_space->m;
    QP_space->line_search_temp1 = new mfloat [n];
    QP_space->line_search_temp2 = new mfloat [m];
    QP_space->w = new mfloat [n];
    QP_space->w_bar = new mfloat [n];
    QP_space->w_diff = new mfloat [n];
    QP_space->y = new mfloat [m];
    QP_space->y_diff = new mfloat [m];
    QP_space->y_bar = new mfloat [m];
    QP_space->z = new mfloat [n];
    QP_space->temp_z = new mfloat [n];
    QP_space->z_old = new mfloat [n];
    QP_space->zorg = new mfloat [n];
    QP_space->z_new = new mfloat [n];
    QP_space->x = new mfloat [n];
    QP_space->x0 = new mfloat [n];
    QP_space->x_old = new mfloat [n];
    QP_space->xorg = new mfloat [n];
    QP_space->y_old = new mfloat [n];
    QP_space->w_old = new mfloat [n];
    QP_space->y_new = new mfloat [n];
    QP_space->w_new = new mfloat [n];
    QP_space->dQdA = new mfloat [m+n];
    QP_space->Qw = new mfloat [n];
    QP_space->Qw_new = new mfloat [n];
    QP_space->normGrad_list = new mfloat [QP_space->max_iter_SSN+1];
    for (int i = 0; i < n; ++i) {
        QP_space->w[i] = 0; QP_space->z[i] = 0; QP_space->x[i] = 0;
        QP_space->w_old[i] = 0; QP_space->z_old[i] = 0; QP_space->x_old[i] = 0;
        QP_space->w_new[i] = 0; QP_space->x0[i] = 0; QP_space->z_old[i] = 0;
        QP_space->Qw_new[i] = 0;
    }
    for (int i = 0; i < m; ++i) {
        QP_space->y[i] = 0;
        QP_space->y_old[i] = 0;
        QP_space->y_new[i] = 0;
    }
//    QP_space->sigma = 1.0;
    QP_space->update_y_rhs = new mfloat [m];
    QP_space->Qw_diff = new mfloat [n];
    QP_space->Qw_org = new mfloat [n];
    QP_space->Qw_old = new mfloat [n];
    QP_space->Qx = new mfloat [n];
    QP_space->Qx_old = new mfloat [n];
    QP_space->Qx_org = new mfloat [n];
    QP_space->RQ = new mfloat [n];
    QP_space->RQ_org = new mfloat [n];
    QP_space->RC = new mfloat [n];
    QP_space->RCorg = new mfloat [n];
    spMV_R(QP_space->Q, QP_space->w, n, n, QP_space->Qw);
    spMV_R(QP_space->Q, QP_space->x, n, n, QP_space->Qx);
    QP_space->ATy = new mfloat [n];
    QP_space->ATy0 = new mfloat [n];
    QP_space->ATdy = new mfloat [n];
    QP_space->Ax = new mfloat [m];
    QP_space->Rd = new mfloat [n];
    QP_space->Rd_sub = new mfloat [n];
    QP_space->Rd_org = new mfloat [n];
    QP_space->Rp = new mfloat [m];
    QP_space->Rp_sub = new mfloat [m];
    QP_space->Rp_org = new mfloat [m];
    QP_space->z_Qw_c = new mfloat [n];
    QP_space->A_times_z_Qw_c = new mfloat [m];
    QP_space->SSN_compute_obj_temp1 = new mfloat [n];
    QP_space->Qz = new mfloat [n];
    QP_space->SSN_grad = new mfloat [m+n];
    QP_space->SSN_grad_invsigma = new mfloat [m+n];
    QP_space->SSN_grad_Q = new mfloat [m+n];
//    QP_space->Qw_old = new mfloat [n];
    QP_space->Az = new mfloat [n];
    QP_space->dz = new mfloat [n];
    QP_space->dzorg = new mfloat [n];
    QP_space->Qz = new mfloat [n];
    QP_space->dw = new mfloat [n];
    QP_space->SSN_dwdy = new mfloat [m+n];
    QP_space->Qdw = new mfloat [n];
    QP_space->update_w_rhs = new mfloat [n];
    QP_space->proj_idx = new bool [n];
    QP_space->zero_idx = new bool [n];
    QP_space->SSN_rhs = new mfloat [n];
    QP_space->SSN_rhs_compressed = new mfloat [n];
    QP_space->proj_idx_sum = new int [n];
    QP_space->SSN_sub_case1_temp1 = new mfloat [n];
    QP_space->SSN_sub_case1_temp2 = new mfloat [m];
    QP_space->SSN_sub_case1_temp3 = new mfloat [m];
    QP_space->SSN_r1_new = new mfloat [n];
    QP_space->U_idx.vec = new int [n];
    QP_space->z_ATy_c = new mfloat [n];
    QP_space->Qw_ATy_c = new mfloat [n];
    QP_space->update_z_unProjected = new mfloat [n];
    QP_space->update_z_projected = new mfloat [n];
    QP_space->pALM_z_unProjected = new mfloat [n];
    QP_space->pALM_z_projected = new mfloat [n];
    QP_space->normQ = norm2(QP_space->Q.value, QP_space->Q.rowStart[QP_space->Q.nRow]);
    QP_space->iter = 1;
//    QP_space->kappa = 1; // [1e-4, 1e2] depends on problem.
    QP_space->sGSADMM_tol = 1e-4;
    QP_space->sGSADMM_finish = false;


//    else
//        QP_space->sigma = 1;


    QP_space->diag_AAT = new mfloat [m];
    for (int i = 0; i < m; ++i) {
        QP_space->diag_AAT[i] = 0;
        for (int j = QP_space->A.rowStart[i]; j < QP_space->A.rowStart[i+1]; ++j) {
            QP_space->diag_AAT[i] += QP_space->A.value[j]*QP_space->A.value[j];
        }
    }

    QP_space->diag_Q = new mfloat [n];
    for (int i = 0; i < n; ++i) {
        for (int j = QP_space->Q.rowStart[i]; j < QP_space->Q.rowStart[i+1]; ++j) {
            if (QP_space->Q.column[j] == i) {
                QP_space->diag_Q[i] = QP_space->Q.value[j];
                break;
            }
        }
    }
    if (!QP_space->use_mkl)
        Eigen_init(QP_space);
    else
        MKL_init(QP_space);

    if (QP_space->SSN_method == 1) {
        QP_space->PDS_Q2 = new PARDISO_var;
        QP_space->PDS_Q2->x = new mfloat [n];
//        QP_space->PDS_Q2->b = new mfloat [n];
//        sparseRowMatrix Q_upper;
//        upper_sparseRowMatrix(QP_space->Q, &QP_space->Q_upper);


//        QP_space->tripletList_Qsub.reserve(QP_space->Q.rowStart[QP_space->Q.nRow]);
//        CSR_to_COO(QP_space->Q, &QP_space->Q_coo);
//        QP_space->Q_sub_counter = new int [QP_space->Q.rowStart[QP_space->Q.nRow]];
    }

//    Eigen::SparseMatrix<mfloat> EigenA = get_eigen_spMat(QP_space->A);
//    Eigen::SparseMatrix<mfloat> EigenAT = get_eigen_spMat(QP_space->AT);
//    Eigen::SparseMatrix<mfloat> EigenAAT = EigenA.transpose()*EigenA;
//
//    cout << EigenA << endl;
//    QP_space->EigenQ = get_eigen_spMat(QP_space->Q);
//    QP_space->EigenI.resize(n, n);
//    QP_space->EigenI.setIdentity();
//    QP_space->EigenIQ = 1/QP_space->sigma*QP_space->EigenI + QP_space->EigenQ;
//    EigenQ = 1/QP_space->sigma*EigenQ + EigenQ*EigenQ;


//    cout << EigenA << endl;

//    cout << EigenAAT << endl;

    // Eigen is using column major
//    Eigen::SparseMatrix<double> upperMatAAT = EigenAAT.triangularView<Eigen::Lower>();
////    upperMat = upperMat.transpose();
//
//    QP_space->AAT.nRow = m; QP_space->AAT.nCol = m;
//    int nnz = upperMatAAT.nonZeros();
//    QP_space->AAT.rowStart = new int [m+1];
//    QP_space->AAT.column = new int [nnz];
//    QP_space->AAT.value = new mfloat [nnz];
//    vMemcpy(upperMatAAT.outerIndexPtr(), m+1, QP_space->AAT.rowStart);
////    QP_space->AAT.rowStart[1] = 2;
////    cout << QP_space->AAT.rowStart[0] << QP_space->AAT.rowStart[1] << QP_space->AAT.rowStart[2] << endl;
//    vMemcpy(upperMatAAT.innerIndexPtr(), upperMatAAT.nonZeros(), QP_space->AAT.column);
////    cout << QP_space->AAT.column[0] << QP_space->AAT.column[1] << QP_space->AAT.column[2] << endl;
////    QP_space->AAT.column[1] = 1;
//    vMemcpy(upperMatAAT.valuePtr(), upperMatAAT.nonZeros(), QP_space->AAT.value);
////    cout << QP_space->AAT.value[0] << QP_space->AAT.value[1] << QP_space->AAT.value[2] << endl;
//
//    cout << upperMatAAT << endl;
//    QP_space->PDS_A = new PARDISO_var;
//    PARDISO_init(QP_space->PDS_A, QP_space->AAT);
//    PARDISO_numerical_fact(QP_space->PDS_A);
//
//
//    QP_space->EigenIQ = 1/QP_space->sigma*QP_space->EigenI + QP_space->EigenQ;
//    QP_space->upperMatQ = QP_space->EigenIQ.triangularView<Eigen::Lower>();
//    cout << QP_space->upperMatQ << endl;
//    QP_space->IQ.nRow = n; QP_space->IQ.nCol = n;
//    QP_space->IQ.rowStart = QP_space->upperMatQ.outerIndexPtr();
//    QP_space->IQ.column = QP_space->upperMatQ.innerIndexPtr();
//    QP_space->IQ.value = QP_space->upperMatQ.valuePtr();
//    QP_space->PDS_IQ = new PARDISO_var;
//    PARDISO_init(QP_space->PDS_IQ, QP_space->IQ);
//    PARDISO_numerical_fact(QP_space->PDS_IQ);
}

void rescaling(QP_struct *qp) {
    int iter = qp->iter;
    if ((max(qp->inf_p_org, qp->inf_d_org) > 1e-2) && (iter < 500)) {
        if (qp->rescale >= 1) {
            if ((iter == 10) || ((iter % 53 == 0) && (iter >100) && (iter < 200)))
                rescalingADMM(qp);
        }
        if (qp->rescale >= 2) {
            if ((iter % 203 == 0) && (iter >200) && (iter < 1000))
                rescalingADMM(qp);
        }
        if (qp->rescale >= 3) {
            if ((iter % 503 == 0) && (iter >1000) && (iter < 10000))
                rescalingADMM(qp);
        }
    }
}

void rescalingADMM(QP_struct *qp) {
    int m = qp->m, n = qp->n;
    mfloat normP = norm2(qp->x, n);
    mfloat normD = max(norm2(qp->z, n), distance(qp->Rd, qp->z, n));
    mfloat PD_ratio = max(normP/normD, normD/normP);
    if (PD_ratio > 1.1) {
        mfloat bscale2 = normP;
        mfloat cscale2 = normD;
        mfloat inv_bscale2 = 1/bscale2;
        mfloat inv_cscale2 = 1/cscale2;
        ax(inv_bscale2, qp->b, qp->b, m);
        ax(inv_cscale2, qp->c, qp->c, n);
        ax(inv_bscale2, qp->l, qp->l, n);
        ax(inv_bscale2, qp->u, qp->u, n);
        qp->normb /= bscale2;
//        qp->normb = norm2(qp->b, m);
//        cout << "normb: " << qp->normb << endl;
//        qp->normc = norm2(qp->c, n);
        qp->normc /= cscale2;
//        cout << "normc: " << qp->normc << endl;

        ax(inv_bscale2, qp->x, qp->x, n);
        ax(inv_cscale2, qp->z, qp->z, n);
        ax(inv_cscale2, qp->y, qp->y, m);
        ax(inv_bscale2, qp->w, qp->w, n);
        ax(inv_cscale2, qp->Qx, qp->Qx, n);
        ax(inv_cscale2, qp->Qw, qp->Qw, n);

        qp->bscale *= bscale2;
        qp->cscale *= cscale2;

        qp->sigma *= cscale2/bscale2;
        mfloat bc_ratio = bscale2 / cscale2;
        for (int r = 0; r < qp->n; ++r) {
            for (int j = qp->Q.rowStart[r]; j < qp->Q.rowStart[r+1]; ++j) {
                if (qp->use_mkl && (qp->Q.column[j] == r))
                    // recompute the diagonal elements of Q
                    qp->MKL_IQ.value[qp->MKL_IQ.rowStart[r]] = qp->Q.value[j];
                qp->Q.value[j] *= bc_ratio;
            }
        }
        qp->EigenQ *= bc_ratio;
        if (!qp->use_mkl) {
            qp->EigenQ = get_eigen_spMat(qp->Q);
            sGSADMM_QP_refact_IQ(qp);
        } else {
            for (int r = 0; r < qp->n; ++r) {
                for (int j = qp->MKL_IQ.rowStart[r]; j < qp->MKL_IQ.rowStart[r+1]; ++j) {
                    qp->MKL_IQ.value[j] *= bc_ratio;
                }
            }
            mfloat inv_sigma_old = 1/qp->sigma_old;
            mfloat inv_sigma = 1/qp->sigma;

            int *rowStart = qp->MKL_IQ.rowStart;
            mfloat *value = qp->MKL_IQ.value;
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(bc_ratio, n, rowStart, value, inv_sigma, inv_sigma_old)
            for (int i = 0; i < n; ++i) {
//                value[rowStart[i]] -= inv_sigma_old*bc_ratio;
                value[rowStart[i]] += inv_sigma;
            }
            PARDISO_numerical_fact(qp->PDS_IQ);
        }
        sGSADMM_QP_compute_status(qp);

        printf("rescale = %d: %d, %3.2e, %3.2e\n", qp->rescale, qp->iter, normP, normD);
        qp->rescale++;
        qp->prim_win = 0;
        qp->dual_win = 0;
    }
}

void sGSADMM_QP_update_y_first(QP_struct *QP_space) {
    mfloat *rhs = QP_space->update_y_rhs;
    mfloat *z_Qw_c = QP_space->z_Qw_c;
    int m = QP_space->m, n = QP_space->n;

    axpy(-1, QP_space->ATy, QP_space->Rd, QP_space->z_Qw_c, n);
    spMV_R(QP_space->A, z_Qw_c, m, n, QP_space->A_times_z_Qw_c);
    axpby(-1, QP_space->A_times_z_Qw_c, 1/QP_space->sigma, QP_space->Rp, rhs, m);

//    spMV_R(QP_space->A, QP_space->x, m, n, QP_space->Ax);
//    axpy(-1, QP_space->Ax, QP_space->b, rhs, m);
//    ax(1/QP_space->sigma, rhs, rhs, m);
//
//    axpy(-1, QP_space->Qw, QP_space->z, z_Qw_c, n);
//    axpy(-1, QP_space->c, z_Qw_c, z_Qw_c, n);
//    spMV_R(QP_space->A, z_Qw_c, m, n, QP_space->A_times_z_Qw_c);
//
//    axpy(-1, QP_space->A_times_z_Qw_c, rhs, rhs, m);

    if (!QP_space->use_mkl) {
        vMemcpy(rhs, m, QP_space->Eigen_rhs_y.data());
        QP_space->Eigen_result_y = QP_space->Eigen_linear_AAT.LLTsolver.solve(QP_space->Eigen_rhs_y);
        vMemcpy(QP_space->Eigen_result_y.data(), m, QP_space->y);
    } else {
        PARDISO_solve(QP_space->PDS_A, rhs);
        vMemcpy(QP_space->PDS_A->x, m, QP_space->y);
//        vMemcpy(rhs, m, QP_space->y);
    }


    spMV_R(QP_space->AT, QP_space->y, n, m, QP_space->ATy);
    axpy(1, QP_space->ATy, QP_space->z_Qw_c, QP_space->Rd, n);

//    cout << norm2(QP_space->ATy, n) << endl;
//    cout << norm2(z_Qw_c, n) << endl;
//            PARDISO_solve(QP_space->PDS_A, rhs);
//    vMemcpy(QP_space->PDS_A->x, n, QP_space->y_bar);
}

void sGSADMM_QP_update_y_second(QP_struct *QP_space) {
//    mfloat *rhs = QP_space->update_y_rhs;
//    mfloat *z_Qw_c = QP_space->z_Qw_c;
//    int m = QP_space->m, n = QP_space->n;
//
//    axpy(-1, QP_space->Ax, QP_space->b, rhs, m);
//    ax(1/QP_space->sigma, rhs, rhs, m);
//
//    spMV_R(QP_space->Q, QP_space->w, n, n, QP_space->Qw);
//    axpy(-1, QP_space->Qw, QP_space->z, z_Qw_c, n);
//    axpy(-1, QP_space->c, z_Qw_c, z_Qw_c, n);
//    spMV_R(QP_space->A, z_Qw_c, m, n, QP_space->A_times_z_Qw_c);
//
//    axpy(-1, QP_space->A_times_z_Qw_c, rhs, rhs, m);
//
//
//    vMemcpy(rhs, m, QP_space->Eigen_rhs_y.data());
//    QP_space->Eigen_result_y = QP_space->Eigen_linear_AAT.LLTsolver.solve(QP_space->Eigen_rhs_y);
//    vMemcpy(QP_space->Eigen_result_y.data(), m, QP_space->y);


    mfloat *rhs = QP_space->update_y_rhs;
    mfloat *z_Qw_c = QP_space->z_Qw_c;
    int m = QP_space->m, n = QP_space->n;

    axpy(-1, QP_space->ATy, QP_space->Rd, QP_space->z_Qw_c, n);
    spMV_R(QP_space->A, z_Qw_c, m, n, QP_space->A_times_z_Qw_c);
    axpby(-1, QP_space->A_times_z_Qw_c, 1/QP_space->sigma, QP_space->Rp, rhs, m);


    if (!QP_space->use_mkl) {
        vMemcpy(rhs, m, QP_space->Eigen_rhs_y.data());
        QP_space->Eigen_result_y = QP_space->Eigen_linear_AAT.LLTsolver.solve(QP_space->Eigen_rhs_y);
        vMemcpy(QP_space->Eigen_result_y.data(), m, QP_space->y);
    } else {
        PARDISO_solve(QP_space->PDS_A, rhs);
        vMemcpy(QP_space->PDS_A->x, m, QP_space->y);
//        vMemcpy(rhs, m, QP_space->y);
    }

    spMV_R(QP_space->AT, QP_space->y, n, m, QP_space->ATy);
    axpy(1, QP_space->ATy, QP_space->z_Qw_c, QP_space->Rd, n);

}

void sGSADMM_QP_update_w_first(QP_struct *QP_space) {
    mfloat *rhs = QP_space->update_w_rhs;
    mfloat *z_ATy_c = QP_space->z_ATy_c;
    int m = QP_space->m, n = QP_space->n;
//    mfloat *rhs2 = new mfloat [n];

    axpy(1, QP_space->Qw, QP_space->Rd, z_ATy_c, n);
    axpy(1/QP_space->sigma, QP_space->x, z_ATy_c, rhs, n);

//    spMV_R(QP_space->AT, QP_space->y_bar, n, m, QP_space->ATy);
//    axpy(1, QP_space->ATy, QP_space->z, z_ATy_c, n);
//    axpy(-1, QP_space->c, z_ATy_c, z_ATy_c, n);
//
//    axpy(1/QP_space->sigma, QP_space->x, QP_space->z_ATy_c, rhs, n);

//    cout << "w rhs " << norm2(rhs, n) << endl;
    if (!QP_space->use_mkl) {
        vMemcpy(rhs, n, QP_space->Eigen_rhs_w.data());
        QP_space->Eigen_result_w = QP_space->Eigen_linear_IQ.LLTsolver.solve(QP_space->Eigen_rhs_w);
        vMemcpy(QP_space->Eigen_result_w.data(), n, QP_space->w);
    } else {
        PARDISO_solve(QP_space->PDS_IQ, rhs);
        vMemcpy(QP_space->PDS_IQ->x, n, QP_space->w);
    }

    spMV_R(QP_space->Q, QP_space->w, n, n, QP_space->Qw);

    axpy(-1, QP_space->Qw, QP_space->z_ATy_c, QP_space->Rd, n);


//    QP_space->Eigen_linear_IQ.LLTsolver.solve(rhs);
//    vMemcpy(QP_space->PDS_IQ->x, n, QP_space->w_bar);
}

void sGSADMM_QP_update_w_second(QP_struct *QP_space) {
//    mfloat *rhs = QP_space->update_w_rhs;
////    mfloat *z_ATy_c = QP_space->z_ATy_c;
//    int n = QP_space->n;
//
////    spMV_R(QP_space->Q, QP_space->dz, n, n, QP_space->Qdz);
//    axpy(1, QP_space->dz, rhs, rhs, n);
//
//
//    vMemcpy(rhs, n, QP_space->Eigen_rhs_w.data());
//    QP_space->Eigen_result_w = QP_space->Eigen_linear_IQ.LLTsolver.solve(QP_space->Eigen_rhs_w);
//    vMemcpy(QP_space->Eigen_result_w.data(), n, QP_space->w);

    mfloat *rhs = QP_space->update_w_rhs;
    mfloat *z_ATy_c = QP_space->z_ATy_c;
    int m = QP_space->m, n = QP_space->n;
//    mfloat *rhs2 = new mfloat [n];

    axpy(1, QP_space->Qw, QP_space->Rd, z_ATy_c, n);
    axpy(1/QP_space->sigma, QP_space->x, z_ATy_c, rhs, n);

//    spMV_R(QP_space->AT, QP_space->y_bar, n, m, QP_space->ATy);
//    axpy(1, QP_space->ATy, QP_space->z, z_ATy_c, n);
//    axpy(-1, QP_space->c, z_ATy_c, z_ATy_c, n);
//
//    axpy(1/QP_space->sigma, QP_space->x, QP_space->z_ATy_c, rhs, n);


    if (!QP_space->use_mkl) {
        vMemcpy(rhs, n, QP_space->Eigen_rhs_w.data());
        QP_space->Eigen_result_w = QP_space->Eigen_linear_IQ.LLTsolver.solve(QP_space->Eigen_rhs_w);
        vMemcpy(QP_space->Eigen_result_w.data(), n, QP_space->w);
    } else {
        PARDISO_solve(QP_space->PDS_IQ, rhs);
        vMemcpy(QP_space->PDS_IQ->x, n, QP_space->w);
    }


    spMV_R(QP_space->Q, QP_space->w, n, n, QP_space->Qw);

    axpy(-1, QP_space->Qw, QP_space->z_ATy_c, QP_space->Rd, n);

//    PARDISO_solve(QP_space->PDS_IQ, rhs);
//    vMemcpy(QP_space->PDS_IQ->x, n, QP_space->w);
}

void sGSADMM_QP_update_z(QP_struct *QP_space) {
    mfloat *unProjected = QP_space->update_z_unProjected;
    mfloat *projected = QP_space->update_z_projected;
    int n = QP_space->n;

    axpy(-1, QP_space->z, QP_space->Rd, QP_space->temp_z, n);
    axpy(QP_space->sigma, QP_space->temp_z, QP_space->x, unProjected, n);
//
//    spMV_R(QP_space->Q, QP_space->w_bar, n, n, QP_space->Qw);
//    axpy(-1, QP_space->Qw, QP_space->ATy, unProjected, n);
//    axpy(-1, QP_space->c, unProjected, unProjected, n);
//    axpy(QP_space->sigma, unProjected, QP_space->x, unProjected, n);

    proj(unProjected, projected, n, QP_space->l, QP_space->u);
//    axpy(-1, unProjected, projected, QP_space->z_new, n);
//    ax(1/QP_space->sigma, QP_space->z_new, QP_space->z_new, n);
//
//    axpy(-1, QP_space->z, QP_space->z_new, QP_space->dz, n);
//    vMemcpy(QP_space->z_new, n, QP_space->z);
    axpy(-1, unProjected, projected, QP_space->z, n);
    ax(1/QP_space->sigma, QP_space->z, QP_space->z, n);

    axpy(1, QP_space->temp_z, QP_space->z, QP_space->Rd, n);
}

void sGSADMM_QP_update_x(QP_struct *QP_space) {
    mfloat gamma = QP_space->gamma;
    mfloat sigma = QP_space->sigma;
    int n = QP_space->n;
    int m = QP_space->m;
    spMV_R(QP_space->AT, QP_space->y, n, m, QP_space->ATy);
    axpy(gamma*sigma, QP_space->Rd, QP_space->x, QP_space->x, n);
//    axpy(gamma*sigma, QP_space->ATy, QP_space->x, QP_space->x, n);
}

void sGSADMM_QP_compute_status(QP_struct *QP_space) {
    int m = QP_space->m, n = QP_space->n;

    // necessary for both org and after-scaling
    spMV_R(QP_space->A, QP_space->x, m, n, QP_space->Ax);
    axpy(-1, QP_space->Ax, QP_space->b, QP_space->Rp, m);

    axpy(-1, QP_space->Qw, QP_space->z, QP_space->Rd, n);
    axpy(-1, QP_space->c, QP_space->Rd, QP_space->Rd, n);
    spMV_R(QP_space->AT, QP_space->y, n, m, QP_space->ATy);
    axpy(1, QP_space->ATy, QP_space->Rd, QP_space->Rd, n);

    spMV_R(QP_space->Q, QP_space->x, n, n, QP_space->Qx);

    QP_space->primal_obj = xTy(QP_space->x, QP_space->Qx, n)/2 + xTy(QP_space->c, QP_space->x, n);
    QP_space->dual_obj = xTy(QP_space->x, QP_space->z, n);
    QP_space->dual_obj += -xTy(QP_space->w, QP_space->Qw, n)/2 + xTy(QP_space->b, QP_space->y, m);

    // only for after-scaling, update sigma use
//    if (QP_space->iter % sGSADMM_sigma_update_iter(QP_space->iter) == 1) {
//    if (QP_space->iter % QP_space->sigma_update_iter == 1) {
    if (1) {
        axpy(-1, QP_space->Qw, QP_space->Qx, QP_space->RQ, n);

        axpy(-1, QP_space->z, QP_space->x, QP_space->dz, n);
        proj(QP_space->dz, QP_space->update_z_projected, n, QP_space->l, QP_space->u);
        axpy(-1, QP_space->update_z_projected, QP_space->x, QP_space->RC, n);

        QP_space->inf_p = norm2(QP_space->Rp, m)/(1 + QP_space->normb);
        QP_space->inf_d = norm2(QP_space->Rd, n)/(1 + QP_space->normc);
        QP_space->inf_Q = norm2(QP_space->RQ, n)/(1 + norm2(QP_space->Qx, n) + norm2(QP_space->Qw, n));
        QP_space->inf_C = norm2(QP_space->RC, n)/(1 + norm2(QP_space->x, n) + norm2(QP_space->z, n));
        QP_space->inf_g = abs(QP_space->primal_obj - QP_space->dual_obj)/(1+abs(QP_space->primal_obj)+abs(QP_space->dual_obj));
    }

    // orginal residuals, only when print
//    if (QP_space->iter % sGSADMM_print_iter(QP_space->iter) == 1) {
    if (1) {
        xdoty(QP_space->x, QP_space->cA, QP_space->xorg, n, true);
        ax(QP_space->bscale, QP_space->xorg, QP_space->xorg, n);
        xdoty(QP_space->z, QP_space->cA, QP_space->zorg, n, false);
        ax(QP_space->cscale, QP_space->zorg, QP_space->zorg, n);

        axpy(-1, QP_space->zorg, QP_space->xorg, QP_space->dzorg, n);
        proj(QP_space->dzorg, QP_space->update_z_projected, n, QP_space->Lorg, QP_space->Uorg);
        axpy(-1, QP_space->update_z_projected, QP_space->xorg, QP_space->RCorg, n);
        QP_space->inf_C_org = norm2(QP_space->RCorg, n)/(1 + norm2(QP_space->xorg, n) + norm2(QP_space->zorg, n));


        xdoty(QP_space->Rp, QP_space->rA, QP_space->Rp_org, m, false);
        ax(QP_space->bscale, QP_space->Rp_org, QP_space->Rp_org, m);
        QP_space->inf_p_org = norm2(QP_space->Rp_org, m)/(1 + QP_space->normborg);

        xdoty(QP_space->Rd, QP_space->cA, QP_space->Rd_org, n, false);
        ax(QP_space->cscale, QP_space->Rd_org, QP_space->Rd_org, n);
        QP_space->inf_d_org = norm2(QP_space->Rd_org, n)/(1 + QP_space->normcorg);

        xdoty(QP_space->Qx, QP_space->cA, QP_space->Qx_org, n, false);
        ax(QP_space->cscale, QP_space->Qx_org, QP_space->Qx_org, n);
        xdoty(QP_space->Qw, QP_space->cA, QP_space->Qw_org, n, false);
        ax(QP_space->cscale, QP_space->Qw_org, QP_space->Qw_org, n);
        axpy(-1, QP_space->Qw_org, QP_space->Qx_org, QP_space->RQ_org, n);
        QP_space->inf_Q_org = norm2(QP_space->RQ_org, n)/(1 + norm2(QP_space->Qx_org, n) + norm2(QP_space->Qw_org, n));

        QP_space->primal_obj_org = QP_space->primal_obj * QP_space->bscale * QP_space->cscale;
        QP_space->dual_obj_org = QP_space->dual_obj * QP_space->bscale * QP_space->cscale;
        QP_space->inf_g_org = abs(QP_space->primal_obj_org - QP_space->dual_obj_org)/(1+abs(QP_space->primal_obj_org)+abs(QP_space->dual_obj_org));
    }

}

void pALM_QP_compute_status(QP_struct *QP_space) {
    int m = QP_space->m, n = QP_space->n;
    spMV_R(QP_space->A, QP_space->x, m, n, QP_space->Ax);
    axpy(-1, QP_space->Ax, QP_space->b, QP_space->Rp, m);
    QP_space->inf_p = norm2(QP_space->Rp, m)/(1 + QP_space->normb);

    axpy(-1, QP_space->Qw, QP_space->z, QP_space->Rd, n);
    axpy(-1, QP_space->c, QP_space->Rd, QP_space->Rd, n);
    spMV_R(QP_space->AT, QP_space->y, n, m, QP_space->ATy);
    axpy(1, QP_space->ATy, QP_space->Rd, QP_space->Rd, n);
    QP_space->inf_d = norm2(QP_space->Rd, n)/(1 + QP_space->normc);

    spMV_R(QP_space->Q, QP_space->x, n, n, QP_space->Qx);
    axpy(-1, QP_space->Qw, QP_space->Qx, QP_space->RQ, n);
    QP_space->inf_Q = norm2(QP_space->RQ, n)/(1 + norm2(QP_space->Qx, n) + norm2(QP_space->Qw, n));

    axpy(-1, QP_space->z, QP_space->x, QP_space->dz, n);
    proj(QP_space->dz, QP_space->update_z_projected, n, QP_space->l, QP_space->u);
    axpy(-1, QP_space->update_z_projected, QP_space->x, QP_space->RC, n);
    QP_space->inf_C = norm2(QP_space->RC, n)/(1 + norm2(QP_space->x, n) + norm2(QP_space->z, n));

    QP_space->primal_obj = xTy(QP_space->x, QP_space->Qx, n)/2 + xTy(QP_space->c, QP_space->x, n);
    QP_space->dual_obj = xTy(QP_space->x, QP_space->z, n);
    QP_space->dual_obj += -xTy(QP_space->w, QP_space->Qw, n)/2 + xTy(QP_space->b, QP_space->y, m);
    QP_space->inf_g = abs(QP_space->primal_obj - QP_space->dual_obj)/(1+abs(QP_space->primal_obj)+abs(QP_space->dual_obj));


    // orginal residuals, only when print
    xdoty(QP_space->x, QP_space->cA, QP_space->xorg, n, true);
    ax(QP_space->bscale, QP_space->xorg, QP_space->xorg, n);
    xdoty(QP_space->z, QP_space->cA, QP_space->zorg, n, false);
    ax(QP_space->cscale, QP_space->zorg, QP_space->zorg, n);

    axpy(-1, QP_space->zorg, QP_space->xorg, QP_space->dzorg, n);
    proj(QP_space->dzorg, QP_space->update_z_projected, n, QP_space->Lorg, QP_space->Uorg);
    axpy(-1, QP_space->update_z_projected, QP_space->xorg, QP_space->RCorg, n);
    QP_space->inf_C_org = norm2(QP_space->RCorg, n)/(1 + norm2(QP_space->xorg, n) + norm2(QP_space->zorg, n));


    xdoty(QP_space->Rp, QP_space->rA, QP_space->Rp_org, m, false);
    ax(QP_space->bscale, QP_space->Rp_org, QP_space->Rp_org, m);
    QP_space->inf_p_org = norm2(QP_space->Rp_org, m)/(1 + QP_space->normborg);

    xdoty(QP_space->Rd, QP_space->cA, QP_space->Rd_org, n, false);
    ax(QP_space->cscale, QP_space->Rd_org, QP_space->Rd_org, n);
    QP_space->inf_d_org = norm2(QP_space->Rd_org, n)/(1 + QP_space->normcorg);

    xdoty(QP_space->Qx, QP_space->cA, QP_space->Qx_org, n, false);
    ax(QP_space->cscale, QP_space->Qx_org, QP_space->Qx_org, n);
    xdoty(QP_space->Qw, QP_space->cA, QP_space->Qw_org, n, false);
    ax(QP_space->cscale, QP_space->Qw_org, QP_space->Qw_org, n);
    axpy(-1, QP_space->Qw_org, QP_space->Qx_org, QP_space->RQ_org, n);
    QP_space->inf_Q_org = norm2(QP_space->RQ_org, n)/(1 + norm2(QP_space->Qx_org, n) + norm2(QP_space->Qw_org, n));

    QP_space->primal_obj_org = QP_space->primal_obj * QP_space->bscale * QP_space->cscale;
    QP_space->dual_obj_org = QP_space->dual_obj * QP_space->bscale * QP_space->cscale;
    QP_space->inf_g_org = abs(QP_space->primal_obj_org - QP_space->dual_obj_org)/(1+abs(QP_space->primal_obj_org)+abs(QP_space->dual_obj_org));
}

void sGSADMM_QP_refact_IQ(QP_struct* QP_space) {
//    QP_space->EigenIQ = 1/QP_space->sigma*QP_space->EigenI + QP_space->EigenQ;
//    QP_space->upperMatQ = QP_space->EigenIQ.triangularView<Eigen::Lower>();
//    cout << upperMatQ << endl;
//    QP_space->IQ.nRow = QP_space->n; QP_space->IQ.nCol = QP_space->n;
//    QP_space->IQ.rowStart = QP_space->upperMatQ.outerIndexPtr();
//    QP_space->IQ.column = QP_space->upperMatQ.innerIndexPtr();
//    QP_space->IQ.value = QP_space->upperMatQ.valuePtr();

//    PARDISO_init(QP_space->PDS_IQ, QP_space->IQ);
//    vMemcpy(QP_space->IQ.value, QP_space->EigenQ.nonZeros(), QP_space->PDS_IQ->a);

//    cout << QP_space->PDS_IQ->a[0] << ' ' << QP_space->PDS_IQ->a[1] << ' ' << QP_space->PDS_IQ->a[2] << endl;
//    cout << QP_space->upperMatQ << endl;
//    QP_space->PDS_IQ->a = QP_space->IQ.value;
//    QP_space->PDS_IQ->ia = QP_space->IQ.rowStart;
//    QP_space->PDS_IQ->ja = QP_space->IQ.column;
//    mfloat inv_sigma_old = 1/QP_space->sigma_old;
//    mfloat inv_sigma = 1/QP_space->sigma;
//    QP_space->EigenIQ -= inv_sigma_old*QP_space->EigenI;
//    QP_space->EigenIQ += inv_sigma*QP_space->EigenI;
//    for (int i = 0; i < QP_space->n; ++i) {
//        QP_space->PDS_IQ->a[QP_space->PDS_IQ->ia[i]-1] += -inv_sigma_old + inv_sigma;

//    }
//    QP_space->EigenI.resize(QP_space->n, QP_space->n);
//    QP_space->EigenI.setIdentity();
    QP_space->EigenIQ = 1/QP_space->sigma*QP_space->EigenIn + QP_space->EigenQ;
    QP_space->Eigen_linear_IQ.LLTsolver.factorize(QP_space->EigenIQ);
//    PARDISO_numerical_fact(QP_space->PDS_IQ);
}

void MKL_refact(QP_struct* qp) {
    mfloat inv_sigma_old = 1/qp->sigma_old;
    mfloat inv_sigma = 1/qp->sigma;

    int n = qp->n;
    int *rowStart = qp->MKL_IQ.rowStart;
    mfloat *value = qp->MKL_IQ.value;
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(n, rowStart, value, inv_sigma, inv_sigma_old)
    for (int i = 0; i < n; ++i) {
        value[rowStart[i]] -= inv_sigma_old;
        value[rowStart[i]] += inv_sigma;
    }
    PARDISO_numerical_fact(qp->PDS_IQ);
}

int sGSADMM_print_iter(int iter) {
//    return 1;
    if (iter <= 200)
        return 50;
    else if (iter <= 2000)
        return 100;
    else
        return 500;
}

int pALM_print_iter(int iter) {
    return 1;
}


void sGSADMM_QP_print_status(QP_struct* QP_space) {
    if (QP_space->iter == 1) {
        printf("iter    obj_p          obj_d           inf_g     inf_p      inf_d       inf_Q      inf_C      sigma       tau      time\n");
    }
    QP_space->inf = {QP_space->inf_p_org, QP_space->inf_d_org, min(max(QP_space->inf_Q_org, QP_space->inf_C_org), QP_space->inf_g_org)};
    if (*max_element(QP_space->inf.begin(), QP_space->inf.end()) < QP_space->sGSADMM_tol) {
        QP_space->sGSADMM_finish = true;
        cout << "sGSADMM error is " << *max_element(QP_space->inf.begin(), QP_space->inf.end()) << endl;
    }
    mfloat time_solve_now = time_since(QP_space->time_solve_start);
    if ((QP_space->iter % sGSADMM_print_iter(QP_space->iter) == 1) || (QP_space->sGSADMM_finish) || (QP_space->iter == QP_space->max_ADMM_iter)) {
//    if (1) {
        printf("%4d    %7.6e   %7.6e    %2.1e    %2.1e    %2.1e    %2.1e    %2.1e    %2.1e    %2.1e    %3.2f\n", QP_space->iter, QP_space->primal_obj_org, QP_space->dual_obj_org, QP_space->inf_g_org, QP_space->inf_p_org, QP_space->inf_d_org, QP_space->inf_Q_org, QP_space->inf_C_org, QP_space->sigma, QP_space->tau, time_solve_now);
    }

}

void pALM_QP_print_status(QP_struct* QP_space) {
    if (QP_space->iter == 1) {
        printf("iter    obj_p          obj_d           inf_g     inf_p      inf_d       inf_Q      inf_C      sigma       tau      time\n");
    }
    std::vector<mfloat> inf;
    mfloat time_solve_now = time_since(QP_space->time_solve_start);
    if (QP_space->iter % pALM_print_iter(QP_space->iter) == 0) {
        printf("%4d    %7.6e   %7.6e    %2.1e    %2.1e    %2.1e    %2.1e    %2.1e    %2.1e    %2.1e    %3.2f\n", QP_space->iter, QP_space->primal_obj_org, QP_space->dual_obj_org, QP_space->inf_g_org, QP_space->inf_p_org, QP_space->inf_d_org, QP_space->inf_Q_org, QP_space->inf_C_org, QP_space->sigma, QP_space->tau, time_solve_now);
        inf = {QP_space->inf_p_org, QP_space->inf_d_org, min(max(QP_space->inf_Q_org, QP_space->inf_C_org), QP_space->inf_g_org)};
        if (*max_element(inf.begin(), inf.end()) < QP_space->pALM_tol) {
            QP_space->pALM_finish = true;
            cout << "pALM error is " << *max_element(inf.begin(), inf.end()) << endl;
            cout << "case1 time = " << QP_space->case1_time << endl;
            cout << "get_eigen_sub_time = " << QP_space->get_eigen_sub_time << endl;
            cout << "partial_CR_spMV_time = " << QP_space->partial_CR_spMV_time << endl;
            cout << "partial_axpy_time = " << QP_space->partial_axpy_time << endl;
            cout << "axpy_time = " << QP_space->axpy_time << endl;
            cout << "eigen_matrix_mult_time = " << QP_space->eigen_matrix_mult_time << endl;
            cout << "eigen_solver_time = " << QP_space->eigen_solver_time << endl;
            cout << "solver_analyze_time = " << QP_space->solver_analyze_time << endl;
            cout << "solver_factorize_time = " << QP_space->solver_factorize_time << endl;
            cout << "solver_solve_time = " << QP_space->solver_solve_time << endl;
            cout << "factorize_times = " << QP_space->fact_times << endl;
        }

    }
}

mfloat support_function(const mfloat* x, const mfloat* l, const mfloat* u, int len, mfloat* infty_x) {
    mfloat res = 0;
    *infty_x = 0;
    for (int i = 0; i < len; ++i) {
        if (x[i] < 0) {
            // -x > 0
            if (u[i] > 1e30) *infty_x += x[i]*x[i];
            else res += -x[i] * u[i];
        } else {
            if (l[i] < -1e30) *infty_x += x[i]*x[i];
            else res += -x[i] * l[i];
        }
    }

    return res;
}


void PCG(const mfloat* x0, const mfloat* rhs, CG_struct* CG_space, mfloat* output) {
    int n = CG_space->dim_n, d = CG_space->dim_d;
    mfloat tol = CG_space->tol;
    double *x_pcg = CG_space->x, *r_pcg = CG_space->r, *z_pcg = CG_space->z;
    double *p_pcg = CG_space->p, *Ap_pcg = CG_space->Ap;
    for (int i = 0; i < n*d; ++i) x_pcg[i] = 0;
    vMemcpy(rhs, n*d, r_pcg);
    mfloat res_temp = norm2(r_pcg, n*d);
//    print_Mat(rhs, n, d);
    if (res_temp < tol) {
//        std::cout << "CG: starting point is ok" << std::endl;
        printf("      %3d", 0);
        vMemcpy(x_pcg, n*d, output);
        return;
//        return x_pcg;
    }

    precond_CG(CG_space);
    vMemcpy(z_pcg, n*d, p_pcg);

    // GPU
    std::chrono::steady_clock::time_point time0_mycg = time_now();
    My_CG();
    CG_space->cgTime += time_since(time0_mycg);

    // GPU
    mfloat s = xTy(p_pcg, Ap_pcg, n*d);
    mfloat rsold = xTy(r_pcg, z_pcg, n*d);
    mfloat t = rsold/s;

    axpy(t, p_pcg, x_pcg, x_pcg, n*d);
    for (int k = 1; k < CG_space->max_iter + 1; ++k) {
        axpy(-t, Ap_pcg, r_pcg, r_pcg, n*d);
        res_temp = norm2(r_pcg, n*d);
//        std::cout << k << ' ' << res_temp << std::endl;
        if (res_temp < tol) {
            CG_space->nCGs += k;
            printf("      %3d", k);
//            std::cout << "CG: quit at iteration " << k << std::endl;
            vMemcpy(x_pcg, n*d, output);
            return;
//            return x_pcg;
        }
        precond_CG(CG_space);
        mfloat rsnew = xTy(r_pcg, z_pcg, n*d);
        mfloat B = rsnew/rsold;
//        int iNUM = n*d;
        axpy(B, p_pcg, z_pcg, p_pcg, n*d);
        rsold = rsnew;

        std::chrono::steady_clock::time_point time0_mycg = time_now();
        My_CG();
        CG_space->cgTime += time_since(time0_mycg);

        s = xTy(p_pcg, Ap_pcg, n*d);
        t = rsold/s;
        axpy(t, p_pcg, x_pcg, x_pcg, n*d);
    }
    std::cout << "\nCG iteration at maximum " << CG_space->max_iter << " res " << res_temp << std::endl;
    vMemcpy(x_pcg, n*d, output);
//    return x_pcg;
}

void pALM_SSN_QP(QP_struct* QP_space) {
    BiCG_struct *BiCG_space = new BiCG_struct;
    mfloat findStep_LHS, findStep_RHS;
    mfloat pow_delta;
    int m = QP_space->m, n = QP_space->n;
    BiCG_space->dim_m = m;
    BiCG_space->dim_n = n;
    mfloat eta = 0.1, v = 1, delta = 0.9, rho = 0.01;
    int findStep_maxiter = 20;

    BiCG_init(BiCG_space);


    QP_space->sigma = max(1.0, QP_space->sigma);
    QP_space->iter = 1;

    // pALM

    pALM_QP_compute_status(QP_space);
    pALM_QP_print_status(QP_space);
    mfloat normGrad_speed;
    while ((QP_space->iter < QP_space->max_ALM_iter) && (!QP_space->pALM_finish) ) {
        mfloat tau_old = QP_space->tau;
        QP_space->tau = max(1e-12, QP_space->kappa*pow(QP_space->iter, -2.5)) * QP_space->sigma;
        // SSN
        int SSNiter = 1;
        mfloat tau_sigma = QP_space->tau/QP_space->sigma;

        vMemcpy(QP_space->y, m, QP_space->y_old);
        vMemcpy(QP_space->w, n, QP_space->w_old);
        vMemcpy(QP_space->x, n, QP_space->x_old);
//        vMemcpy(QP_space->z, m, QP_space->z0);
        vMemcpy(QP_space->Qw, n, QP_space->Qw_old);

        mfloat normGrad;
        SSN_findStep_update_variables(QP_space);
//        QP_space->SSN_tol = min(0.1, norm2(QP_space->b, m));
        while (SSNiter < QP_space->max_iter_SSN) {
            // compute rhs of SSN
//            cout << "normGrad = " << normGrad << endl;
            QP_space->obj_old = SSN_obj_val(QP_space);
            spMV_R(QP_space->A, QP_space->x, m, n, QP_space->Ax);
            axpy(-1, QP_space->Ax, QP_space->b, QP_space->Rp_sub, m);
            SSN_compute_grad(QP_space);
            normGrad = norm2(QP_space->SSN_grad, m+n);
            QP_space->normGrad_list[SSNiter] = normGrad;


            axpy(1, QP_space->temp_z, QP_space->z, QP_space->Rd_sub, n);
            mfloat primal_infeas_sub = norm2(QP_space->Rp_sub, m)/(1+QP_space->normb);
            mfloat dual_infeas_sub = norm2(QP_space->Rd_sub, n)/(1+QP_space->normc);

            QP_space->SSN_tol = max(QP_space->stop_tol, min(1e-3, 10*QP_space->tau/QP_space->sigma*sqrt(QP_space->x_diff_norm+QP_space->y_diff_norm+QP_space->dwQdw)));
//            BiCG_space->tol = min(eta, pow(norm2(QP_space->SSN_grad, m+n), 1+v));
//            BiCG_space->tol /= max(QP_space->normQ, 1.0);

            printf("        %3d| L=%- 7.6e| etapsub=%2.1e etadsub=%2.1e| normgrad=%2.1e tolSSN=%2.1e|",     SSNiter, QP_space->obj_old, primal_infeas_sub, dual_infeas_sub, normGrad, QP_space->SSN_tol);
//            BiCG_space->tol = 1e-6;

            if (normGrad < QP_space->SSN_tol) {
                printf("good termination in SSN\n");
                break;
            }
            if (SSNiter == QP_space->max_iter_SSN) {
                printf("maximum iteration in SSN reached\n");
                break;
            }
            if ((SSNiter > 5) && (QP_space->iter_step == QP_space->max_iter_step)) {
                printf("maximum iteration in step size search reached\n");
                break;
            }
            if (SSNiter > 20) {
                normGrad_speed = 0.0;
                for (int i = SSNiter-10; i <= SSNiter-1; ++i)
                    normGrad_speed += QP_space->normGrad_list[i+1]/QP_space->normGrad_list[i];
                normGrad_speed /= 10;

                if (abs(1-normGrad_speed) < 2e-2) {
                    printf("slow progress in SSN\n");
                    break;
                }
            }

            /// compute Newton direction
            if (QP_space->SSN_method == 0) {
                QP_space->precond = 1;
                mfloat *dQ = QP_space->dQdA;
                for (int row = 0; row < n; ++row) dQ[row] = 1.0+tau_sigma;
                spVec_int p = QP_space->U_idx;
//            mfloat *Q_value = QP_space->Q.value;
//            int *Q_rowStart = QP_space->Q.rowStart;
                for (int idx = 0; idx < QP_space->U_idx.number; ++idx) {
                    int r = p.vec[idx];
                    // scaled Q or original Q?
                    dQ[r] += QP_space->sigma*QP_space->diag_Q[r];
                }

                mfloat *dA = QP_space->dQdA+n;
                for (int row = 0; row < m; ++row) dA[row] = tau_sigma + QP_space->sigma * QP_space->diag_AAT[row];

                for (int row = 0; row < m+n; ++row) QP_space->dQdA[row] = 1.0 / max(1e-3, QP_space->dQdA[row]);
                BiCG_space->tol = max(1e-7, 1e-1*min(1.0, pow(normGrad,1.12)));
                BiCG(BiCG_space->x0, QP_space->SSN_grad, QP_space->SSN_dwdy, BiCG_space, QP_space);
            } else if (QP_space->SSN_method == 1) {
                if ((0 < QP_space->U_idx.number) && (QP_space->U_idx.number < n)) {
                    SSN_sub_case1(QP_space);
                } else if (QP_space->U_idx.number == n) {
                    if (QP_space->first_full_idx) {
                        QP_space->first_full_idx = false;
                        QP_space->diag_modifier.resize(m+n, 1);
//                        cout << QP_space->EigenQ.coeff(1,1) << endl;
                        QP_space->AQ = QP_space->EigenA*QP_space->EigenQ;
                        QP_space->hessian_1 = -(1+QP_space->tau/QP_space->sigma)/QP_space->sigma*QP_space->EigenIn - QP_space->EigenQ;
//                        cout << QP_space->hessian_1.norm() << endl;
                        QP_space->hessian_4 = -QP_space->tau/QP_space->sigma/QP_space->sigma*QP_space->EigenIm - QP_space->EigenAAT;
//                        cout << QP_space->hessian_4.norm() << endl;
                        QP_space->hessian_full_idx = concat_4_spMat(QP_space->hessian_1, QP_space->EigenAT, QP_space->AQ, QP_space->hessian_4);
//                        cout << QP_space->hessian_full_idx << endl;
                        QP_space->Eigen_linear_hessian_full_idx.LUsolver.analyzePattern(QP_space->hessian_full_idx);
                        QP_space->Eigen_linear_hessian_full_idx.LUsolver.factorize(QP_space->hessian_full_idx);
                    } else {

                        // can be improved
                        QP_space->diag_modifier.segment(0, n).setConstant((1+tau_old/QP_space->sigma_old)/QP_space->sigma_old - (1+QP_space->tau/QP_space->sigma)/QP_space->sigma);
                        QP_space->diag_modifier.segment(n, m).setConstant(tau_old/QP_space->sigma_old/QP_space->sigma_old - QP_space->tau/QP_space->sigma/QP_space->sigma);

                        QP_space->hessian_full_idx.diagonal() += QP_space->diag_modifier;


//                        QP_space->hessian_1 = -(1+QP_space->tau/QP_space->sigma)/QP_space->sigma*QP_space->EigenIn - QP_space->EigenQ;
//                        QP_space->hessian_4 = -QP_space->tau/QP_space->sigma/QP_space->sigma*QP_space->EigenIm - QP_space->EigenAAT;
//                        QP_space->hessian_full_idx = concat_4_spMat(QP_space->hessian_1, QP_space->EigenAT, QP_space->AQ, QP_space->hessian_4);
//                        QP_space->Eigen_linear_hessian_full_idx.LLTsolver.analyzePattern(QP_space->hessian_full_idx);
                        QP_space->Eigen_linear_hessian_full_idx.LUsolver.factorize(QP_space->hessian_full_idx);
//                        cout << QP_space->hessian_full_idx.norm() << endl;
//                        cout << QP_space->hessian_full_idx.coeff(1,1) << endl;
                    }
                    ax(1/QP_space->sigma, QP_space->SSN_grad, QP_space->SSN_grad_invsigma, m+n);
                    vMemcpy(QP_space->SSN_grad_invsigma, m+n, QP_space->Eigen_rhs_dwdy.data());
//                    cout << QP_space->Eigen_rhs_dwdy << endl;
                    QP_space->Eigen_result_dwdy = QP_space->Eigen_linear_hessian_full_idx.LUsolver.solve(QP_space->Eigen_rhs_dwdy);
//                    cout << QP_space->Eigen_result_dwdy << endl;
                    vMemcpy(QP_space->Eigen_result_dwdy.data(), m+n, QP_space->SSN_dwdy);

                    // get -dwdy in this section
                    ax(-1, QP_space->SSN_dwdy, QP_space->SSN_dwdy, m+n);
                } else {}
            }

//            Taylor_test(BiCG_space, QP_space);


//        SSN_compute_grad(QP_space);
//            mfloat temp_rhs = rho * xTy(QP_space->SSN_grad_Q, QP_space->SSN_dwdy, m+n);
//
//
//            for (i = 0; i < findStep_maxiter; ++i) {
//                pow_delta = pow(delta, i);
//                axpy(pow_delta, QP_space->SSN_dwdy, QP_space->w_old, QP_space->w, n);
//                axpy(pow_delta, QP_space->SSN_dwdy+n, QP_space->y_old, QP_space->y, m);
//                SSN_findStep_update_variables(QP_space);
//                findStep_LHS = SSN_obj_val(QP_space);
//                findStep_RHS = obj_old + pow_delta * temp_rhs;
//                if (findStep_LHS <= findStep_RHS) break;
//            }
//            QP_space->step_size = pow_delta;
//#ifdef SSN_debug
//            cout << "     step size " << pow_delta << " normGrad " << normGrad << endl;
//#endif
            line_search(QP_space);

//            vMemcpy(QP_space->y, m, QP_space->y_old);
//            vMemcpy(QP_space->w, n, QP_space->w_old);

//            if (i > 0)
//                cout << "find step iteration " << i << endl;
//            if (i == findStep_maxiter)
//                cout << "find step reaches maximum iterations " << findStep_maxiter << endl;

            SSNiter++;
        }

//        if ((QP_space->iter % sGSADMM_sigma_update_iter(QP_space->iter) == 0) || (QP_space->iter % sGSADMM_print_iter(QP_space->iter) == 0))
//        pALM_update_variables(QP_space);
        pALM_QP_compute_status(QP_space);
        update_sigma_pALM(QP_space);
//        QP_space->SSN_tol = 0.1*QP_space->inf_p;
        //        update_sigma(QP_space);
        QP_space->iter++;
        pALM_QP_print_status(QP_space);
    }

}

void Taylor_test(BiCG_struct* BiCG_space, QP_struct* QP_space) {
    int m = QP_space->m, n = QP_space->n;
    mfloat *old_grad_Q = new mfloat [m+n];
    mfloat *old_grad = new mfloat [m+n];
    mfloat *old_w = new mfloat [n];
    mfloat *old_y = new mfloat [m];
    vMemcpy(QP_space->SSN_grad, m+n, old_grad);
    vMemcpy(QP_space->SSN_grad_Q, m+n, old_grad_Q);
    vMemcpy(QP_space->w, n, old_w);
    vMemcpy(QP_space->y, n, old_y);

    mfloat *dwdy = new mfloat [m+n];
    for (int i = 0; i < m+n; ++i) {
        dwdy[i] = (rand() % 100) / 100.0;
    }

    mfloat *temp_grad = new mfloat[m+n];
    mfloat *mVdwdy = new mfloat[m+n];
    mfloat *temp2 = new mfloat [m+n];

    for (int i = 0; i < 4; ++i) {
        SSN_findStep_update_variables(QP_space);
        SSN_compute_grad(QP_space);

        vMemcpy(QP_space->SSN_grad, m+n, temp_grad);

        BiCG_prod(dwdy, mVdwdy, BiCG_space, QP_space);

        axpy(1, dwdy, old_w, QP_space->w, n);
        axpy(1, dwdy+n, old_y, QP_space->y, m);
        SSN_findStep_update_variables(QP_space);
        SSN_compute_grad(QP_space);

        axpy(-1, temp_grad, QP_space->SSN_grad, temp2, m+n);


//        cout << "w " << norm2(temp2, n) << endl;
//
//        cout << "y " << norm2(temp2+n, m) << endl;

        axpy(1, mVdwdy, temp2, temp2, m+n);

        cout << "w " << norm2(temp2, n)/norm2(dwdy, n) << endl;

        cout << "y " << norm2(temp2+n, m)/norm2(dwdy+n, m) << endl;

        vMemcpy(old_w, n, QP_space->w);
        vMemcpy(old_y, m, QP_space->y);
        vMemcpy(old_grad, m+n, QP_space->SSN_grad);
        vMemcpy(old_grad_Q, m+n, QP_space->SSN_grad_Q);
        for (int j = 0 ; j < m+n; ++j)
            dwdy[j] /= 10;
    }
    SSN_findStep_update_variables(QP_space);


}

void line_search(QP_struct* qp) {
    // the performance is not exactly same with MATLAB version due to numerical error,
    // also influence the following iterations.
    int m = qp->m, n = qp->n;
    mfloat *dw = qp->SSN_dwdy, *dy = qp->SSN_dwdy+n;
    spMV_R(qp->Q, dw, n, n, qp->Qdw);
    spMV_R(qp->AT, dy, n, m, qp->ATdy);

    mfloat g0 = xTy(qp->Qdw, qp->SSN_grad, n);
    g0 += xTy(dy, qp->SSN_grad+n, m);

    if (g0 < 0) {
        printf("need descent direction\n");
        ax(1, qp->SSN_grad, dw, n);
        spMV_R(qp->Q, dw, n, n, qp->Qdw);
        ax(1, qp->SSN_grad+n, dy, m);
        spMV_R(qp->AT, dy, n, m, qp->ATdy);
    }


    mfloat c1 = 1e-4, c2 = 0.9;
    mfloat tau_sigma = qp->tau/qp->sigma, sigma = qp->sigma;
    spMV_R(qp->AT, qp->y, qp->n, qp->m, qp->ATy0);

    /// Armijo rule
    mfloat alpha = 1.0, alpha_const = 0.5, LB = 0, UB = 1;
    mfloat g_LB, g_UB;
    int iter;
    for (iter = 1; iter <= qp->max_iter_step; ++iter) {
        if (iter == 1) {
            alpha = 1.0;
            LB = 0; UB = 1;
        } else
            alpha = alpha_const*(LB+UB);

        axpy(alpha, dw, qp->w, qp->w_new, n);
        axpy(alpha, qp->Qdw, qp->Qw, qp->Qw_new, n);
        axpby(-1, qp->Qw_new, -1, qp->c, qp->temp_z, n);

        axpy(alpha, dy, qp->y, qp->y_new, m);
        axpy(alpha, qp->ATdy, qp->ATy0, qp->ATy, n);
        axpy(1, qp->ATy, qp->temp_z, qp->temp_z, n);

        axpy(sigma, qp->temp_z, qp->x_old, qp->pALM_z_unProjected, n);
        proj(qp->pALM_z_unProjected, qp->x, n, qp->l, qp->u, qp->proj_idx);

        axpby(1/sigma, qp->x, -1/sigma, qp->pALM_z_unProjected, qp->z, n);

        // compute obj
        axpy(-1, qp->w_old, qp->w_new, qp->w_diff, n);
        axpy(-1, qp->Qw_old, qp->Qw_new, qp->Qw_diff, n);
        axpy(tau_sigma, qp->w_diff, qp->w_new, qp->line_search_temp1, n);
        axpy(-1, qp->x, qp->line_search_temp1, qp->line_search_temp1, n);
        mfloat g_alpha = -xTy(qp->Qdw, qp->line_search_temp1, n);
        mfloat LQ = -0.5*xTy(qp->w_new, qp->Qw_new, n) - xTy(qp->x, qp->temp_z, n) + 0.5/sigma*pow(distance(qp->x, qp->x_old, n), 2)
                - 0.5*tau_sigma*xTy(qp->w_diff, qp->Qw_diff, n);

        axpy(-1, qp->y_old, qp->y_new, qp->y_diff, m);
        axpy(-tau_sigma, qp->y_diff, qp->b, qp->line_search_temp2, m);
        g_alpha += xTy(dy, qp->line_search_temp2, m) - xTy(qp->x, qp->ATdy, n);
        LQ += xTy(qp->b, qp->y_new, m) - 0.5*tau_sigma*xTy(qp->y_diff, qp->y_diff, m);

        if (iter == 1) {
            g_LB = g0;
            g_UB = g_alpha;
            if (g_LB*g_UB > 0) break;
        }

        if ((abs(g_alpha) < c2*abs(g0)) &&
        (LQ - qp->obj_old - c1*alpha*g0 > -1e-12/max(1.0, abs(qp->obj_old))))
            break;

        if (g_alpha*g_UB < 0) {
            LB = alpha;
            g_LB = g_alpha;
        } else if (g_alpha*g_LB < 0) {
            UB = alpha;
            g_UB = g_alpha;
        }
    }

    vMemcpy(qp->w_new, n, qp->w);
    vMemcpy(qp->Qw_new, n, qp->Qw);
    vMemcpy(qp->y_new, m, qp->y);


    printf(" [step = %2.1e iter = %2d]\n", alpha, iter);

    qp->iter_step = iter;
    int nnz = 0;
    int *p = qp->U_idx.vec;
    for (int i = 0; i < n; ++i) {
        if (qp->proj_idx[i]) {
            p[nnz] = i;
            nnz++;
        }
    }
    qp->U_idx.number = nnz;
}

void SSN_findStep_update_variables(QP_struct* QP_space) {
    int m = QP_space->m, n = QP_space->n;
//    spMV_R(QP_space->Q, QP_space->w, n, n, QP_space->Qw);
//    spMV_R(QP_space->AT, QP_space->y, n, m, QP_space->ATy);
//    axpy(-1, QP_space->ATy, QP_space->Qw, QP_space->Qw_ATy_c, n);
//    axpy(1, QP_space->c, QP_space->Qw_ATy_c, QP_space->Qw_ATy_c, n);
//
//    axpy(-QP_space->sigma, QP_space->Qw_ATy_c, QP_space->x, QP_space->pALM_z_unProjected, n);
    axpy(-1, QP_space->z, QP_space->Rd, QP_space->temp_z, n);
    axpy(QP_space->sigma, QP_space->temp_z, QP_space->x_old, QP_space->pALM_z_unProjected, n);

    // maybe not exactly x, just somewhere to store.
    // --not true, should be x
    proj(QP_space->pALM_z_unProjected, QP_space->x, n, QP_space->l, QP_space->u, QP_space->proj_idx);

    int nnz = 0;
    int *p = QP_space->U_idx.vec;
    for (int i = 0; i < n; ++i) {
        if (QP_space->proj_idx[i]) {
            p[nnz] = i;
            nnz++;
        }
    }
    QP_space->U_idx.number = nnz;
}

void pALM_update_variables(QP_struct* QP_space) {
    int n = QP_space->n;
    vMemcpy(QP_space->pALM_z_projected, n, QP_space->x);
    axpy(-1, QP_space->pALM_z_unProjected, QP_space->pALM_z_projected, QP_space->z, n);
    ax(1/QP_space->sigma, QP_space->z, QP_space->z, n);
}

mfloat SSN_obj_val(QP_struct* QP_space) {
    mfloat res = 0.0;
    int m = QP_space->m, n = QP_space->n;
    res -= xTy(QP_space->temp_z, QP_space->x, n);


    QP_space->x_diff_norm = pow(distance(QP_space->x, QP_space->x_old, n), 2);
//    axpy(-1, QP_space->x, QP_space->pALM_z_projected, QP_space->SSN_compute_obj_temp1, n);
//    QP_space->x_diff_norm = xTy(QP_space->SSN_compute_obj_temp1, QP_space->SSN_compute_obj_temp1, n);
    res -= 1/(2*QP_space->sigma)* QP_space->x_diff_norm;

    // check here, different from the paper
    res -= 1.0/2*xTy(QP_space->w, QP_space->Qw, n);

    res += xTy(QP_space->b, QP_space->y, m);


    // seems no need
    axpy(-1, QP_space->w_old, QP_space->w, QP_space->w_diff, n);
//    spMV_R(QP_space->Q, QP_space->w_diff, n, n, QP_space->Qdw);

//    QP_space->dwQdw = xTy(QP_space->w_diff, QP_space->Qdw, n);
    axpy(-1, QP_space->Qw_old, QP_space->Qw, QP_space->Qw_diff, n);
    QP_space->dwQdw = xTy(QP_space->w_diff, QP_space->Qw_diff, n);
    axpy(-1, QP_space->y_old, QP_space->y, QP_space->y_diff, m);
    QP_space->y_diff_norm = pow(norm2(QP_space->y_diff, m), 2);
    res -= QP_space->tau/(2*QP_space->sigma) * (QP_space->dwQdw + QP_space->y_diff_norm);

    return res;
}

void SSN_compute_grad(QP_struct* QP_space) {
    int m = QP_space->m, n = QP_space->n;
    mfloat tau = QP_space->tau, sigma = QP_space->sigma;
    axpby(1, QP_space->x, -1, QP_space->w, QP_space->SSN_grad, n);
    axpy(-tau/sigma, QP_space->w_diff, QP_space->SSN_grad, QP_space->SSN_grad, n);
    spMV_R(QP_space->Q, QP_space->SSN_grad, n, n, QP_space->SSN_grad_Q);


//    spMV_R(QP_space->A, QP_space->x, m, n, QP_space->Az);
//    axpy(-1, QP_space->b, QP_space->Az, QP_space->SSN_grad+n, m);
    axpy(-tau/sigma, QP_space->y_diff, QP_space->Rp_sub, QP_space->SSN_grad+n, m);
//    axpy(-tau/sigma, QP_space->y_old, QP_space->SSN_grad+n, QP_space->SSN_grad+n, m);
    vMemcpy(QP_space->SSN_grad+n, m, QP_space->SSN_grad_Q+n);
//    ax(-1, QP_space->SSN_grad, QP_space->SSN_grad,)
}

void BiCG_prod(const mfloat *x, mfloat* output, BiCG_struct *BiCG_space, QP_struct *QP_space) {
    // minus linear system, Ax = -b -> -Ax = b
//    spMV_R(BiCG_space->A, x, BiCG_space->A.nRow, BiCG_space->A.nCol, output);
    int n = BiCG_space->dim_n, m = BiCG_space->dim_m;
    mfloat *output1 = output, *output2 = output + n;
    const mfloat *x1 = x;
    const mfloat *x2 = x + n;
    spVec_int U_idx = QP_space->U_idx;
    mfloat *temp1 = BiCG_space->BiCG_prod_temp1;
    mfloat *temp2 = BiCG_space->BiCG_prod_temp2;
    mfloat *Qx = BiCG_space->Qx;
    mfloat *ATy = BiCG_space->ATy;
    mfloat tau = QP_space->tau, sigma = QP_space->sigma;
    sparseRowMatrix Q = QP_space->Q, AT = QP_space->AT;

    for (int i = 0; i < n; ++i) {
        temp1[i] = 0;
        temp2[i] = 0;
        output1[i] = 0;
    }
    for (int i = 0; i < m; ++i) {
        output2[i] = 0;
    }

    partial_spMV_R(Q, x1, U_idx.number, n, Qx, U_idx);
    axpby(1+tau/sigma, x1, sigma, Qx, temp1, n);
    partial_spMV_R(AT, x2, U_idx.number, m, ATy, U_idx);
    axpy(-sigma, ATy, temp1, output1, n);

//    partial_spMV_R(Q, x1, U_idx.number, n, temp2, U_idx);
//    axpy(1+tau/sigma, x1, temp1, temp1, n);

    // strange here, check why spMV_2
    partial_spMV_2(AT, Qx, n, m, temp1, U_idx);

//    partial_spMV_R(AT, x2, U_idx.number, m, temp1, U_idx);

    partial_spMV_2(AT, ATy, n, m, output2, U_idx);
    axpby(tau/sigma, x2, sigma, output2, output2, m);
    axpy(-sigma, temp1, output2, output2, m);
}

void BiCG_init(BiCG_struct* CG_space) {
    int n = CG_space->dim_n + CG_space->dim_m;
    CG_space->x = new mfloat [n];
    CG_space->Qx = new mfloat [CG_space->dim_n];
    CG_space->ATy = new mfloat [CG_space->dim_n];
    CG_space->r = new mfloat [n];
    CG_space->r0 = new mfloat [n];
    CG_space->p_hat = new mfloat [n];
    CG_space->s = new mfloat [n];
    CG_space->s_hat = new mfloat [n];
    CG_space->t = new mfloat [n];
    CG_space->p = new mfloat [n];
    CG_space->Ap = new mfloat [n];
    CG_space->v = new mfloat [n];
    CG_space->BiCG_prod_temp1 = new mfloat [CG_space->dim_n];
    CG_space->BiCG_prod_temp2 = new mfloat [CG_space->dim_n];
    CG_space->x0 = new mfloat [n];
    for (int i = 0; i < n; ++i) CG_space->x0[i] = 0;
}

void precond_BiCG(const mfloat* input, mfloat* output, QP_struct *qp) {
    int m = qp->m, n = qp->n;
    if (qp->precond == 0)
        vMemcpy(input, m+n, output);
    else if (qp->precond == 1) {
        for (int i = 0; i < m+n; ++i)
            output[i] = input[i] * qp->dQdA[i];
    }
}

void BiCG(const mfloat* x0, const mfloat* rhs, mfloat* output, BiCG_struct* CG_space, QP_struct *QP_space) {
    int n = CG_space->dim_n + CG_space->dim_m;
    mfloat tol = CG_space->tol;
    double *x = CG_space->x, *r = CG_space->r;
    mfloat *r0 = CG_space->r0, *p_hat = CG_space->p_hat, *s = CG_space->s;
    mfloat *s_hat = CG_space->s_hat, *t = CG_space->t;
    double *p = CG_space->p, *Ap = CG_space->Ap, *v = CG_space->v;
    mfloat rho_old = 1, alpha = 1, omega = 1;
//    mfloat bnorm = norm2(rhs, n)+1;
    mfloat bnorm = 1;
//    for (int i = 0; i < n; ++i) {
//        v[i] = 0;
//        p[i] = 0;
//    }
    vMemcpy(x0, n, x);
    BiCG_prod(x0, Ap, CG_space, QP_space);
//    cout << Ap[0] << ' ' << Ap[1] << ' ' << Ap[2] << ' ' << Ap[3] << endl;
    axpy(-1, Ap, rhs, r, n);
    vMemcpy(r, n, r0);

    int iter = 1;
    mfloat rho = 1;
    mfloat beta = 1;
    mfloat res = norm2(r0, n);
    if (res < tol) {
//        printf("      %3d", 0);
        cout << "BiCG input is okay" << endl;
        vMemcpy(x, n, output);
        return;
    }
    while (iter < CG_space->max_iter) {
        rho = xTy(r, r0, n);
        if (abs(rho) == 0) {
            cout << "BiCG error: rho = 0" << endl;
            return;
        }
        if (iter > 1) {
            beta = rho/rho_old*alpha/omega;
            axpy(-omega, v, p, p, n);
            axpy(beta, p, r, p, n);
        } else {
            vMemcpy(r, n, p);
        }

        precond_BiCG(p, p_hat, QP_space);

        BiCG_prod(p_hat, v, CG_space, QP_space);
        alpha = rho/ xTy(r0, v, n);
//        axpy(alpha, p_pcg, x_pcg, h, n);

        axpy(-alpha, v, r, s, n);
        mfloat norms = norm2(s, n);
        if (norms < tol) {
            axpy(alpha, p_hat, x, x, n);
//            res = norms/bnorm;
            vMemcpy(x, n, output);
            return;
        }

        precond_BiCG(s, s_hat, QP_space);
        BiCG_prod(s_hat, t, CG_space, QP_space);
        omega = xTy(t, s, n)/xTy(t, t, n);
        axpy(alpha, p_hat, x, x, n);
        axpy(omega, s_hat, x, x, n);

        axpy(-omega, t, s, r, n);
        CG_space->error = norm2(r, n);
//        printf("iter %d err %f\n", iter, CG_space->error);
        if (CG_space->error < tol) {
            vMemcpy(x, n, output);
            return;
        }

        if (abs(omega) == 0) {
            cout << "omega = 0 in BiCG" << endl;
            return;
        }
        rho_old = rho;
        iter++;
    }

    cout << "BiCG reaches maximum iteration number " << CG_space->max_iter << " error " << CG_space->error << endl;
}



Mat creat_Mat(int m, int n) {
    Mat output;
    output.nRow = m;
    output.nCol = n;
    output.vec = new mfloat [m*n];
    return output;
}

void print_Mat(const mfloat* X, int m, int n) {
    std::cout << "Matrix: " << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            std::cout << X[j*m+i] << ' ';
        std::cout << std::endl;
    }
}

void partial_spMV_R(sparseRowMatrix A, const mfloat* x, int m, int n, mfloat* Ax, spVec_int nzidx) {
    mfloat* value = A.value;
    int* rowStart = A.rowStart;
    int* column = A.column;
    assert(A.nCol == n);
    assert(nzidx.number == m);
    int* nzidxv = nzidx.vec;
//#pragma omp parallel for
    for (int idx = 0; idx < nzidx.number; ++idx) {
//        int i = nzidxv[idx];
        Ax[idx] = 0;
//        mfloat temp = 0;
        for (int j = rowStart[nzidxv[idx]]; j < rowStart[nzidxv[idx]+1]; ++j) {
            Ax[idx] += value[j]*x[column[j]];
        }
//        Ax[idx] = temp;
    }
}

void partial_CR_spMV(sparseRowMatrix A, const mfloat* x, int nRow, int nCol, mfloat* Ax, bool *row_idx, bool *col_idx) {
    mfloat* value = A.value;
    int* rowStart = A.rowStart;
    int* column = A.column;
    assert(A.nCol == nCol);
//    int row_num = 0;
    if (row_idx == nullptr) {
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) shared(col_idx,row_idx,nRow, Ax, rowStart, value, x, column, nCol) default(none)
        for (int i = 0; i < nRow; ++i) {
            Ax[i] = 0;
            for (int j = rowStart[i]; j < rowStart[i+1]; ++j) {
//            assert(column[j] < nCol);
                if (col_idx[column[j]])
                    Ax[i] += value[j]*x[column[j]];
            }
        }
    } else if (col_idx == nullptr) {
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) shared(col_idx,row_idx,nRow, Ax, rowStart, value, x, column, nCol) default(none)
        for (int i = 0; i < nRow; ++i) {
            Ax[i] = 0;
            if (row_idx[i]) {
                for (int j = rowStart[i]; j < rowStart[i+1]; ++j) {
//            assert(column[j] < nCol);
                    Ax[i] += value[j]*x[column[j]];
                }
            }

        }
    } else {
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) shared(col_idx,row_idx,nRow, Ax, rowStart, value, x, column, nCol) default(none)
        for (int i = 0; i < nRow; ++i) {
            Ax[i] = 0;
            if (row_idx[i]) {
                for (int j = rowStart[i]; j < rowStart[i+1]; ++j) {
//            assert(column[j] < nCol);
                    if (col_idx[column[j]])
                        Ax[i] += value[j]*x[column[j]];
                }
            }

        }
    }

}

sparseRowMatrix B_sub_Row(spVec_int nzidx, sparseRowMatrix BT) {
    sparseRowMatrix Bf;
    int num = nzidx.number;
    if (num > 0) {
        int n = BT.nCol;
//    int* rowIndex = new int [2*num];
        int nnz = 2*num;
        Bf.value = new mfloat [nnz];
        Bf.rowStart = new int [n+2];
        Bf.column = new int [nnz];
        mfloat* newValue = Bf.value;
        int *rowStart = Bf.rowStart;
        int *column = Bf.column;

        mfloat* oldValue = BT.value;
        int *colStart = BT.rowStart;
        int *row = BT.column;
        for (int i = 0; i < n+2; ++i) {
            rowStart[i] = 0;
        }
        // count per row
        int* nzidxv = nzidx.vec;
        for (int idx = 0; idx < num; ++idx) {
            int i = nzidxv[idx];
            for (int j = colStart[i]; j < colStart[i+1]; ++j)
                ++rowStart[row[j]+2];
        }
//    for (int i = 0; i < nnz; ++i) {
//        ++rowStart[row[i]+2];
//    }

        // generate rowStart
        for (int i = 2; i < n+2; ++i) {
            rowStart[i] += rowStart[i-1];
        }

        // main part
        for (int idx = 0; idx < num; ++idx) {
            int i = nzidxv[idx];
            assert(i < BT.nRow);
            for (int j = colStart[i]; j < colStart[i+1]; ++j) {
                int newIndex = rowStart[row[j] + 1]++;
                newValue[newIndex] = oldValue[j];
                column[newIndex] = idx;
            }
        }
        Bf.nCol = nzidx.number;
        Bf.nRow = BT.nCol;
    } else {
        Bf.nCol = nzidx.number;
        Bf.nRow = 0;
    }

    return Bf;
}

void Runexperiment_MINST() {
}

void create_SNAL() {
}

void initial_SNAL() {
}

void SNAL_new() {
}

//bool collect_infeasibility() {
//
//}
void create_SSNCG() {
}

void SSNCG_new() {
}

void create_find_step() {
}

void find_step(int stepopt) {
}


//bool check_termination_SSN(int iter) {
//}

int sGSADMM_sigma_update_iter(int iter) {
    int sigma_update_iter = 1;
    if (iter < 10)
        sigma_update_iter = 2;
    else if (iter < 20)
        sigma_update_iter = 3;
    else if (iter < 200)
        sigma_update_iter = 5;
    else if (iter < 500)
        sigma_update_iter = 10;

    return sigma_update_iter;
}

void save_solution(QP_struct *qp) {
    vMemcpy(qp->x, qp->n, qp->x_old);
    vMemcpy(qp->z, qp->n, qp->z_old);
    vMemcpy(qp->w, qp->n, qp->w_old);
    vMemcpy(qp->Qx, qp->n, qp->Qx_old);
    vMemcpy(qp->Qw, qp->n, qp->Qw_old);
    vMemcpy(qp->y, qp->m, qp->y_old);
}

void read_solution(QP_struct *qp) {
    vMemcpy(qp->x_old, qp->n, qp->x);
    vMemcpy(qp->z_old, qp->n, qp->z);
    vMemcpy(qp->w_old, qp->n, qp->w);
    vMemcpy(qp->Qx_old, qp->n, qp->Qx);
    vMemcpy(qp->Qw_old, qp->n, qp->Qw);
    vMemcpy(qp->y_old, qp->m, qp->y);
}

void update_gamma(QP_struct *qp) {
    int m = qp->m, n = qp->n;
    qp->const0;
    if ((qp->gamma > 1.618) && (qp->iter >= qp->gamma_reset_start) && qp->gamma_test) {
        mfloat tmp = pow(distance(qp->Qw, qp->Qw_old, qp->n), 2);
        if ((qp->iter == qp->gamma_reset_start) || (qp->iter % 1000 == 0)) {
            qp->const0 = max(1.0, qp->sigma*(1.0/qp->gamma* pow(norm2(qp->Rd, n), 2) + tmp) * 3);
        }
        if (((qp->sigma/qp->gamma*pow(norm2(qp->Rd, n),2) + qp->sigma*tmp) * (pow(qp->gamma,2)-qp->gamma-1.0)) > (qp->const0/(pow(qp->iter, 1.2)))) {
            qp->gamma = max(1.618, qp->gamma*0.97);
            read_solution(qp);
            sGSADMM_QP_compute_status(qp);
        }
    }
}

void update_sigma(QP_struct *QP_space) {
//    bool use_inforg = false;
    int iter = QP_space->iter;
    if (min(QP_space->inf_p_org, QP_space->inf_d_org) < 50 * QP_space->stop_tol) {
        QP_space->use_inforg = true;
//        cout << "iter = " << iter << " use inforg" << endl;
    }
    mfloat errPD, errP, errD;
    if (QP_space->use_inforg) {
        errPD = max(QP_space->inf_Q_org, QP_space->inf_C_org);
        errP = QP_space->inf_p_org;
        errD = QP_space->inf_d_org;
    } else {
        errPD = max(QP_space->inf_Q, QP_space->inf_C);
        errP = QP_space->inf_p;
        errD = QP_space->inf_d;
    }

    mfloat ratio = max(errP, 0.1*errPD) / errD;
    if (ratio < 0.95)
        QP_space->prim_win++;
    else if (ratio > 1/0.95) {
        QP_space->dual_win++;
//        cout << "errP = " << errP << " errD = " << errD << " errPD = " << errPD << " ratio = " << ratio << endl;
    }



//    mfloat inf_primal = max(QP_space->inf_p, max(QP_space->inf_Q, QP_space->inf_C));

    if ((!QP_space->fix_sigma) && (iter % QP_space->sigma_update_iter == 1)) {
        QP_space->sigma_old = QP_space->sigma;
//        cout << "primal win: " << QP_space->prim_win << " dual win: " << QP_space->dual_win << endl;
        if (QP_space->prim_win > QP_space->dual_win) {
            // primal good, increase sigma
            QP_space->sigma = min(QP_space->sigma_max, QP_space->sigma_old * QP_space->sigma_scale);
            QP_space->sigma_change++;
            QP_space->prim_win = 0;
        } else if (QP_space->dual_win > QP_space->prim_win) {
            // dual good, decrease sigma
            QP_space->sigma = max(QP_space->sigma_min, QP_space->sigma_old / QP_space->sigma_scale);
            QP_space->sigma_change++;
            QP_space->dual_win = 0;
        } else {
            // no winner, keep sigma
            QP_space->sigma = QP_space->sigma_old;
        }

        if (abs(QP_space->sigma_old - QP_space->sigma) > 1e-10) {
            if (!QP_space->use_mkl)
                sGSADMM_QP_refact_IQ(QP_space);
            else
                MKL_refact(QP_space);
        }
#ifdef sigma_debug
        if (use_inforg)
            printf("inf_p: %3.2e, inf_Q: %3.2e, inf_C: %3.2e, inf_d: %3.2e, sigma: %3.2e\n", QP_space->inf_p_org, QP_space->inf_Q_org, QP_space->inf_C_org, QP_space->inf_d_org, QP_space->sigma);
        else
            printf("inf_p: %3.2e, inf_Q: %3.2e, inf_C: %3.2e, inf_d: %3.2e, sigma: %3.2e\n", QP_space->inf_p, QP_space->inf_Q, QP_space->inf_C, QP_space->inf_d, QP_space->sigma);
#endif
    }
}

void update_sigma_pALM(QP_struct* QP_space) {
    int iter = QP_space->iter;

    mfloat inf_primal = max(QP_space->inf_p_org, max(QP_space->inf_Q_org, QP_space->inf_C_org));
    QP_space->sigma_old = QP_space->sigma;
    if (iter % 1 == 0) {
        if (inf_primal < 0.75*QP_space->inf_d_org) {
            QP_space->sigma *= QP_space->sigma_scale;
            QP_space->sigma_change++;
        }
        else if (inf_primal > 1.33*QP_space->inf_d_org) {
            QP_space->sigma /= QP_space->sigma_scale;
            QP_space->sigma_change++;
        }

        QP_space->sigma = min(QP_space->sigma, QP_space->sigma_max);
        QP_space->sigma = max(QP_space->sigma, QP_space->sigma_min);
    }
}

void print_spM(sparseRowMatrix input) {
    for (int i = 0; i < input.nRow; ++i) {
        for (int j = input.rowStart[i]; j < input.rowStart[i+1]; ++j) {
            std::cout << i << ' ' << input.column[j] << ' ' << input.value[j] << std::endl;
        }
    }
}

Eigen::SparseMatrix<mfloat> get_eigen_spMat(sparseRowMatrix B) {
    typedef Eigen::Triplet<mfloat> T;
    std::vector<T> tripletList;
    tripletList.reserve(B.rowStart[B.nRow]);
    int *rowStart = B.rowStart;
    mfloat *value = B.value;
    int *column = B.column;
    for (int i = 0; i < B.nRow; ++i) {
        for (int j = rowStart[i]; j < rowStart[i+1]; ++j)
            tripletList.emplace_back(i, column[j],value[j]);
    }
    Eigen::SparseMatrix<mfloat> mat(B.nRow, B.nCol);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
// mat is ready to go!

}

void get_eigen_sub_spMat_Q(sparseRowMatrix B, bool *idx, std::vector<Eigen::Triplet<mfloat>> tripletList, int *idx_sum, Eigen::SparseMatrix<mfloat> *mat) {
//    typedef Eigen::Triplet<mfloat> T;
//    std::vector<T> tripletList;
//    tripletList.reserve(B.rowStart[B.nRow]);
    tripletList.clear();
    tripletList.reserve(B.rowStart[B.nRow]);
    int *rowStart = B.rowStart;
    mfloat *value = B.value;
    int *column = B.column;
    int nRow = 0;
    for (int i = 0; i < B.nRow; ++i) {
        if (!idx[i]) continue;
        for (int j = rowStart[i]; j < rowStart[i+1]; ++j) {
            int col = column[j];
            if (!idx[col]) continue;
            tripletList.emplace_back(i-idx_sum[i], col-idx_sum[col],value[j]);
        }
    }
//    cout << nRow << endl;
//    auto data = tripletList.data();
//    cout << data[0] << endl;
//    cout << tripletList.size() << endl;
    mat->resize(B.nRow-idx_sum[B.nRow-1], B.nRow-idx_sum[B.nRow-1]);
    mat->setFromTriplets(tripletList.begin(), tripletList.end());
//    return Q_sub;
}

void get_eigen_sub_spMat_A(sparseRowMatrix B, bool *idx, std::vector<Eigen::Triplet<mfloat>> tripletList, int *idx_sum, Eigen::SparseMatrix<mfloat> *mat) {
//    typedef Eigen::Triplet<mfloat> T;
//    std::vector<T> tripletList;
//    tripletList.reserve(B.rowStart[B.nRow]);

    // get A partial column or get AT partial row, which is faster??
    tripletList.clear();
    tripletList.reserve(B.rowStart[B.nRow]);
    int *rowStart = B.rowStart;
    mfloat *value = B.value;
    int *column = B.column;
    int nRow = 0;
    for (int i = 0; i < B.nRow; ++i) {
        for (int j = rowStart[i]; j < rowStart[i+1]; ++j) {
            int col = column[j];
            if (!idx[col]) continue;
            tripletList.emplace_back(i, col-idx_sum[col],value[j]);
        }
    }
//    cout << nRow << endl;
//    auto data = tripletList.data();
//    cout << data[0] << endl;
//    cout << tripletList.size() << endl;
    mat->resize(B.nRow, B.nCol-idx_sum[B.nCol-1]);
    mat->setFromTriplets(tripletList.begin(), tripletList.end());
//    return Q_sub;
}

void precond_CG(CG_struct* CG_space) {
    if (CG_space->LLT.isEmpty) {
        vMemcpy(CG_space->r, CG_space->dim_n*CG_space->dim_d, CG_space->z);
    } else {
//#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(n,m,input,LLT,output, temp_pre, temp2_pre)
//        for (int i = 0; i < n; ++i) {
//            vMemcpy(input+i*m, m, temp_pre[i].data());
//            temp2_pre[i] = LLT.LLTsolver.solve(temp_pre[i]);
//            vMemcpy(temp2_pre[i].data(), m, output+i*m);
//        }
    }
}

void preprocessing(QP_struct *qp) {
    qp->AT = get_BT(qp->A);
    qp->Lorg = new mfloat[qp->n];
    qp->Uorg = new mfloat[qp->n];
    vMemcpy(qp->l, qp->n, qp->Lorg);
    vMemcpy(qp->u, qp->n, qp->Uorg);

    qp->normborg = norm2(qp->b, qp->m);
    qp->normcorg = norm2(qp->c, qp->n);

    qp->cA = new mfloat[qp->n];
    qp->rA = new mfloat[qp->m];
    if (qp->use_scale)
        ruiz_equilibration(qp->A, &qp->A, qp->AT, &qp->AT, qp->cA, qp->rA);
    else {
        vSet(qp->cA, qp->n, 1.0);
        vSet(qp->rA, qp->m, 1.0);
    }

    xdoty(qp->b, qp->rA, qp->b, qp->m, true);
    xdoty(qp->c, qp->cA, qp->c, qp->n, true);
    xdoty(qp->l, qp->cA, qp->l, qp->n, false);
    xdoty(qp->u, qp->cA, qp->u, qp->n, false);


    for (int r = 0; r < qp->n; ++r) {
        for (int j = qp->Q.rowStart[r]; j < qp->Q.rowStart[r+1]; ++j) {
            qp->Q.value[j] *= qp->cA[r];
            qp->Q.value[j] *= qp->cA[qp->Q.column[j]];
        }
    }

    qp->bscale = max(1.0, norm2(qp->b, qp->m));
    qp->cscale = max(1.0, norm2(qp->c, qp->n));
//    cout << qp->bscale * qp->cscale << endl;
    mfloat invbscale = 1/qp->bscale;
    mfloat invcscale = 1/qp->cscale;

    ax(invbscale, qp->b, qp->b, qp->m);
    ax(invcscale, qp->c, qp->c, qp->n);
    ax(invbscale, qp->l, qp->l, qp->n);
    ax(invbscale, qp->u, qp->u, qp->n);

    qp->normb = norm2(qp->b, qp->m);
    qp->normc = norm2(qp->c, qp->n);

    mfloat bcratio = qp->bscale / qp->cscale;
    for (int r = 0; r < qp->n; ++r) {
        for (int j = qp->Q.rowStart[r]; j < qp->Q.rowStart[r+1]; ++j) {
            qp->Q.value[j] *= bcratio;
        }
    }
}


void ruiz_equilibration(sparseRowMatrix A, sparseRowMatrix *A_new, sparseRowMatrix AT, sparseRowMatrix *AT_new, mfloat *D, mfloat *E) {
    int m = A.nRow;
    int n = A.nCol;
//    A_new->nRow = m; A_new->nCol = n;
//    A_new->rowStart = new int[m+1];
//    vMemcpy(A.rowStart, m+1, A_new->rowStart);
//    A_new->column = new int[A.rowStart[m]];
//    vMemcpy(A.column, A.rowStart[m], A_new->column);
//    A_new->value = new mfloat[A.rowStart[m]];
//    vMemcpy(A.value, A.rowStart[m], A_new->value);
//
//    AT_new->nRow = n; AT_new->nCol = m;
//    AT_new->rowStart = new int[n+1];
//    vMemcpy(AT.rowStart, n+1, AT_new->rowStart);
//    AT_new->column = new int[AT.rowStart[n]];
//    vMemcpy(AT.column, AT.rowStart[n], AT_new->column);
//    AT_new->value = new mfloat[AT.rowStart[n]];
//    vMemcpy(AT.value, AT.rowStart[n], AT_new->value);

    if (E == nullptr)
        E = new mfloat[m];
    if (D == nullptr)
        D = new mfloat[n];

    for (int i = 0; i < m; ++i)
        E[i] = 1.0;
    for (int i = 0; i < n; ++i)
        D[i] = 1.0;

    mfloat *E_temp = new mfloat[m];
    mfloat *D_temp = new mfloat[n];

    int max_iter = 50;
    mfloat tol = 1e-6;

    for (int i = 0; i < max_iter; ++i) {
        for (int r = 0; r < m; ++r) {
            mfloat temp_value = 0.0;
            for (int j = A_new->rowStart[r]; j < A_new->rowStart[r+1]; ++j) {
                if (abs(A_new->value[j]) > temp_value )
                    temp_value = abs(A_new->value[j]);
            }
            temp_value = sqrt(temp_value);
            if (temp_value < 1e-2)
                temp_value = 1e-2;
            if (temp_value > 1e2)
                temp_value = 1e2;

            E_temp[r] = temp_value;
        }

        for (int c = 0; c < n; ++c) {
            mfloat temp_value = 0.0;
            for (int j = AT_new->rowStart[c]; j < AT_new->rowStart[c+1]; ++j) {
                if (abs(AT_new->value[j]) > temp_value )
                    temp_value = abs(AT_new->value[j]);
            }
            temp_value = sqrt(temp_value);
            if (temp_value < 1e-2)
                temp_value = 1e-2;
            if (temp_value > 1e2)
                temp_value = 1e2;

            D_temp[c] = temp_value;
        }

        for (int r = 0; r < m; ++r) {
            for (int j = A_new->rowStart[r]; j < A_new->rowStart[r+1]; ++j) {
                A_new->value[j] /= E_temp[r];
                A_new->value[j] /= D_temp[A_new->column[j]];
            }
        }

        for (int c = 0; c < n; ++c) {
            for (int j = AT_new->rowStart[c]; j < AT_new->rowStart[c+1]; ++j) {
                AT_new->value[j] /= D_temp[c];
                AT_new->value[j] /= E_temp[AT_new->column[j]];
            }
        }

//        print_spM(*A_new);

        for (int r = 0; r < m; ++r)
            E[r] /= E_temp[r];
        for (int c = 0; c < n; ++c)
            D[c] /= D_temp[c];

        mfloat norm1 = 0.0, norm2 = 0.0;
        for (int r = 0; r < m; ++r)
            norm1 += pow(1-E_temp[r]*E_temp[r], 2);
        for (int c = 0; c < n; ++c)
            norm2 += pow(1-D_temp[c]*D_temp[c], 2);
        mfloat error = max(sqrt(norm1), sqrt(norm2));
        if (error < tol) {
            cout << "ruiz iteration: " << i << endl;
            break;
        }

    }
}

void initial_Eigen() {
//    In.resize(Ainput.B.nRow, Ainput.B.nRow);
//    In.setIdentity();
//    wInum.resize(Ainput.B.nCol, Ainput.B.nCol);
//    wInum.setIdentity();
//    wInum2.resize(Ainput.B.nRow, Ainput.B.nRow);
//    wInum2.setIdentity();
//    Ei_B = get_eigen_spMat(Ainput.BT);
//    Ei_BT = Ei_B.transpose();
//    LLT.isEmpty = false;
//    tempW.resize(Ainput.B.nCol, 1);
//    temp_pre = new Eigen::VectorX<mfloat> [Ainput.dim_d];
//    temp2_pre = new Eigen::VectorX<mfloat> [Ainput.dim_d];
//    for (int i = 0; i < Ainput.dim_d; ++i) {
//        temp_pre[i].resize(Ainput.B.nRow, 1);
//        temp2_pre[i].resize(Ainput.B.nRow, 1);
//    }
}

void create_ADMM() {
}

void ADMM() {
}

void update_sigma_admm(){
}

void Eigen_init(QP_struct *QP_space) {
    QP_space->EigenA = get_eigen_spMat(QP_space->A);
    QP_space->EigenAT = get_eigen_spMat(QP_space->AT);
    QP_space->EigenAAT = QP_space->EigenA*QP_space->EigenA.transpose();

//    QP_space->EigenA = EigenA;
    QP_space->Eigen_rhs_w.resize(QP_space->n, 1);
    QP_space->Eigen_result_w.resize(QP_space->n, 1);
    QP_space->Eigen_rhs_y.resize(QP_space->m, 1);
    QP_space->Eigen_result_y.resize(QP_space->m, 1);
    QP_space->Eigen_rhs_dwdy.resize(QP_space->n+QP_space->m, 1);
    QP_space->Eigen_result_dwdy.resize(QP_space->n+QP_space->m, 1);
//    cout << EigenA << endl;
//    cout << EigenAAT << endl;
//    cout << QP_space->Q.value[0] << endl;
    QP_space->EigenQ = get_eigen_spMat(QP_space->Q);
    QP_space->EigenIn.resize(QP_space->n, QP_space->n);
    QP_space->EigenIn.setIdentity();
    QP_space->EigenIm.resize(QP_space->m, QP_space->m);
    QP_space->EigenIm.setIdentity();
    QP_space->EigenIQ = 1/QP_space->sigma*QP_space->EigenIn + QP_space->EigenQ;

    QP_space->Eigen_linear_AAT.LLTsolver.analyzePattern(QP_space->EigenAAT);
    QP_space->Eigen_linear_AAT.LLTsolver.factorize(QP_space->EigenAAT);
    QP_space->Eigen_linear_IQ.LLTsolver.analyzePattern(QP_space->EigenIQ);
    QP_space->Eigen_linear_IQ.LLTsolver.factorize(QP_space->EigenIQ);
}

int PARDISO_init(PARDISO_var *PDS, sparseRowMatrix *A) {
    PDS->n = A->nRow;
    if (A->nRow != A->nCol) {
        cout << "not square matrix" << endl;
        return 0;
    }
    int nnz = A->rowStart[A->nRow];
    PDS->debug = false;
//    PDS->ia = A.rowStart;
    PDS->ia = A->rowStart;
    PDS->ja = A->column;
    PDS->a = A->value;
//    vMemcpy(A.rowStart, A.nRow+1, PDS->ia);
////    PDS->ja = A.column;
//    vMemcpy(A.column, nnz, PDS->ja);
////    PDS->a = A.value;
//    vMemcpy(A.value, nnz, PDS->a);
    PDS->mtype = -2;        /* Real symmetric matrix */

    /* RHS and solution vectors. */
    PDS->x = (mfloat *) malloc(sizeof(mfloat) * PDS->n);

    PDS->nrhs = 1;          /* Number of right hand sides. */

    for (int i = 0; i < 64; i++ )
    {
        PDS->iparm[i] = 0;
    }
    PDS->iparm[0] = 1;         /* No solver default */
    PDS->iparm[1] = 3;         /* Fill-in reordering from METIS */
    PDS->iparm[3] = 0;         /* No iterative-direct algorithm */
    PDS->iparm[4] = 0;         /* No user fill-in reducing permutation */
    PDS->iparm[5] = 0;         /* Write solution into x */
    PDS->iparm[6] = 0;         /* Not in use */
    PDS->iparm[7] = 0;         /* Max numbers of iterative refinement steps */
    PDS->iparm[8] = 0;         /* Not in use */
    PDS->iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
    PDS->iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
    PDS->iparm[11] = 0;        /* Not in use */
    PDS->iparm[12] = 0;        /* Maximum weighted matching algorithm is switched-off (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy */
    PDS->iparm[13] = 0;        /* Output: Number of perturbed pivots */
    PDS->iparm[14] = 0;        /* Not in use */
    PDS->iparm[15] = 0;        /* Not in use */
    PDS->iparm[16] = 0;        /* Not in use */
    PDS->iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
    PDS->iparm[18] = -1;       /* Output: Mflops for LU factorization */
    PDS->iparm[19] = 0;        /* Output: Numbers of CG Iterations */
    PDS->iparm[34] = 1;        /* Zero based indexing */

    for (int i = 0; i < 64; i++ )
    {
        PDS->pt[i] = 0;
    }

/* -------------------------------------------------------------------- */
/* ..  Setup Pardiso control parameters.                                */
/* -------------------------------------------------------------------- */

//    PDS->error = 0;
//    PDS->solver = 0; /* use sparse direct solver */
//    pardisoinit (PDS->pt,  &(PDS->mtype), &(PDS->solver), PDS->iparm, PDS->dparm, &(PDS->error));

//    if (PDS->error != 0)
//    {
//        if (PDS->error == -10 )
//            printf("No license file found \n");
//        if (PDS->error == -11 )
//            printf("License is expired \n");
//        if (PDS->error == -12 )
//            printf("Wrong username or hostname \n");
//        return 1;
//    }
//    else
//        printf("[PARDISO]: License check was successful ... \n");

    /* Numbers of processors, value of OMP_NUM_THREADS */
//    char *var = getenv("OMP_NUM_THREADS");
//    if(var != NULL)
//        sscanf( var, "%d", &num_procs );
//    else {
//        printf("Set environment OMP_NUM_THREADS to 1");
//        num_procs = 1;
////        exit(1);
//    }
//    PDS->iparm[2]  = 1;

    PDS->maxfct = 1;		/* Maximum number of numerical factorizations.  */
    PDS->mnum   = 1;         /* Which factorization to use. */

    PDS->msglvl = 0;         /* Print statistical information  */
    PDS->error  = 0;         /* Initialize error flag */

/* -------------------------------------------------------------------- */
/* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
/*     notation.                                                        */
/* -------------------------------------------------------------------- */
//    for (int i = 0; i < A.nRow + 1; ++i)
//        (PDS->ia[i])++;
//    for (int i = 0; i < nnz; ++i)
//        (PDS->ja[i])++;

/* -------------------------------------------------------------------- */
/*  .. pardiso_chk_matrix(...)                                          */
/*     Checks the consistency of the given matrix.                      */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */

//    if (PDS->debug) {
////        pardiso_chkmatrix  (&(PDS->mtype), &(PDS->n), PDS->a, PDS->ia, PDS->ja, &(PDS->error));
//        if (PDS->error != 0) {
//            printf("\nERROR in consistency of matrix: %d", PDS->error);
//            exit(1);
//        }
//    }


/* -------------------------------------------------------------------- */
/* ..  pardiso_chkvec(...)                                              */
/*     Checks the given vectors for infinite and NaN values             */
/*     Input parameters (see PARDISO user manual for a description):    */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */

//    if (PDS->debug) {
////        pardiso_chkvec(&(PDS->n), &(PDS->nrhs), PDS->b, &(PDS->error));
//        if (PDS->error != 0) {
//            printf("\nERROR  in right hand side: %d", PDS->error);
//            exit(1);
//        }
//    }
/* -------------------------------------------------------------------- */
/* .. pardiso_printstats(...)                                           */
/*    prints information on the matrix to STDOUT.                       */
/*    Use this functionality only for debugging purposes                */
/* -------------------------------------------------------------------- */

//    if (PDS->debug) {
////        pardiso_printstats(&(PDS->mtype), &(PDS->n), PDS->a, PDS->ia, PDS->ja, &(PDS->nrhs), PDS->b, &(PDS->error));
//        if (PDS->error != 0) {
//            printf("\nERROR right hand side: %d", PDS->error);
//            exit(1);
//        }
//    }
//    -------------------------------------------------------------------- */
/* ..  Reordering and Symbolic Factorization.  This step also allocates */
/*     all memory that is necessary for the factorization.              */
/* -------------------------------------------------------------------- */
    PDS->phase = 11;

    PARDISO (PDS->pt, &(PDS->maxfct), &(PDS->mnum), &(PDS->mtype), &(PDS->phase),
             &(PDS->n), PDS->a, PDS->ia, PDS->ja, &(PDS->idum), &(PDS->nrhs), PDS->iparm, &(PDS->msglvl), &(PDS->ddum), &(PDS->ddum), &(PDS->error));

    if ( PDS->error != 0 )
    {
        printf ("\nERROR during symbolic factorization: " "%i", PDS->error);
        exit (1);
    }
//    printf ("\nReordering completed ... ");
//    printf ("\nNumber of nonzeros in factors = " "%i", PDS->iparm[17]);
//    printf ("\nNumber of factorization MFLOPS = " "%i", PDS->iparm[18]);
/* -------------------------------------------------------------------- */

    return 0;
}

int PARDISO_init(PARDISO_var *PDS, Eigen::SparseMatrix<double> *A) {
    PDS->n = A->rows();
    if (A->rows() != A->cols()) {
        cout << "not square matrix" << endl;
        return 0;
    }
    int nnz = A->nonZeros();
    PDS->debug = false;

    PDS->ia = A->outerIndexPtr();
    PDS->ja = A->innerIndexPtr();
    PDS->a = A->valuePtr();

//    PDS->ia = (int *) malloc(sizeof(int) * (A.rows()+1));
//    PDS->ja = (int *) malloc(sizeof(int) * nnz);
//    PDS->a = (mfloat *) malloc(sizeof(mfloat) * nnz);
//    vMemcpy(A.outerIndexPtr(), A.rows()+1, PDS->ia);
//    vMemcpy(A.innerIndexPtr(), nnz, PDS->ja);
//    vMemcpy(A.valuePtr(), nnz, PDS->a);

//    cout << PDS->ia[0] << ' ' << PDS->ia[1] << ' ' << PDS->ia[2] << endl;
//    cout << PDS->ja[0] << ' ' << PDS->ja[1] << ' ' << PDS->ja[2] << ' ' << PDS->ja[3] << endl;
//    cout << PDS->a[0] << ' ' << PDS->a[1] << ' ' << PDS->a[2] << ' ' << PDS->a[3] << endl;

    PDS->mtype = -2;        /* Real symmetric matrix */

    /* RHS and solution vectors. */

    PDS->nrhs = 1;          /* Number of right hand sides. */

    for (int i = 0; i < 64; i++ )
    {
        PDS->iparm[i] = 0;
    }
    PDS->iparm[0] = 1;         /* No solver default */
    PDS->iparm[1] = 3;         /* Fill-in reordering from METIS */
    PDS->iparm[3] = 0;         /* No iterative-direct algorithm */
    PDS->iparm[4] = 0;         /* No user fill-in reducing permutation */
    PDS->iparm[5] = 0;         /* Write solution into x */
    PDS->iparm[6] = 0;         /* Not in use */
    PDS->iparm[7] = 0;         /* Max numbers of iterative refinement steps */
    PDS->iparm[8] = 0;         /* Not in use */
    PDS->iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
    PDS->iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
    PDS->iparm[11] = 0;        /* Not in use */
    PDS->iparm[12] = 0;        /* Maximum weighted matching algorithm is switched-off (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy */
    PDS->iparm[13] = 0;        /* Output: Number of perturbed pivots */
    PDS->iparm[14] = 0;        /* Not in use */
    PDS->iparm[15] = 0;        /* Not in use */
    PDS->iparm[16] = 0;        /* Not in use */
    PDS->iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
    PDS->iparm[18] = -1;       /* Output: Mflops for LU factorization */
    PDS->iparm[19] = 0;        /* Output: Numbers of CG Iterations */
    PDS->iparm[34] = 1;        /* Zero based indexing */

    for (int i = 0; i < 64; i++ )
    {
        PDS->pt[i] = 0;
    }

    PDS->maxfct = 1;		/* Maximum number of numerical factorizations.  */
    PDS->mnum   = 1;         /* Which factorization to use. */

    PDS->msglvl = 0;         /* Print statistical information  */
    PDS->error  = 0;         /* Initialize error flag */
    PDS->phase = 11;

//    cout << PDS->ia[0] << ' ' << PDS->ia[1] << ' ' << PDS->ia[2] << endl;
//    cout << PDS->ja[0] << ' ' << PDS->ja[1] << ' ' << PDS->ja[2] << ' ' << PDS->ja[3] << endl;
//    cout << PDS->a[0] << ' ' << PDS->a[1] << ' ' << PDS->a[2] << ' ' << PDS->a[3] << endl;
    PARDISO (PDS->pt, &(PDS->maxfct), &(PDS->mnum), &(PDS->mtype), &(PDS->phase),
             &(PDS->n), PDS->a, PDS->ia, PDS->ja, &(PDS->idum), &(PDS->nrhs), PDS->iparm, &(PDS->msglvl), &(PDS->ddum), &(PDS->ddum), &(PDS->error));

//    cout << PDS->ia[0] << ' ' << PDS->ia[1] << ' ' << PDS->ia[2] << endl;
//    cout << PDS->ja[0] << ' ' << PDS->ja[1] << ' ' << PDS->ja[2] << ' ' << PDS->ja[3] << endl;
//    cout << PDS->a[0] << ' ' << PDS->a[1] << ' ' << PDS->a[2] << ' ' << PDS->a[3] << endl;

    if ( PDS->error != 0 )
    {
        printf ("\nERROR during symbolic factorization: " "%i", PDS->error);
        exit (1);
    }
//    printf ("\nReordering completed ... ");
//    printf ("\nNumber of nonzeros in factors = " "%i", PDS->iparm[17]);
//    printf ("\nNumber of factorization MFLOPS = " "%i", PDS->iparm[18]);
/* -------------------------------------------------------------------- */

    return 0;
}

int PARDISO_numerical_fact(PARDISO_var *PDS) {
    /* -------------------------------------------------------------------- */
    /* ..  Numerical factorization.                                         */
    /* -------------------------------------------------------------------- */
    PDS->phase = 22;
//    PDS->iparm[32] = 0; /* compute determinant */

//    pardiso (PDS->pt, &(PDS->maxfct), &(PDS->mnum), &(PDS->mtype), &(PDS->phase),
//             &(PDS->n), PDS->a, PDS->ia, PDS->ja, &(PDS->idum), &(PDS->nrhs),
//             PDS->iparm, &(PDS->msglvl), &(PDS->ddum), &(PDS->ddum), &(PDS->error), PDS->dparm);

//    cout << PDS->ia[0] << ' ' << PDS->ia[1] << ' ' << PDS->ia[2] << endl;
//    cout << PDS->ja[0] << ' ' << PDS->ja[1] << ' ' << PDS->ja[2] << ' ' << PDS->ja[3] << endl;
//    cout << PDS->a[0] << ' ' << PDS->a[1] << ' ' << PDS->a[2] << ' ' << PDS->a[3] << endl;

    PARDISO (PDS->pt, &(PDS->maxfct), &(PDS->mnum), &(PDS->mtype), &(PDS->phase),
             &(PDS->n), PDS->a, PDS->ia, PDS->ja, &(PDS->idum), &(PDS->nrhs), PDS->iparm, &(PDS->msglvl), &(PDS->ddum), &(PDS->ddum), &(PDS->error));

//    cout << PDS->ia[0] << ' ' << PDS->ia[1] << ' ' << PDS->ia[2] << endl;
//    cout << PDS->ja[0] << ' ' << PDS->ja[1] << ' ' << PDS->ja[2] << ' ' << PDS->ja[3] << endl;
//    cout << PDS->a[0] << ' ' << PDS->a[1] << ' ' << PDS->a[2] << ' ' << PDS->a[3] << endl;

    if ( PDS->error != 0 )
    {
        printf ("\nERROR during numerical factorization: " "%i", PDS->error);
        exit (2);
    }
//    printf ("\nFactorization completed ... ");

    PDS->isInitialized = true;
    return 0;
}

int PARDISO_solve(PARDISO_var *PDS, mfloat *rhs) {
    /* ---------------------------------numerical_fact_----------------------------------- */
    /* ..  Back substitution and iterative refinement.                      */
    /* -------------------------------------------------------------------- */
    PDS->phase = 33;

//    PDS->iparm[7] = 0;       /* Max numbers of iterative refinement steps. */

    if (!(PDS->isInitialized)) {
        cout << "PARDISO not initialized" << endl;
        return 0;
    }

    PDS->b = rhs;
//    vMemcpy(rhs, PDS->n, PDS->b);

//    pardiso (PDS->pt, &(PDS->maxfct), &(PDS->mnum), &(PDS->mtype), &(PDS->phase),
//             &(PDS->n), PDS->a, PDS->ia, PDS->ja, &(PDS->idum), &(PDS->nrhs),
//             PDS->iparm, &(PDS->msglvl), PDS->b, PDS->x, &(PDS->error), PDS->dparm);

    PARDISO (PDS->pt, &(PDS->maxfct), &(PDS->mnum), &(PDS->mtype), &(PDS->phase),
             &(PDS->n), PDS->a, PDS->ia, PDS->ja, &(PDS->idum), &(PDS->nrhs), PDS->iparm, &(PDS->msglvl), PDS->b, PDS->x, &(PDS->error));


    if (PDS->error != 0) {
        printf("\nERROR during solution: %d", PDS->error);
        exit(3);
    }

//    printf("\nSolve completed ... ");
//    printf("\nThe solution of the system is: ");
//    for (int i = 0; i < min(PDS->n, 5); i++) {
//        printf("\n x [%d] = % f", i, PDS->x[i] );
//    }
//    printf ("\n");

    return 0;
}

int PARDISO_release(PARDISO_var *PDS) {
    /* -------------------------------------------------------------------- */
    /* ..  Termination and release of memory.                               */
    /* -------------------------------------------------------------------- */
    PDS->phase = -1;                 /* Release internal memory. */

//    pardiso (PDS->pt, &(PDS->maxfct), &(PDS->mnum), &(PDS->mtype), &(PDS->phase),
//             &(PDS->n), PDS->a, PDS->ia, PDS->ja, &(PDS->idum), &(PDS->nrhs),
//             PDS->iparm, &(PDS->msglvl), &(PDS->ddum), &(PDS->ddum), &(PDS->error), PDS->dparm);

    PARDISO (PDS->pt, &(PDS->maxfct), &(PDS->mnum), &(PDS->mtype), &(PDS->phase),
             &(PDS->n), &(PDS->ddum), PDS->ia, PDS->ja, &(PDS->idum), &(PDS->nrhs), PDS->iparm, &(PDS->msglvl), &(PDS->ddum), &(PDS->ddum), &(PDS->error));
//    printf("\nRelease completed ... ");

    return 0;
}

Eigen::SparseMatrix<mfloat> concat_4_spMat(Eigen::SparseMatrix<mfloat> A, Eigen::SparseMatrix<mfloat> B, Eigen::SparseMatrix<mfloat> C, Eigen::SparseMatrix<mfloat> D) {
    typedef Eigen::Triplet<mfloat> T;
    std::vector<T> tripletList;
    if (A.rows() != B.rows() || A.cols() != C.cols() || B.cols() != D.cols() || C.rows() != D.rows()) {
        printf("ERROR: concat_4_spMat, dimension inconsistent\n");
        exit(1);
    }
    tripletList.reserve(A.nonZeros()+B.nonZeros()+C.nonZeros()+D.nonZeros());
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<mfloat>::InnerIterator it(A, k); it; ++it) {
            tripletList.emplace_back(it.row(), it.col(), it.value());
        }
    }

    for (int k = 0; k < B.outerSize(); ++k) {
        for (Eigen::SparseMatrix<mfloat>::InnerIterator it(B, k); it; ++it) {
            tripletList.emplace_back(it.row(), it.col()+A.cols(), it.value());
        }
    }

    for (int k = 0; k < C.outerSize(); ++k) {
        for (Eigen::SparseMatrix<mfloat>::InnerIterator it(C, k); it; ++it) {
            tripletList.emplace_back(it.row()+A.rows(), it.col(), it.value());
        }
    }

    for (int k = 0; k < D.outerSize(); ++k) {
        for (Eigen::SparseMatrix<mfloat>::InnerIterator it(D, k); it; ++it) {
            tripletList.emplace_back(it.row()+A.rows(), it.col()+A.cols(), it.value());
        }
    }

    Eigen::SparseMatrix<mfloat> mat(A.rows()+C.rows(), A.cols()+B.cols());
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
}

void test_concat_4_spMat() {
    sparseRowMatrix Q;
    Q.value = new mfloat [4];
    Q.nRow = 2; Q.nCol = 2;
    Q.value[0] = 4; Q.value[1] = 2; Q.value[2] = 2; Q.value[3] = 6;
    Q.rowStart = new int [3];
    Q.rowStart[0] = 0; Q.rowStart[1] = 2; Q.rowStart[2] = 4;
    Q.column = new int [4];
    Q.column[0] = 0; Q.column[1] = 1; Q.column[2] = 0; Q.column[3] = 1;

    Eigen::SparseMatrix<mfloat> Q_eigen = get_eigen_spMat(Q);

    auto QQQQ = concat_4_spMat(Q_eigen, Q_eigen, Q_eigen, Q_eigen);
    std::cout << QQQQ << std::endl;

}

void CSR_to_COO(sparseRowMatrix Q, sparseMatrixCOO *Q_coo) {
    Q_coo->nRow = Q.nRow;
    Q_coo->nCol = Q.nCol;
    Q_coo->nElements = Q.rowStart[Q.nRow];
    Q_coo->elements = new sparseMatrixElement [Q_coo->nElements];

    int count = 0;
    for (int r = 0; r < Q.nRow; ++r) {
        for (int j = Q.rowStart[r]; j < Q.rowStart[r+1]; ++j) {
            Q_coo->elements[count].row = r;
            Q_coo->elements[count].col = Q.column[j];
            Q_coo->elements[count].value = Q.value[j];
            count++;
        }
    }
}

void printSparseMatrix(sparseMatrixCOO matrix) {
    printf("Sparse Matrix (%d x %d):\n", matrix.nRow, matrix.nCol);
    for (int i = 0; i < matrix.nElements; i++) {
        sparseMatrixElement element = matrix.elements[i];
        printf("(%d, %d) = %f\n", element.row, element.col, element.value);
    }
}

void test_CSR_to_COO() {
    sparseRowMatrix Q;
    Q.value = new mfloat [4];
    Q.nRow = 2; Q.nCol = 2;
    Q.value[0] = 4; Q.value[1] = 2; Q.value[2] = 2; Q.value[3] = 6;
    Q.rowStart = new int [3];
    Q.rowStart[0] = 0; Q.rowStart[1] = 2; Q.rowStart[2] = 4;
    Q.column = new int [4];
    Q.column[0] = 0; Q.column[1] = 1; Q.column[2] = 0; Q.column[3] = 1;

    sparseMatrixCOO Q_coo;
    CSR_to_COO(Q, &Q_coo);
    printSparseMatrix(Q_coo);
}

void test_eigen_sub() {
    sparseRowMatrix Q;
    Q.value = new mfloat [4];
    Q.nRow = 2; Q.nCol = 2;
    Q.value[0] = 4; Q.value[1] = 2; Q.value[2] = 2; Q.value[3] = 6;
    Q.rowStart = new int [3];
    Q.rowStart[0] = 0; Q.rowStart[1] = 2; Q.rowStart[2] = 4;
    Q.column = new int [4];
    Q.column[0] = 0; Q.column[1] = 1; Q.column[2] = 0; Q.column[3] = 1;

    sparseRowMatrix A;
    A.value = new mfloat [4];
    A.nRow = 2; A.nCol = 2;
    A.value[0] = 1; A.value[1] = 1; A.value[2] = 2; A.value[3] = -1;
    A.rowStart = new int [3];
    A.rowStart[0] = 0; A.rowStart[1] = 2; A.rowStart[2] = 4;
    A.column = new int [4];
    A.column[0] = 0; A.column[1] = 1; A.column[2] = 0; A.column[3] = 1;

    bool *U_idx = new bool[2];
    U_idx[0] = false; U_idx[1] = true;
    int *idx_sum = new int [2];
    idx_sum[0] = (U_idx[0])? 0 : 1;
    for (int i = 1; i < 2; ++i) idx_sum[i] = (U_idx[i]) ? idx_sum[i-1] : (idx_sum[i-1] + 1);
    cout << idx_sum[0] << ' ' << idx_sum[1] << endl;
    std::vector<Eigen::Triplet<mfloat>> tri;
    Eigen::SparseMatrix<mfloat> A_sub;
//    tri.reserve(4);

//    {
        Eigen::SparseMatrix<mfloat> Q_sub;
        get_eigen_sub_spMat_A(A, U_idx, tri, idx_sum, &A_sub);

        cout << A_sub.rows() << endl;
        cout << A_sub << endl;
//    }

//    Q_sub.de
//    Q_sub = new Eigen::SparseMatrix<mfloat>(2,2);
//    {
        U_idx[0] = 1; U_idx[1] = 1;
        idx_sum[0] = (U_idx[0])? 0 : 1;
        for (int i = 1; i < 2; ++i) idx_sum[i] = (U_idx[i]) ? idx_sum[i-1] : (idx_sum[i-1] + 1);
        cout << idx_sum[0] << ' ' << idx_sum[1] << endl;

        get_eigen_sub_spMat_Q(Q, U_idx, tri, idx_sum, &Q_sub);
        cout << Q_sub << endl;
//    }

//    delete Q_sub;
//    Q_sub = new Eigen::SparseMatrix<mfloat>(1,1);
//    {
        U_idx[0] = 1; U_idx[1] = 0;
        idx_sum[0] = (U_idx[0])? 0 : 1;
        for (int i = 1; i < 2; ++i) idx_sum[i] = (U_idx[i]) ? idx_sum[i-1] : (idx_sum[i-1] + 1);
        cout << idx_sum[0] << ' ' << idx_sum[1] << endl;

        get_eigen_sub_spMat_Q(Q, U_idx, tri, idx_sum, &Q_sub);
        cout << Q_sub << endl;
//    }
}

void SSN_sub_case1(QP_struct *qp) {

    auto time0 = time_now();
    int m = qp->m;
    int n = qp->n;
    qp->proj_idx_sum[0] = (qp->proj_idx[0]) ? 0 : 1;
    for (int i = 1; i < n; ++i)
        qp->proj_idx_sum[i] = (qp->proj_idx[i]) ? qp->proj_idx_sum[i - 1] : (qp->proj_idx_sum[i - 1] + 1);

    auto get_eigen_sub_time0 = time_now();
    get_eigen_sub_spMat_Q(qp->Q, qp->proj_idx, qp->tripletList_Qsub, qp->proj_idx_sum, &qp->Eigen_Q_sub);
    qp->get_eigen_sub_time += time_since(get_eigen_sub_time0);

    for (int i = 0; i < n; ++i) qp->zero_idx[i] = !qp->proj_idx[i];

    auto partial_CR_spMV_time0 = time_now();
    partial_CR_spMV(qp->Q, qp->SSN_grad, n, n, qp->SSN_sub_case1_temp1, qp->proj_idx, qp->zero_idx);
    qp->partial_CR_spMV_time += time_since(partial_CR_spMV_time0);

    auto partial_axpy_time0 = time_now();
    partial_axpy(-qp->sigma / (1 + qp->tau / qp->sigma), qp->SSN_sub_case1_temp1, qp->SSN_grad, qp->SSN_r1_new, n,
                 qp->proj_idx);
    qp->partial_axpy_time += time_since(partial_axpy_time0);

    qp->Eigen_Q_sub *= qp->sigma;
    qp->diagNUM.resize(qp->U_idx.number, 1);
    qp->diagNUM.setConstant(1 + qp->tau / qp->sigma);
    qp->Eigen_Q_sub += qp->diagNUM.asDiagonal();

    // if A exists
    get_eigen_sub_time0 = time_now();
    get_eigen_sub_spMat_A(qp->A, qp->proj_idx, qp->tripletList_Qsub, qp->proj_idx_sum, &qp->Eigen_A_sub);
    qp->get_eigen_sub_time += time_since(get_eigen_sub_time0);

    partial_CR_spMV_time0 = time_now();
    partial_CR_spMV(qp->A, qp->SSN_sub_case1_temp1, m, n, qp->SSN_sub_case1_temp2, nullptr, qp->proj_idx);
    partial_CR_spMV(qp->A, qp->SSN_r1_new, m, n, qp->SSN_sub_case1_temp3, nullptr, qp->proj_idx);
    qp->partial_CR_spMV_time += time_since(partial_CR_spMV_time0);

    auto axpy_time0 = time_now();
    axpy(qp->sigma / (1 + qp->tau / qp->sigma), qp->SSN_sub_case1_temp2, qp->SSN_sub_case1_temp3,
         qp->SSN_sub_case1_temp3, m);
    axpy(1, qp->SSN_grad + n, qp->SSN_sub_case1_temp3, qp->SSN_sub_case1_temp3, m);
    qp->axpy_time += time_since(axpy_time0);

    partial_CR_spMV_time0 = time_now();
    partial_CR_spMV(qp->AT, qp->SSN_sub_case1_temp3, n, m, qp->SSN_rhs, qp->proj_idx, nullptr);
    qp->partial_CR_spMV_time += time_since(partial_CR_spMV_time0);

    partial_axpy_time0 = time_now();
    partial_axpby(-qp->sigma / (1 + qp->tau / qp->sigma), qp->SSN_sub_case1_temp1, qp->sigma * qp->sigma / qp->tau,
                  qp->SSN_rhs, qp->SSN_rhs, n, qp->proj_idx);
    partial_axpy(1, qp->SSN_grad, qp->SSN_rhs, qp->SSN_rhs, n, qp->proj_idx);
    qp->partial_axpy_time += time_since(partial_axpy_time0);

    auto eigen_matrix_mult_time0 = time_now();
    auto temp = qp->Eigen_A_sub.transpose() * qp->Eigen_A_sub;
    qp->eigen_matrix_mult_time += time_since(eigen_matrix_mult_time0);
//    cout << temp.norm() << endl;

    qp->Eigen_Q_sub += (qp->sigma * qp->sigma * (1 + qp->tau / qp->sigma) / qp->tau) * temp;


//    qp->Eigen_Q_sub += qp->sigma*qp->sigma*(1+qp->tau/qp->sigma)/qp->tau*(qp->Eigen_A_sub.transpose()*qp->Eigen_A_sub);

//    cout << qp->Eigen_Q_sub.coeff(1,1) << endl;
//    cout << qp->Eigen_Q_sub.coeff(1,2) << endl;
//    cout << qp->Eigen_Q_sub.norm() << endl;

    auto eigen_solver_time0 = time_now();

    if (!qp->use_mkl) {
        qp->Eigen_rhs_w.resize(qp->U_idx.number, 1);
        mfloat *p = qp->Eigen_rhs_w.data();
        int count = 0;
        int *mask = qp->U_idx.vec;
        for (int i = 0; i < qp->U_idx.number; ++i)
            p[i] = qp->SSN_rhs[mask[i]];

        qp->Eigen_result_w.resize(qp->U_idx.number, 1);
//    qp->Eigen_result_w.resize(qp->U_idx.number, 1);
//    cout << norm2(p, qp->U_idx.number) << endl;


//        qp->Eigen_Q_sub.makeCompressed();
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > chol(qp->Eigen_Q_sub);
//    auto solver_analyze_time0 = time_now();
//    qp->Eigen_linear_IQ.LLTsolver.analyzePattern(qp->Eigen_Q_sub);
//    qp->solver_analyze_time += time_since(solver_analyze_time0);
//
//    auto solver_factorize_time0 = time_now();
//    qp->Eigen_linear_IQ.LLTsolver.factorize(qp->Eigen_Q_sub);
//    qp->solver_factorize_time += time_since(solver_factorize_time0);

        auto solver_solve_time0 = time_now();
        qp->Eigen_result_w = chol.solve(qp->Eigen_rhs_w);
        qp->solver_solve_time += time_since(solver_solve_time0);

        ax(0, qp->SSN_dwdy, qp->SSN_dwdy, n);
        p = qp->Eigen_result_w.data();
        for (int i = 0; i < qp->U_idx.number; ++i) {
            qp->SSN_dwdy[mask[i]] = p[i];
        }
    } else if (qp->use_mkl) {
        int *mask = qp->U_idx.vec;
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) shared(mask, qp) default(none)
        for (int i = 0; i < qp->U_idx.number; ++i)
            qp->SSN_rhs_compressed[i] = qp->SSN_rhs[mask[i]];

//        sparseRowMatrix Q;
//        sparseRowMatrix Q_upper;
//        upper_sparseRowMatrix(Q, &Q_upper);

//        if (qp->PDS_IQ != nullptr)
//            PARDISO_release(qp->PDS_IQ);
//        qp->PDS_IQ = new PARDISO_var;
        Eigen::SparseMatrix<double> temp = qp->Eigen_Q_sub.triangularView<Eigen::Lower>();
//        cout << temp.nonZeros() << ' ' << qp->Eigen_Q_sub.nonZeros() << endl;
//        auto rowStart =  qp->Eigen_Q_sub.outerIndexPtr();
//        auto colIdx =  qp->Eigen_Q_sub.innerIndexPtr();
//        auto val =  qp->Eigen_Q_sub.valuePtr();
//        cout << rowStart[0] << ' ' << rowStart[1] << ' ' << rowStart[2] << endl;
//        cout << colIdx[0] << ' ' << colIdx[1] << ' ' << colIdx[2] << ' ' << colIdx[3] << ' ' << colIdx[4] << endl;
//        cout << val[0] << ' ' << val[1] << ' ' << val[2] << ' ' << val[3] << ' ' << val[4] << endl;
        auto time0 = time_now();
        PARDISO_init(qp->PDS_Q2,  &temp);
        qp->solver_analyze_time += time_since(time0);
//        mkl_sparse_convert_csr()

        time0 = time_now();
        PARDISO_numerical_fact(qp->PDS_Q2);
        qp->solver_factorize_time += time_since(time0);

        time0 = time_now();
        PARDISO_solve(qp->PDS_Q2, qp->SSN_rhs_compressed);
        qp->solver_solve_time += time_since(time0);

        ax(0, qp->SSN_dwdy, qp->SSN_dwdy, n);
        auto p = qp->PDS_Q2->x;
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) shared(mask, qp, p) default(none)
        for (int i = 0; i < qp->U_idx.number; ++i) {
            qp->SSN_dwdy[mask[i]] = p[i];
        }
        qp->fact_times++;
//        PARDISO_release(qp->PDS_IQ);
    }


    qp->eigen_solver_time += time_since(eigen_solver_time0);

//    cout << norm2(p, qp->U_idx.number) << endl;

    partial_axpy_time0 = time_now();
    partial_axpy(1/(1+qp->tau/qp->sigma), qp->SSN_grad, qp->SSN_dwdy, qp->SSN_dwdy, n, qp->zero_idx);
    qp->partial_axpy_time += time_since(partial_axpy_time0);


    partial_CR_spMV_time0 = time_now();
    partial_CR_spMV(qp->A, qp->SSN_dwdy, m, n, qp->SSN_dwdy+n, nullptr, qp->proj_idx);
    qp->partial_CR_spMV_time += time_since(partial_CR_spMV_time0);

    axpy_time0 = time_now();
    axpby(qp->sigma/qp->tau, qp->SSN_sub_case1_temp3, -(1+qp->sigma/qp->tau), qp->SSN_dwdy+n, qp->SSN_dwdy+n, m);
    qp->axpy_time += time_since(axpy_time0);

    qp->case1_time += time_since(time0);
}

mfloat vec_max(const mfloat *x, int len) {
    mfloat max = x[0];
    for (int i = 1; i < len; ++i) {
        if (x[i] > max) max = x[i];
    }
    return max;
}

void test_eigen_segment_set() {
    int vectorSize = 10;
    double firstConstant = 1.0;
    double secondConstant = 2.0;

    Eigen::VectorXd myVector(vectorSize);
    myVector.segment(0, 3).setConstant(firstConstant);
    myVector.segment(5, 3).setConstant(secondConstant);

    std::cout << "Vector: " << myVector.transpose() << std::endl;
}

void test_MKL_pardiso() {
    sparseRowMatrix Q;
    Q.value = new mfloat [4];
    Q.nRow = 2; Q.nCol = 2;
    Q.value[0] = 4; Q.value[1] = 2; Q.value[2] = 2; Q.value[3] = 6;
    Q.rowStart = new int [3];
    Q.rowStart[0] = 0; Q.rowStart[1] = 2; Q.rowStart[2] = 4;
    Q.column = new int [4];
    Q.column[0] = 0; Q.column[1] = 1; Q.column[2] = 0; Q.column[3] = 1;


    auto eigen_Q = get_eigen_spMat(Q);
    Eigen::SparseMatrix<double> temp = eigen_Q.triangularView<Eigen::Lower>();
    auto p = new PARDISO_var;
    auto rhs = new mfloat [2];
    rhs[0] = 1.5; rhs[1] = 2.5;
    auto temp_r = temp.outerIndexPtr();
    auto temp_c = temp.innerIndexPtr();
    auto temp_v = temp.valuePtr();
//    cout << temp_r[0] << ' ' << temp_r[1] << ' ' << temp_r[2] << endl;
//    cout << temp_c[0] << ' ' << temp_c[1] << ' ' << temp_c[2] << ' ' << temp_c[3] << endl;
//    cout << temp_v[0] << ' ' << temp_v[1] << ' ' << temp_v[2] << ' ' << temp_v[3] << endl;

    PARDISO_init(p, &temp);
    cout << temp_r << ' ' << temp_r[0] << endl;
//    cout << temp_r[0] << ' ' << temp_r[1] << ' ' << temp_r[2] << endl;
//    cout << temp_c[0] << ' ' << temp_c[1] << ' ' << temp_c[2] << ' ' << temp_c[3] << endl;
//    cout << temp_v[0] << ' ' << temp_v[1] << ' ' << temp_v[2] << ' ' << temp_v[3] << endl;
    PARDISO_numerical_fact(p);
    cout << temp_r << ' ' << temp_r[0] << endl;
//    cout << temp_r[0] << ' ' << temp_r[1] << ' ' << temp_r[2] << endl;
//    cout << temp_c[0] << ' ' << temp_c[1] << ' ' << temp_c[2] << ' ' << temp_c[3] << endl;
//    cout << temp_v[0] << ' ' << temp_v[1] << ' ' << temp_v[2] << ' ' << temp_v[3] << endl;
    PARDISO_solve(p, rhs);
//    cout << temp_r[0] << ' ' << temp_r[1] << ' ' << temp_r[2] << endl;
//    cout << temp_c[0] << ' ' << temp_c[1] << ' ' << temp_c[2] << ' ' << temp_c[3] << endl;
//    cout << temp_v[0] << ' ' << temp_v[1] << ' ' << temp_v[2] << ' ' << temp_v[3] << endl;
    printf("\nThe solution of the system is: ");
    for (int i = 0; i < min(p->n, 5); i++) {
        printf("\n x [%d] = % f", i, p->x[i] );
    }
//    cout << endl;
//    cout << Q.rowStart[0] << ' ' << Q.rowStart[1] << ' ' << Q.rowStart[2] << endl;
//    cout << Q.column[0] << ' ' << Q.column[1] << ' ' << Q.column[2] << ' ' << Q.column[3] << endl;
//    cout << Q.value[0] << ' ' << Q.value[1] << ' ' << Q.value[2] << ' ' << Q.value[3] << endl;
//    PARDISO_release(p);
}

void upper_sparseRowMatrix(sparseRowMatrix A, sparseRowMatrix *A_upper) {
    A_upper->nRow = A.nRow;
    A_upper->nCol = A.nCol;
    int nnz = A.rowStart[A.nRow];
    A_upper->rowStart = new int [A.nRow+1];
    A_upper->column = new int [(A.nRow+nnz)/2];
    A_upper->value = new mfloat [(A.nRow+nnz)/2];
    int count = 0;
    A_upper->rowStart[0] = 0;
    for (int i = 0; i < A.nRow; ++i) {
        for (int j = A.rowStart[i]; j < A.rowStart[i+1]; ++j) {
            if (A.column[j] >= i) {
                A_upper->column[count] = A.column[j];
                A_upper->value[count] = A.value[j];
                count++;
            }
            A_upper->rowStart[i+1] = count;
        }
    }
}

void load_para(QP_struct *qp, input_parameters para) {
    qp->kappa = para.kappa;
    qp->gamma = para.gamma;
    qp->max_ADMM_iter = para.max_ADMM_iter;
    qp->max_ALM_iter = para.max_ALM_iter;
    qp->sigma = para.sigma;
    qp->use_mkl = para.use_mkl;
    qp->use_scale = para.use_scale;

    cout << "kappa: " << qp->kappa << endl;
    cout << "gamma: " << qp->gamma << endl;
    cout << "max_ADMM_iter: " << qp->max_ADMM_iter << endl;
    cout << "max_ALM_iter: " << qp->max_ALM_iter << endl;
    cout << "sigma: " << qp->sigma << endl;
    cout << "use_mkl: " << qp->use_mkl << endl;
    cout << "use_scale: " << qp->use_scale << endl;
}

void vSet(mfloat *x, int len, mfloat val) {
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(len, x, val)
    for (int i = 0; i < len; ++i) {
        x[i] = val;
    }
}

void MKL_init(QP_struct *qp) {
    mfloat inv_sigma = 1/qp->sigma;
    int n = qp->n, m = qp->m;

    qp->PDS_A = new PARDISO_var;
    qp->PDS_A->x = new mfloat [m];
//    qp->PDS_A->b = new mfloat [m];
    qp->EigenA = get_eigen_spMat(qp->A);
    qp->EigenAT = get_eigen_spMat(qp->AT);
    qp->EigenAAT = qp->EigenA*qp->EigenA.transpose();
    qp->EigenAAT = qp->EigenAAT.triangularView<Eigen::Lower>();

    PARDISO_init(qp->PDS_A, &qp->EigenAAT);
    PARDISO_numerical_fact(qp->PDS_A);

    upper_sparseRowMatrix(qp->Q, &qp->MKL_IQ);
    int *rowStart = qp->MKL_IQ.rowStart;
    mfloat *value = qp->MKL_IQ.value;
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(n, rowStart, value, inv_sigma)
    for (int i = 0; i < n; ++i) {
        value[rowStart[i]] += inv_sigma;
    }
    qp->PDS_IQ = new PARDISO_var;
    qp->PDS_IQ->x = new mfloat [n];
//    qp->PDS_IQ->b = new mfloat [n];
    PARDISO_init(qp->PDS_IQ, &qp->MKL_IQ);
    PARDISO_numerical_fact(qp->PDS_IQ);

}