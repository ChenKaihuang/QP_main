//
// Created by chenkaihuang on 6/7/22.
//

#include <string.h>
#include "algorithm.h"

//#define USE_CBLAS 1

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

void proj(const mfloat* input, mfloat* output, int len, const mfloat* l, const mfloat* u) {
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(len, input, output, l ,u)
    for (int i = 0; i < len; ++i) {
        if (input[i] < l[i])
            output[i] = l[i];
        else if (input[i] > u[i])
            output[i] = u[i];
        else
            output[i] = input[i];
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

/// Todo:
void axpy(const mfloat a, const mfloat* x, const mfloat* y, mfloat* z, const int len){
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(len, y, z, a, x)
        for (int i = 0; i < len; ++i)
            z[i] = y[i] + a*x[i];
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

mfloat* xdoty(const mfloat* x, const mfloat* yf, int len) {
    mfloat* zf = new mfloat [len];
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(len,zf,x,yf)
    for (int i = 0; i < len; ++i)
        zf[i] = x[i] * yf[i];
    return zf;
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
    Bf.rowStart = new int [BT.nCol];
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

void spMV_R(sparseRowMatrix A, const mfloat* x, int m, int n, mfloat* Ax) {
    mfloat* value = A.value;
    int* rowStart = A.rowStart;
    int* column = A.column;
    assert(A.nCol == n);
    for (int i = 0; i < m; ++i) {
        mfloat temp = 0;
        for (int j = rowStart[i]; j < rowStart[i+1]; ++j) {
            assert(column[j] < n);
            temp += value[j]*x[column[j]];
        }
        Ax[i] = temp;
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
    BiCG_space->U_idx = U_idx;
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
    QP_space->l[0] = -1e300; QP_space->l[1] = -1e300;

    QP_space->u = new mfloat [2];
    QP_space->u[0] = 1e300; QP_space->u[1] = 1e300;

    QP_space->m = 2; QP_space->n = 2;

    input_parameters para;
    sGSADMM_QP(Q, A, QP_space->b, QP_space->c, QP_space->l, QP_space->u, QP_space->m, QP_space->n, para);
}


void sGSADMM_QP(sparseRowMatrix Q, sparseRowMatrix A, mfloat* b, mfloat* c, mfloat* l, mfloat* u, int m, int n, input_parameters para) {
    QP_struct *QP_space = new QP_struct;
    QP_space->Q = Q;
    QP_space->A = A;
    QP_space->b = b;
    QP_space->c = c;
    QP_space->l = l;
    QP_space->u = u;
    QP_space->m = m;
    QP_space->n = n;
    QP_space->input_para = para;
    sGSADMM_QP_init(QP_space);

    while (QP_space->iter < QP_space->maxADMMiter) {
        sGSADMM_QP_update_y_first(QP_space);
        sGSADMM_QP_update_w_first(QP_space);
        sGSADMM_QP_update_z(QP_space);
        sGSADMM_QP_update_w_second(QP_space);
        sGSADMM_QP_update_y_second(QP_space);
        sGSADMM_QP_update_x(QP_space);

        sGSADMM_QP_compute_status(QP_space);
        update_sigma(QP_space);
        sGSADMM_QP_print_status(QP_space);
        QP_space->iter++;
    }

}

void sGSADMM_QP_init(QP_struct *QP_space) {
    int n = QP_space->n, m = QP_space->m;
    QP_space->w = new mfloat [n];
    QP_space->w_bar = new mfloat [n];
    QP_space->y = new mfloat [m];
    QP_space->y_bar = new mfloat [m];
    QP_space->z = new mfloat [n];
    QP_space->z_new = new mfloat [n];
    QP_space->x = new mfloat [n];
    for (int i = 0; i < n; ++i) {
        QP_space->w[i] = 0; QP_space->z[i] = 0; QP_space->x[i] = 0;
    }
    for (int i = 0; i < m; ++i) {
        QP_space->y[i] = 0;
    }
    QP_space->gamma = 1.618;
    QP_space->sigma = 1.0;
    QP_space->update_y_rhs = new mfloat [m];
    QP_space->Qw = new mfloat [n];
    QP_space->Qx = new mfloat [n];
    QP_space->RQ = new mfloat [n];
    QP_space->RC = new mfloat [n];
    spMV_R(QP_space->Q, QP_space->w, n, n, QP_space->Qw);
    QP_space->ATy = new mfloat [n];
    QP_space->Ax = new mfloat [m];
    QP_space->Rd = new mfloat [n];
    QP_space->Rp = new mfloat [m];
    QP_space->z_Qw_c = new mfloat [n];
    QP_space->A_times_z_Qw_c = new mfloat [m];
    QP_space->dz = new mfloat [n];
    QP_space->Qdz = new mfloat [n];
    QP_space->update_w_rhs = new mfloat [n];
    QP_space->z_ATy_c = new mfloat [n];
    QP_space->update_z_unProjected = new mfloat [n];
    QP_space->update_z_projected = new mfloat [n];
    QP_space->iter = 0;

    if (QP_space->input_para.max_ADMM_iter != -1)
        QP_space->maxADMMiter = QP_space->input_para.max_ADMM_iter;
    else
        QP_space->maxADMMiter = 1000;

    QP_space->AT = get_BT(QP_space->A);

    Eigen_init(QP_space);
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

void sGSADMM_QP_update_y_first(QP_struct *QP_space) {
    mfloat *rhs = QP_space->update_y_rhs;
    mfloat *z_Qw_c = QP_space->z_Qw_c;
    int m = QP_space->m, n = QP_space->n;

    spMV_R(QP_space->A, QP_space->x, m, n, QP_space->Ax);
    axpy(-1, QP_space->Ax, QP_space->b, rhs, m);
    ax(1/QP_space->sigma, rhs, rhs, m);

    axpy(-1, QP_space->Qw, QP_space->z, z_Qw_c, n);
    axpy(-1, QP_space->c, z_Qw_c, z_Qw_c, n);
    spMV_R(QP_space->A, z_Qw_c, m, n, QP_space->A_times_z_Qw_c);

    axpy(-1, QP_space->A_times_z_Qw_c, rhs, rhs, m);

    vMemcpy(rhs, m, QP_space->Eigen_rhs_y.data());
    QP_space->Eigen_result_y = QP_space->Eigen_linear_AAT.LLTsolver.solve(QP_space->Eigen_rhs_y);
    vMemcpy(QP_space->Eigen_result_y.data(), m, QP_space->y_bar);

//            PARDISO_solve(QP_space->PDS_A, rhs);
//    vMemcpy(QP_space->PDS_A->x, n, QP_space->y_bar);
}

void sGSADMM_QP_update_y_second(QP_struct *QP_space) {
    mfloat *rhs = QP_space->update_y_rhs;
    mfloat *z_Qw_c = QP_space->z_Qw_c;
    int m = QP_space->m, n = QP_space->n;

    axpy(-1, QP_space->Ax, QP_space->b, rhs, m);
    ax(1/QP_space->sigma, rhs, rhs, m);

    spMV_R(QP_space->Q, QP_space->w, n, n, QP_space->Qw);
    axpy(-1, QP_space->Qw, QP_space->z, z_Qw_c, n);
    axpy(-1, QP_space->c, z_Qw_c, z_Qw_c, n);
    spMV_R(QP_space->A, z_Qw_c, m, n, QP_space->A_times_z_Qw_c);

    axpy(-1, QP_space->A_times_z_Qw_c, rhs, rhs, m);


    vMemcpy(rhs, m, QP_space->Eigen_rhs_y.data());
    QP_space->Eigen_result_y = QP_space->Eigen_linear_AAT.LLTsolver.solve(QP_space->Eigen_rhs_y);
    vMemcpy(QP_space->Eigen_result_y.data(), m, QP_space->y);

//    PARDISO_solve(QP_space->PDS_A, rhs);
//    vMemcpy(QP_space->PDS_A->x, n, QP_space->y);
}

void sGSADMM_QP_update_w_first(QP_struct *QP_space) {
    mfloat *rhs = QP_space->update_w_rhs;
    mfloat *z_ATy_c = QP_space->z_ATy_c;
    int m = QP_space->m, n = QP_space->n;
//    mfloat *rhs2 = new mfloat [n];

    spMV_R(QP_space->AT, QP_space->y_bar, n, m, QP_space->ATy);
    axpy(-1, QP_space->ATy, QP_space->z, z_ATy_c, n);
    axpy(-1, QP_space->c, z_ATy_c, z_ATy_c, n);

    axpy(1/QP_space->sigma, QP_space->x, QP_space->z_ATy_c, rhs, n);


    vMemcpy(rhs, n, QP_space->Eigen_rhs_w.data());
    QP_space->Eigen_result_w = QP_space->Eigen_linear_IQ.LLTsolver.solve(QP_space->Eigen_rhs_w);
    vMemcpy(QP_space->Eigen_result_w.data(), n, QP_space->w_bar);


//    spMV_R(QP_space->Q, rhs, n, n, rhs2);


//    QP_space->Eigen_linear_IQ.LLTsolver.solve(rhs);
//    vMemcpy(QP_space->PDS_IQ->x, n, QP_space->w_bar);
}

void sGSADMM_QP_update_w_second(QP_struct *QP_space) {
    mfloat *rhs = QP_space->update_w_rhs;
    mfloat *z_ATy_c = QP_space->z_ATy_c;
    int m = QP_space->m, n = QP_space->n;

//    spMV_R(QP_space->Q, QP_space->dz, n, n, QP_space->Qdz);
    axpy(1, QP_space->dz, rhs, rhs, n);


    vMemcpy(rhs, n, QP_space->Eigen_rhs_w.data());
    QP_space->Eigen_result_w = QP_space->Eigen_linear_IQ.LLTsolver.solve(QP_space->Eigen_rhs_w);
    vMemcpy(QP_space->Eigen_result_w.data(), n, QP_space->w);

//    PARDISO_solve(QP_space->PDS_IQ, rhs);
//    vMemcpy(QP_space->PDS_IQ->x, n, QP_space->w);
}

void sGSADMM_QP_update_z(QP_struct *QP_space) {
    mfloat *unProjected = QP_space->update_z_unProjected;
    mfloat *projected = QP_space->update_z_projected;
    int n = QP_space->n;

    spMV_R(QP_space->Q, QP_space->w_bar, n, n, QP_space->Qw);
    axpy(-1, QP_space->Qw, QP_space->ATy, unProjected, n);
    axpy(-1, QP_space->c, unProjected, unProjected, n);
    axpy(QP_space->sigma, unProjected, QP_space->x, unProjected, n);

    proj(unProjected, projected, n, QP_space->l, QP_space->u);
    axpy(-1, unProjected, projected, QP_space->z_new, n);
    ax(1/QP_space->sigma, QP_space->z_new, QP_space->z_new, n);

    axpy(-1, QP_space->z, QP_space->z_new, QP_space->dz, n);
    vMemcpy(QP_space->z_new, n, QP_space->z);
}

void sGSADMM_QP_update_x(QP_struct *QP_space) {
    mfloat gamma = QP_space->gamma;
    mfloat sigma = QP_space->sigma;
    int n = QP_space->n;
    int m = QP_space->m;
    spMV_R(QP_space->AT, QP_space->y, n, m, QP_space->ATy);
    axpy(gamma*sigma, QP_space->z_Qw_c, QP_space->x, QP_space->x, n);
    axpy(gamma*sigma, QP_space->ATy, QP_space->x, QP_space->x, n);
}

void sGSADMM_QP_compute_status(QP_struct *QP_space) {
    int m = QP_space->m, n = QP_space->n;
    axpy(-1, QP_space->Ax, QP_space->b, QP_space->Rp, m);
    QP_space->inf_p = norm2(QP_space->Rp, m)/(1 + norm2(QP_space->b, m));

    axpy(-1, QP_space->Qw, QP_space->z, QP_space->Rd, n);
    axpy(-1, QP_space->c, QP_space->Rd, QP_space->Rd, n);
    spMV_R(QP_space->AT, QP_space->y, n, m, QP_space->ATy);
    axpy(1, QP_space->ATy, QP_space->Rd, QP_space->Rd, n);
    QP_space->inf_d = norm2(QP_space->Rd, n)/(1 + norm2(QP_space->c, n));

    spMV_R(QP_space->Q, QP_space->x, n, n, QP_space->Qx);
    axpy(-1, QP_space->Qw, QP_space->Qx, QP_space->RQ, n);
    QP_space->inf_Q = norm2(QP_space->RQ, n)/(1 + norm2(QP_space->Qx, n) + norm2(QP_space->Qw, n));

    axpy(-1, QP_space->z, QP_space->x, QP_space->dz, n);
    proj(QP_space->dz, QP_space->update_z_projected, n, QP_space->l, QP_space->u);
    axpy(-1, QP_space->update_z_projected, QP_space->x, QP_space->RC, n);
    QP_space->inf_C = norm2(QP_space->RC, n)/(1 + norm2(QP_space->x, n) + norm2(QP_space->z, n));

    QP_space->primal_obj = xTy(QP_space->x, QP_space->Qx, n)/2 + xTy(QP_space->c, QP_space->x, n);
    QP_space->dual_obj = -support_function(QP_space->z, QP_space->l, QP_space->u, n, &QP_space->infty_z);
    QP_space->dual_obj += -xTy(QP_space->w, QP_space->Qw, n)/2 + xTy(QP_space->b, QP_space->y, m);
    QP_space->inf_g = (QP_space->primal_obj - QP_space->dual_obj)/(1+abs(QP_space->primal_obj)+abs(QP_space->dual_obj));
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
    mfloat inv_sigma_old = 1/QP_space->sigma_old;
    mfloat inv_sigma = 1/QP_space->sigma;
    QP_space->EigenIQ -= inv_sigma_old*QP_space->EigenI;
    QP_space->EigenIQ += inv_sigma*QP_space->EigenI;
//    for (int i = 0; i < QP_space->n; ++i) {
//        QP_space->PDS_IQ->a[QP_space->PDS_IQ->ia[i]-1] += -inv_sigma_old + inv_sigma;

//    }
    QP_space->Eigen_linear_IQ.LLTsolver.factorize(QP_space->EigenIQ);
//    PARDISO_numerical_fact(QP_space->PDS_IQ);
}

void sGSADMM_QP_print_status(QP_struct* QP_space) {
    if (QP_space->iter == 0) {
        printf("iter    inf_p      inf_d       inf_Q       inf_C       obj_p        obj_d       inf_g       sigma\n");
    }
    if (QP_space->iter % 20 == 0) {
        printf("%4d    %3.2e   %3.2e    %3.2e    %3.2e    %3.2e    %3.2e    %3.2e    %3.2e\n", QP_space->iter, QP_space->inf_p, QP_space->inf_d, QP_space->inf_Q, QP_space->inf_C, QP_space->primal_obj, QP_space->dual_obj, QP_space->inf_g, QP_space->sigma);
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
        int iNUM = n*d;
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

void SSN_QP(QP_struct* QP_space) {




}

void BiCG_prod(const mfloat *x, mfloat* output, BiCG_struct *BiCG_space, QP_struct *QP_space) {
//    spMV_R(BiCG_space->A, x, BiCG_space->A.nRow, BiCG_space->A.nCol, output);
    int n = BiCG_space->dim_n, m = BiCG_space->dim_m;
    mfloat *output1 = output, *output2 = output + n;
    const mfloat *x1 = x;
    const mfloat *x2 = x + n;
    spVec_int U_idx = BiCG_space->U_idx;
    mfloat *temp1 = BiCG_space->BiCG_prod_temp1;
    mfloat *temp2 = BiCG_space->BiCG_prod_temp2;
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

    partial_spMV_R(Q, x1, U_idx.number, n, temp1, U_idx);
    axpy(1+tau/sigma, x1, temp1, temp1, n);
    partial_spMV_R(AT, x2, U_idx.number, m, output1, U_idx);
    axpy(-1, output1, temp1, output1, n);

    partial_spMV_R(Q, x1, U_idx.number, n, temp2, U_idx);
//    axpy(1+tau/sigma, x1, temp1, temp1, n);
    partial_spMV_2(AT, temp2, n, m, output2, U_idx);

    partial_spMV_R(AT, x2, U_idx.number, m, temp2, U_idx);
    partial_spMV_2(AT, temp2, n, m, temp1, U_idx);
    axpy(tau/sigma, x2, temp1, temp1, m);
    axpy(-1, output2, temp1, output2, m);
}

void BiCG_init(BiCG_struct* CG_space) {
    int n = CG_space->dim_n + CG_space->dim_m;
    CG_space->x = new mfloat [n];
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
}

void precond_BiCG(const mfloat* input, mfloat* output, BiCG_struct* CG_space) {
    vMemcpy(input, CG_space->dim_n+CG_space->dim_m, output);
}

void BiCG(const mfloat* x0, const mfloat* rhs, mfloat* output, BiCG_struct* CG_space, QP_struct *QP_space) {
    int n = CG_space->dim_n + CG_space->dim_m;
    mfloat tol = CG_space->tol;
    double *x = CG_space->x, *r = CG_space->r;
    mfloat *r0 = CG_space->r0, *p_hat = CG_space->p_hat, *s = CG_space->s;
    mfloat *s_hat = CG_space->s_hat, *t = CG_space->t;
    double *p = CG_space->p, *Ap = CG_space->Ap, *v = CG_space->v;
    mfloat rho_old = 1, alpha = 1, omega = 1;
    mfloat bnorm = norm2(rhs, n)+1;
//    for (int i = 0; i < n; ++i) {
//        v[i] = 0;
//        p[i] = 0;
//    }
    BiCG_prod(x0, Ap, CG_space, QP_space);
    cout << Ap[0] << ' ' << Ap[1] << ' ' << Ap[2] << ' ' << Ap[3] << endl;
    axpy(-1, Ap, rhs, r, n);
    vMemcpy(r, n, r0);

    int iter = 1;
    mfloat rho = 1;
    mfloat beta = 1;
    mfloat res = norm2(r0, n);
    if (res/bnorm < tol) {
//        printf("      %3d", 0);
        cout << "BiCG input is okay" << endl;
        vMemcpy(x, n, output);
        return;
    }
    while (iter < CG_space->max_iter) {
        rho = xTy(r, r0, n);
        if (rho < 1e-10) {
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

        precond_BiCG(p, p_hat, CG_space);

        BiCG_prod(p_hat, v, CG_space, QP_space);
        alpha = rho/ xTy(r0, v, n);
//        axpy(alpha, p_pcg, x_pcg, h, n);

        axpy(-alpha, v, r, s, n);
        mfloat norms = norm2(s, n);
        if (norms < tol) {
            axpy(alpha, p_hat, x, x, n);
            res = norms/bnorm;
            vMemcpy(x, n, output);
            return;
        }

        precond_BiCG(s, s_hat, CG_space);
        BiCG_prod(s_hat, t, CG_space, QP_space);
        omega = xTy(t, s, n)/xTy(t, t, n);
        axpy(alpha, p_hat, x, x, n);
        axpy(omega, s_hat, x, x, n);

        axpy(-omega, t, s, r, n);
        CG_space->error = norm2(r, n)/bnorm;
//        printf("iter %d err %f\n", iter, CG_space->error);
        if (CG_space->error < tol) {
            vMemcpy(x, n, output);
            return;
        }

        if (omega < 1e-10) {
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


void update_sigma(QP_struct *QP_space) {
//    if (primfeas < dualfeas)
//        primWin++;
//    else
//        dualWin++;

    int iter = QP_space->iter;
    int sigma_update_iter = 2;
    if (iter < 10)
        sigma_update_iter = 2;
    else if (iter < 20)
        sigma_update_iter = 3;
    else if (iter < 200)
        sigma_update_iter = 3;
    else if (iter < 500)
        sigma_update_iter = 10;

//    mfloat factor = 5.0;
    mfloat inf_primal = max(QP_space->inf_p, max(QP_space->inf_Q, QP_space->inf_C));
    QP_space->sigma_old = QP_space->sigma;
    if (iter % sigma_update_iter == 0) {
        if (inf_primal < 0.75*QP_space->inf_d) {
            QP_space->sigma *= 1.25;
        }
        else if (inf_primal > 1.33*QP_space->inf_d) {
            QP_space->sigma *= 0.8;
        }

        QP_space->sigma = min(QP_space->sigma, 1e5);
        QP_space->sigma = max(QP_space->sigma, 1e-10);
    }
    if (abs(QP_space->sigma_old - QP_space->sigma) > 1e-11) {
        sGSADMM_QP_refact_IQ(QP_space);
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
    Eigen::SparseMatrix<mfloat> EigenA = get_eigen_spMat(QP_space->A);
    Eigen::SparseMatrix<mfloat> EigenAT = get_eigen_spMat(QP_space->AT);
    Eigen::SparseMatrix<mfloat> EigenAAT = EigenA*EigenA.transpose();

    QP_space->Eigen_rhs_w.resize(QP_space->n, 1);
    QP_space->Eigen_result_w.resize(QP_space->n, 1);
    QP_space->Eigen_rhs_y.resize(QP_space->m, 1);
    QP_space->Eigen_result_y.resize(QP_space->m, 1);
//    cout << EigenA << endl;
//    cout << EigenAAT << endl;
    QP_space->EigenQ = get_eigen_spMat(QP_space->Q);
    QP_space->EigenI.resize(QP_space->n, QP_space->n);
    QP_space->EigenI.setIdentity();
    QP_space->EigenIQ = 1/QP_space->sigma*QP_space->EigenI + QP_space->EigenQ;

    QP_space->Eigen_linear_AAT.LLTsolver.analyzePattern(EigenAAT);
    QP_space->Eigen_linear_AAT.LLTsolver.factorize(EigenAAT);
    QP_space->Eigen_linear_IQ.LLTsolver.analyzePattern(QP_space->EigenIQ);
    QP_space->Eigen_linear_IQ.LLTsolver.factorize(QP_space->EigenIQ);
}

int PARDISO_init(PARDISO_var *PDS, sparseRowMatrix A) {
    PDS->n = A.nRow;
    if (A.nRow != A.nCol) {
        cout << "not square matrix" << endl;
        return 0;
    }
    PDS->debug = false;
    PDS->ia = A.rowStart;
    PDS->ja = A.column;
    int nnz = A.rowStart[A.nRow];
    PDS->a = A.value;
    PDS->mtype = -2;        /* Real symmetric matrix */

    /* RHS and solution vectors. */
    PDS->x = (mfloat *) malloc(sizeof(mfloat) * PDS->n);

    PDS->nrhs = 1;          /* Number of right hand sides. */

/* -------------------------------------------------------------------- */
/* ..  Setup Pardiso control parameters.                                */
/* -------------------------------------------------------------------- */

    PDS->error = 0;
    PDS->solver = 0; /* use sparse direct solver */
//    pardisoinit (PDS->pt,  &(PDS->mtype), &(PDS->solver), PDS->iparm, PDS->dparm, &(PDS->error));

    if (PDS->error != 0)
    {
        if (PDS->error == -10 )
            printf("No license file found \n");
        if (PDS->error == -11 )
            printf("License is expired \n");
        if (PDS->error == -12 )
            printf("Wrong username or hostname \n");
        return 1;
    }
    else
        printf("[PARDISO]: License check was successful ... \n");

    /* Numbers of processors, value of OMP_NUM_THREADS */
//    char *var = getenv("OMP_NUM_THREADS");
//    if(var != NULL)
//        sscanf( var, "%d", &num_procs );
//    else {
//        printf("Set environment OMP_NUM_THREADS to 1");
//        num_procs = 1;
////        exit(1);
//    }
    PDS->iparm[2]  = 1;

    PDS->maxfct = 1;		/* Maximum number of numerical factorizations.  */
    PDS->mnum   = 1;         /* Which factorization to use. */

    PDS->msglvl = 0;         /* Print statistical information  */
    PDS->error  = 0;         /* Initialize error flag */

/* -------------------------------------------------------------------- */
/* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
/*     notation.                                                        */
/* -------------------------------------------------------------------- */
    for (int i = 0; i < A.nRow + 1; ++i)
        (PDS->ia[i])++;
    for (int i = 0; i < nnz; ++i)
        (PDS->ja[i])++;

/* -------------------------------------------------------------------- */
/*  .. pardiso_chk_matrix(...)                                          */
/*     Checks the consistency of the given matrix.                      */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */

    if (PDS->debug) {
//        pardiso_chkmatrix  (&(PDS->mtype), &(PDS->n), PDS->a, PDS->ia, PDS->ja, &(PDS->error));
        if (PDS->error != 0) {
            printf("\nERROR in consistency of matrix: %d", PDS->error);
            exit(1);
        }
    }


/* -------------------------------------------------------------------- */
/* ..  pardiso_chkvec(...)                                              */
/*     Checks the given vectors for infinite and NaN values             */
/*     Input parameters (see PARDISO user manual for a description):    */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */

    if (PDS->debug) {
//        pardiso_chkvec(&(PDS->n), &(PDS->nrhs), PDS->b, &(PDS->error));
        if (PDS->error != 0) {
            printf("\nERROR  in right hand side: %d", PDS->error);
            exit(1);
        }
    }
/* -------------------------------------------------------------------- */
/* .. pardiso_printstats(...)                                           */
/*    prints information on the matrix to STDOUT.                       */
/*    Use this functionality only for debugging purposes                */
/* -------------------------------------------------------------------- */

    if (PDS->debug) {
//        pardiso_printstats(&(PDS->mtype), &(PDS->n), PDS->a, PDS->ia, PDS->ja, &(PDS->nrhs), PDS->b, &(PDS->error));
        if (PDS->error != 0) {
            printf("\nERROR right hand side: %d", PDS->error);
            exit(1);
        }
    }
//    -------------------------------------------------------------------- */
/* ..  Reordering and Symbolic Factorization.  This step also allocates */
/*     all memory that is necessary for the factorization.              */
/* -------------------------------------------------------------------- */
    PDS->phase = 11;

//    pardiso (PDS->pt, &(PDS->maxfct), &(PDS->mnum), &(PDS->mtype), &(PDS->phase),
//             &(PDS->n), PDS->a, PDS->ia, PDS->ja, &(PDS->idum), &(PDS->nrhs),
//             PDS->iparm, &(PDS->msglvl), &(PDS->ddum), &(PDS->ddum), &(PDS->error), PDS->dparm);

    if (PDS->error != 0) {
        printf("\nERROR during symbolic factorization: %d", PDS->error);
        exit(1);
    }
    printf("\nReordering completed ... ");
    printf("\nNumber of nonzeros in factors  = %d", PDS->iparm[17]);
    printf("\nNumber of factorization MFLOPS = %d", PDS->iparm[18]);

    return 0;
}

int PARDISO_numerical_fact(PARDISO_var *PDS) {
    /* -------------------------------------------------------------------- */
    /* ..  Numerical factorization.                                         */
    /* -------------------------------------------------------------------- */
    PDS->phase = 22;
    PDS->iparm[32] = 0; /* compute determinant */

//    pardiso (PDS->pt, &(PDS->maxfct), &(PDS->mnum), &(PDS->mtype), &(PDS->phase),
//             &(PDS->n), PDS->a, PDS->ia, PDS->ja, &(PDS->idum), &(PDS->nrhs),
//             PDS->iparm, &(PDS->msglvl), &(PDS->ddum), &(PDS->ddum), &(PDS->error), PDS->dparm);

    if (PDS->error != 0) {
        printf("\nERROR during numerical factorization: %d", PDS->error);
        exit(2);
    }
    printf("\nFactorization completed ...\n ");

    PDS->isInitialized = true;
    return 0;
}

int PARDISO_solve(PARDISO_var *PDS, mfloat *rhs) {
    /* ---------------------------------numerical_fact_----------------------------------- */
    /* ..  Back substitution and iterative refinement.                      */
    /* -------------------------------------------------------------------- */
    PDS->phase = 33;

    PDS->iparm[7] = 0;       /* Max numbers of iterative refinement steps. */

    if (!(PDS->isInitialized)) {
        cout << "PARDISO not initialized" << endl;
        return 0;
    }
    PDS->b = new mfloat [PDS->n];
    vMemcpy(rhs, PDS->n, PDS->b);

//    pardiso (PDS->pt, &(PDS->maxfct), &(PDS->mnum), &(PDS->mtype), &(PDS->phase),
//             &(PDS->n), PDS->a, PDS->ia, PDS->ja, &(PDS->idum), &(PDS->nrhs),
//             PDS->iparm, &(PDS->msglvl), PDS->b, PDS->x, &(PDS->error), PDS->dparm);

    if (PDS->error != 0) {
        printf("\nERROR during solution: %d", PDS->error);
        exit(3);
    }

//    printf("\nSolve completed ... ");
//    printf("\nThe solution of the system is: ");
//    for (int i = 0; i < 8; i++) {
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

    printf("\nRelease completed ... ");

    return 0;
}