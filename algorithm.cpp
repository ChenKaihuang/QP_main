//
// Created by chenkaihuang on 6/7/22.
//

#include <string.h>
#include "algorithm.h"

//#define USE_CBLAS 1

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

void get_spR(const int* NodeArcMatrix, spR* BT, int num) {
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


spR get_BT(int num, int n, spR BT) {
    spR Bf;
    int* rowIndex = new int [2*num];
    int nnz = 2*num;
    Bf.value = new mfloat [nnz];
    Bf.rowStart = new int [n+2];
//    BT->rowLength = new int [m];
    Bf.column = new int [nnz];
    mfloat* newValue = Bf.value;
    int *rowStart = Bf.rowStart;
//    int *rowLength = BT->rowLength;
    int *column = Bf.column;
    mfloat* oldValue = BT.value;
    int *colStart = BT.rowStart;
    int *row = BT.column;
    for (int i = 0; i < n+2; ++i) {
        rowStart[i] = 0;
    }
    // count per row
    for (int i = 0; i < nnz; ++i) {
        ++rowStart[row[i]+2];
    }

    // generate rowStart
    for (int i = 2; i < n+2; ++i) {
        rowStart[i] += rowStart[i-1];
    }

    // main part
    for (int i = 0; i < num; ++i) {
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

void spMV_R(spR A, const mfloat* x, int m, int n, mfloat* Ax) {
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

void partial_spMV_2(spR A, const mfloat* x, int m, int n, mfloat* Ax, spVec_int nzidx) {
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

    Eigen_solve(r_pcg, z_pcg, n, d);
    vMemcpy(z_pcg, n*d, p_pcg);

    // GPU
    std::chrono::steady_clock::time_point time0_mycg = time_now();
    My_CG();
    mycgTime += time_since(time0_mycg);

    // GPU
    mfloat s = xTy(p_pcg, Ap_pcg, n*d);
    mfloat rsold = xTy(r_pcg, z_pcg, n*d);
    mfloat t = rsold/s;

    axpy(t, p_pcg, x_pcg, x_pcg, n*d);
    for (int k = 1; k < par.maxit + 1; ++k) {
        axpy(-t, Ap_pcg, r_pcg, r_pcg, n*d);
        res_temp = norm2(r_pcg, n*d);
//        std::cout << k << ' ' << res_temp << std::endl;
        if (res_temp < tol) {
            cgNum += k;
            printf("      %3d", k);
//            std::cout << "CG: quit at iteration " << k << std::endl;
            vMemcpy(x_pcg, n*d, output);
            return;
//            return x_pcg;
        }
        precondfun(r_pcg, z_pcg, n, d);
        mfloat rsnew = xTy(r_pcg, z_pcg, n*d);
        mfloat B = rsnew/rsold;
        int iNUM = n*d;
        axpy(B, p_pcg, z_pcg, p_pcg, n*d);
        rsold = rsnew;

        std::chrono::steady_clock::time_point time0_mycg = time_now();
        My_CG();
        mycgTime += time_since(time0_mycg);

        s = xTy(p_pcg, Ap_pcg, n*d);
        t = rsold/s;
        axpy(t, p_pcg, x_pcg, x_pcg, n*d);
    }
    std::cout << "\nCG iteration at maximum " << CG_space->max_iter << " res " << res_temp << std::endl;
    vMemcpy(x_pcg, n*d, output);
//    return x_pcg;
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

void partial_spMV_R(spR A, const mfloat* x, int m, int n, mfloat* Ax, spVec_int nzidx) {
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

spR B_sub_Row(spVec_int nzidx, spR BT) {
    spR Bf;
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
            assert(i < Ainput.BT.nRow);
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

bool collect_infeasibility() {

}
void create_SSNCG() {
}

void SSNCG_new() {
}

void create_find_step() {
}

void find_step(int stepopt) {
}

mfloat obj_val_SSN() {
}

mfloat primal_obj() {
}

mfloat dual_obj() {
}

bool check_termination_SSN(int iter) {
}


void update_sigma(mfloat primfeas, mfloat dualfeas, int iter) {
    if (primfeas < dualfeas)
        primWin++;
    else
        dualWin++;

    int sigma_update_iter = 2;
    if (iter < 10)
        sigma_update_iter = 2;
    else if (iter < 20)
        sigma_update_iter = 3;
    else if (iter < 200)
        sigma_update_iter = 3;
    else if (iter < 500)
        sigma_update_iter = 10;

    mfloat factor = 5.0;
    if (iter % sigma_update_iter == 0) {
        if (primWin > std::max(1.0, 1.2*dualWin)) {
            primWin = 0;
            par.sigma = std::max(1e-4, par.sigma/factor);
        } else if (dualWin > std::max(1.0, 1.2*primWin)) {
            dualWin = 0;
            par.sigma = std::min(1e5, par.sigma*factor);
        }
    }
}

void print_spM(spR input) {
    for (int i = 0; i < input.nRow; ++i) {
        for (int j = input.rowStart[i]; j < input.rowStart[i+1]; ++j) {
            std::cout << i << ' ' << input.column[j] << ' ' << input.value[j] << std::endl;
        }
    }
}

Eigen::SparseMatrix<mfloat> get_eigen_spMat(spR BT) {
    typedef Eigen::Triplet<mfloat> T;
    std::vector<T> tripletList;
    tripletList.reserve(BT.nRow*2);
    int *colStart = BT.rowStart;
    mfloat *value = BT.value;
    int *row = BT.column;
    for (int i = 0; i < BT.nRow; ++i) {
        for (int j = colStart[i]; j < colStart[i+1]; ++j)
            tripletList.emplace_back(row[j],i,value[j]);
    }
    Eigen::SparseMatrix<mfloat> mat(BT.nCol, BT.nRow);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
// mat is ready to go!

}

void precond_CG(const mfloat *input, mfloat *output, int m, int n) {
    if (LLT.isEmpty) {
        vMemcpy(input, m*n, output);
    } else {
#pragma omp parallel for num_threads(NUM_THREADS_OpenMP) default(none) shared(n,m,input,LLT,output, temp_pre, temp2_pre)
        for (int i = 0; i < n; ++i) {
            vMemcpy(input+i*m, m, temp_pre[i].data());
            temp2_pre[i] = LLT.LLTsolver.solve(temp_pre[i]);
            vMemcpy(temp2_pre[i].data(), m, output+i*m);
        }
    }
}

Eigen::VectorX<mfloat> *temp_pre, *temp2_pre;
Eigen::SparseMatrix<mfloat> In, wInum, wInum2, Ei_B, Ei_BT, Ei_A;
linearSolver LLT;
Eigen::VectorX<mfloat> tempW;
void initial_Eigen() {
    In.resize(Ainput.B.nRow, Ainput.B.nRow);
    In.setIdentity();
    wInum.resize(Ainput.B.nCol, Ainput.B.nCol);
    wInum.setIdentity();
    wInum2.resize(Ainput.B.nRow, Ainput.B.nRow);
    wInum2.setIdentity();
    Ei_B = get_eigen_spMat(Ainput.BT);
    Ei_BT = Ei_B.transpose();
    LLT.isEmpty = false;
    tempW.resize(Ainput.B.nCol, 1);
    temp_pre = new Eigen::VectorX<mfloat> [Ainput.dim_d];
    temp2_pre = new Eigen::VectorX<mfloat> [Ainput.dim_d];
    for (int i = 0; i < Ainput.dim_d; ++i) {
        temp_pre[i].resize(Ainput.B.nRow, 1);
        temp2_pre[i].resize(Ainput.B.nRow, 1);
    }
}

void create_ADMM() {
}

void ADMM() {
}

void update_sigma_admm(){
}


void writeCOO() {
    using namespace std;
#ifdef NUM_THREADS_OpenMP
    omp_set_num_threads(NUM_THREADS_OpenMP);
    printf("PARALLEL Switch On: %d threads\n", NUM_THREADS_OpenMP);
#endif
    int d = 10, n = 30000, k_n = 10;
    mfloat phi = 0.5;
    auto* data = new mfloat [d*n];
    read_bin("../mnist_1000000_10.bin", data, d*n);

    auto* weightVec = new mfloat [k_n*n];
    auto *NodeArcMatrix = new int [k_n*n*2];

    int num = compute_weight(data, k_n, d, n, phi, 1, NodeArcMatrix);

    spR BT;
    get_spR(NodeArcMatrix, &BT, num);
    BT.nRow = num;
    BT.nCol = n;
    spR B;
    B = get_BT(num, n, BT);

    auto* rIdx = new int [2*num];
    for (int i = 0; i < B.nRow; ++i) {
        for (int j = B.rowStart[i]; j < B.rowStart[i+1]; ++j) {
//            if (i == 0)
//                std::cout << j << std::endl;
            rIdx[j] = i;
        }

    }

    for (int i = 0; i < 20; ++i)
        std::cout << B.column[i] << std::endl;
    ofstream out("../data/mnist_1000000_rIdx.bin", ios::out | ios::binary);
    if(!out) {
        cout << "Cannot open file.";
        return;
    }
    out.write((char *) rIdx, sizeof(int)*2*num);

    std::cout << num << std::endl;

    out.close();

    ofstream out2("../data/mnist_1000000_cIdx.bin", ios::out | ios::binary);
    if(!out2) {
        cout << "Cannot open file.";
        return;
    }

    out2.write((char *) B.column, sizeof(int)*num*2);

    out2.close();

    ofstream out3("../data/mnist_1000000_value.bin", ios::out | ios::binary);
    if(!out3) {
        cout << "Cannot open file.";
        return;
    }

    out3.write((char *) B.value, sizeof(mfloat)*num*2);

    out3.close();


}