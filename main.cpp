#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "algorithm.h"

using namespace std;

/* PARDISO prototype. */
//extern "C" void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
//extern "C" void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
//                             double *, int    *,    int *, int *,   int *, int *,
//                             int *, double *, double *, int *, double *);
//extern "C" void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
//extern "C" void pardiso_chkvec     (int *, int *, double *, int *);
//extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *,
//                                    double *, int *);


void run_bin(const char* file_name) {
#ifdef NUM_THREADS_OpenMP
    omp_set_num_threads(NUM_THREADS_OpenMP);
    printf("PARALLEL Switch On: %d threads\n", NUM_THREADS_OpenMP);
#endif
    mfloat *size = new mfloat [4];
    read_bin(file_name, size, 4);

    int m = static_cast<int>(size[0]);
    int n = static_cast<int>(size[1]);
    int nnz_Q = static_cast<int>(size[2]);
    int nnz_A = static_cast<int>(size[3]);

    cout << "m = " << m << " n = " << n << endl;
    cout << "nnz_Q = " << nnz_Q << " nnz_A = " << nnz_A << endl;

    int total = 4+nnz_Q*2+n+1+nnz_A*2+m+1+3*n+m;
    mfloat *read = new mfloat [total];
    read_bin(file_name, read, total);

    mfloat *data = read + 4;

    mfloat *Q_value = new mfloat [nnz_Q];
    mfloat *Q_rowStart_d = new mfloat [n+1];
    mfloat *Q_column_d = new mfloat [nnz_Q];
    mfloat *A_value = new mfloat [nnz_A];
    mfloat *A_rowStart_d = new mfloat [m+1];
    mfloat *A_column_d = new mfloat [nnz_A];
    mfloat *b = new mfloat [m];
    mfloat *c = new mfloat [n];
    mfloat *L = new mfloat [n];
    mfloat *U = new mfloat [n];

    vMemcpy(data, nnz_Q, Q_value);
    vMemcpy(data+nnz_Q, n+1, Q_rowStart_d);
    vMemcpy(data+nnz_Q+n+1, nnz_Q, Q_column_d);
    vMemcpy(data+nnz_Q+n+1+nnz_Q, nnz_A, A_value);
    vMemcpy(data+nnz_Q+n+1+nnz_Q+nnz_A, m+1, A_rowStart_d);
    vMemcpy(data+nnz_Q+n+1+nnz_Q+nnz_A+m+1, nnz_A, A_column_d);
    vMemcpy(data+nnz_Q+n+1+nnz_Q+nnz_A+m+1+nnz_A, m, b);
    vMemcpy(data+nnz_Q+n+1+nnz_Q+nnz_A+m+1+nnz_A+m, n, c);
    vMemcpy(data+nnz_Q+n+1+nnz_Q+nnz_A+m+1+nnz_A+m+n, n, L);
    vMemcpy(data+nnz_Q+n+1+nnz_Q+nnz_A+m+1+nnz_A+m+n+n, n, U);

    int *Q_rowStart = new int [n+1];
    int *Q_column = new int [nnz_Q];
    int *A_rowStart = new int [m+1];
    int *A_column = new int [nnz_A];
    for (int i = 0; i < n+1; i++) {
        Q_rowStart[i] = static_cast<int>(Q_rowStart_d[i]);
    }
    for (int i = 0; i < nnz_Q; i++) {
        Q_column[i] = static_cast<int>(Q_column_d[i]);
    }
    for (int i = 0; i < m+1; i++) {
        A_rowStart[i] = static_cast<int>(A_rowStart_d[i]);
    }
    for (int i = 0; i < nnz_A; i++) {
        A_column[i] = static_cast<int>(A_column_d[i]);
    }

    input_parameters *ip = new input_parameters;
    ip->max_ADMM_iter = 1000;
    ip->max_ALM_iter = 100;
    ip->sigma = 1;
    ip->kappa = 1e-4;
    ip->gamma = 1.95;
    ip->use_scale = 1;
    ip->use_mkl = 1;

    sparseRowMatrix Q,A;
    Q.nRow = n;
    Q.nCol = n;
    Q.value = Q_value;
    Q.rowStart = Q_rowStart;
    Q.column = Q_column;
    A.nRow = m;
    A.nCol = n;
    A.value = A_value;
    A.rowStart = A_rowStart;
    A.column = A_column;

    output_parameters *op = new output_parameters;
    QP_solve(Q, A, b, c, L, U, m, n, *ip, op);

}

int main( void )
{
    const char* filename = "../python_test/QPDATA-presolved/bin/EXDATA.bin";
//    test_MKL_pardiso();
//    test_eigen_segment_set();
//    QP_struct *QP_space = new QP_struct;
//    sparseRowMatrix Q;
//    Q.value = new mfloat [4];
//    Q.nRow = 2; Q.nCol = 2;
//    Q.value[0] = 4; Q.value[1] = 2; Q.value[2] = 2; Q.value[3] = 6;
//    Q.rowStart = new int [3];
//    Q.rowStart[0] = 0; Q.rowStart[1] = 2; Q.rowStart[2] = 4;
//    Q.column = new int [4];
//    Q.column[0] = 0; Q.column[1] = 1; Q.column[2] = 0; Q.column[3] = 1;
//
//    auto eigen_Q = get_eigen_spMat(Q);
//    cout << eigen_Q << endl;
//    eigen_Q.diagonal().array() += 1;
//    cout << eigen_Q << endl;
    run_bin(filename);
//     test_concat_4_spMat();
//    test_eigen_sub();
//    simple_QP_test();
//    simple_ruiz_test();
//    BiCG_test();
//    putenv("OMP_NUM_THREADS=1");
//    /* Matrix data. */
//    int    n = 8;
//    int    ia[ 9] = { 0, 4, 7, 9, 11, 14, 16, 17, 18 };
//    int    ja[18] = { 0,    2,       5, 6,
//                      1, 2,    4,
//                      2,             7,
//                      3,       6,
//                      4, 5, 6,
//                      5,    7,
//                      6,
//                      7 };
//    double  a[18] = { 7.0,      1.0,           2.0, 7.0,
//                      -4.0, 8.0,           2.0,
//                      1.0,                     5.0,
//                      7.0,           9.0,
//                      5.0, -1.0, 5.0,
//                      0.0,      5.0,
//                      11.0,
//                      5.0 };
//    double b[8];
//    sparseRowMatrix A;
//    A.nRow = 8; A.nCol = 8;
//    A.rowStart = ia; A.column = ja; A.value = a;
//    auto *PDS = new PARDISO_var;
//    for (int i = 0; i < 8; ++i) b[i] = 1;
//    PARDISO_init(PDS, A);
//
//    PARDISO_numerical_fact(PDS);
//
//    PARDISO_solve(PDS, b);
//
//    PARDISO_release(PDS);
    return 0;
}