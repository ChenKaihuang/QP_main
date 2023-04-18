//
// Created by chenkaihuang on 6/7/22.
//


#ifndef C_CODE_ALGORITHM_H
#define C_CODE_ALGORITHM_H


#include <math.h>
#include <string.h>
#include "utils.h"
#include <fstream>
#include <iostream>
#include <assert.h>
#include <time.h>
//#include <cblas.h>
#include <omp.h>
//#include "mkl.h"
//#include "mkl_solvers_ee.h"
#include <Eigen/Sparse>

#define mfloat double
#define NUM_THREADS_OpenMP 8

//extern "C" void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
//extern "C" void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
//                             double *, int    *,    int *, int *,   int *, int *,
//                             int *, double *, double *, int *, double *);
////extern "C" void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
//extern "C" void pardiso_chkvec     (int *, int *, double *, int *);
//extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *,
//                                    double *, int *);

typedef struct {
    bool isEmpty;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<mfloat>> LLTsolver;
} linearSolver;

typedef struct {
    int nRow;
    int nCol;
    mfloat* vec;
} Mat;

typedef struct {
    int number;
    int* vec;
} spVec_int;


typedef struct {
    int nRow;
    int nCol;
    mfloat* value;
    int* rowStart;
    int* rowLength;
    int* column;
} sparseRowMatrix;

typedef struct BiCG_struct{
    mfloat *x,*r,*p,*Ap, *v, *r0, *p_hat, *s, *s_hat, *t;
    int dim_n, max_iter = 100, nCGs;
    mfloat tol = 1e-6, cgTime, error;
    linearSolver LLT;
    sparseRowMatrix A;
} BiCG_struct;

typedef struct {
    mfloat *x,*r,*z,*p,*Ap,*Atp;
    int dim_n, dim_d, max_iter, nCGs;
    mfloat tol, cgTime;
    linearSolver LLT;
} CG_struct;

typedef struct {
    /* Internal solver memory pointer pt,                  */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
    /* or void *pt[64] should be OK on both architectures  */
    void *pt[64];
    bool debug, isInitialized;
    int maxfct, mnum, mtype, phase, n, idum, nrhs, msglvl;
    /* Pardiso control parameters: dparm iparm */
    mfloat *a, *b, *x;
    double dparm[64], ddum;
    int *ia, *ja, iparm[64], error, solver;
} PARDISO_var;

typedef struct input_parameters {
    int max_ADMM_iter = -1;
} input_parameters;

typedef struct {
    // input: 1/2 xQx + cx, s.t. Ax = b x \in C = [l, u]
    // quadratic matrix Q and constraint matrix A;
    sparseRowMatrix Q, A;
    // vector c
    mfloat *c, *b, *l, *u;
    int m, n;

    // work variables
    sparseRowMatrix AT, AAT, IQ;
    mfloat *w, *y, *z, *x, sigma, gamma, sigma_old;
    PARDISO_var *PDS_IQ, *PDS_A;
    mfloat *update_y_rhs, *Qw, *ATy, *Ax, *Rd, *Rp, *z_Qw_c, *A_times_z_Qw_c, *dz, *Qx, *Qdz;
    mfloat *z_new;
    mfloat *w_bar, *y_bar;
    mfloat *update_w_rhs, *z_ATy_c;
    mfloat *update_z_unProjected, *update_z_projected;
    mfloat inf_p, inf_d, inf_Q, inf_C, inf_g, primal_obj, dual_obj, infty_z;
    mfloat *RQ, *RC;
    input_parameters input_para;
    Eigen::SparseMatrix<mfloat> EigenI, EigenQ, EigenIQ;
    Eigen::SparseMatrix<mfloat> upperMatQ;
    linearSolver Eigen_linear_AAT, Eigen_linear_IQ;
    int iter, maxADMMiter;
    Eigen::VectorX<mfloat> Eigen_rhs_w, Eigen_rhs_y, Eigen_result_w, Eigen_result_y;
} QP_struct;

void Eigen_init(QP_struct *QP_space);

void sGSADMM_QP(sparseRowMatrix Q, sparseRowMatrix A, mfloat* b, mfloat* c, mfloat* l, mfloat* u, int m, int n, input_parameters para);

void sGSADMM_QP_init(QP_struct *QP_space);

void simple_QP_test();

void BiCG_test();

void sGSADMM_QP_update_y_first(QP_struct *QP_space);
void sGSADMM_QP_update_y_second(QP_struct *QP_space);

void sGSADMM_QP_update_w_first(QP_struct *QP_space);
void sGSADMM_QP_update_w_second(QP_struct *QP_space);

void sGSADMM_QP_update_z(QP_struct *QP_space);

void sGSADMM_QP_update_x(QP_struct *QP_space);

void sGSADMM_QP_compute_status(QP_struct *QP_space);

void sGSADMM_QP_print_status(QP_struct* QP_space);

void sGSADMM_QP_refact_IQ(QP_struct* QP_space);


/// support function of -x
mfloat support_function(const mfloat* x, const mfloat* l, const mfloat* u, int len, mfloat* infty_x);

/// read binary file
void read_bin(const char *lpFileName, mfloat* X, int len);
void write_bin(const char *lpFileName, int* X, int len);
void write_bin(const char *lpFileName, mfloat* X, int len);
/// projection on to an Unit Ball. (proximal mapping of conjugate of weighted l2 norm)
void proj_l2(const mfloat* input, mfloat weight, mfloat* output, int len, mfloat& norm_input, bool& rr);

/// projection on to an interval. (proximal mapping of indicate function)
void proj(const mfloat* input, mfloat* output, int len, const mfloat* l, const mfloat* u);

/// weighted l2 norm, column wise for matrix input (m-by-n). Matrix is stored in column order for (one column's) continuous memory.
void proj_l2_mat(const mfloat* input, const mfloat* weight, mfloat* output, int m, int n, mfloat* norm_input, bool* rr, bool byRow = true);
/// vector 2-norm
mfloat norm2(const mfloat* x, int len);

/// distance 2-norm
mfloat distance(const mfloat* x, const mfloat* y, int len);

/// matrix 2-norm
mfloat norm2_mat(const mfloat* x, int m, int n);

/// z = y + a*x. a can be 1 and -1. z is output. z can be y.
void axpy(mfloat a, const mfloat* x, const mfloat* y, mfloat* z, int len);
/// z = a*x. a can be 1 and -1. z is output. z can be x.
void ax(mfloat a, const mfloat* x, mfloat* z, int len);
/// z = y + a*x'. a can be 1 and -1. z is output. z can be y.
void axpyT(mfloat a, const mfloat* x, const mfloat* y, mfloat* z, int m, int n);

/// axpy matrix
void axpy_mat(const mfloat a, const mfloat* x, mfloat* y, int m, int n);

/// z = b*y + a*x
void axpby2(const mfloat a, const mfloat* x, const mfloat b, const mfloat* y, mfloat* z, const int len);

void axpy_R(const mfloat* a, const mfloat* x, const mfloat* y, mfloat* z, int m, int n, bool byRow = true);

/// z = x.*y, dot product. z can be y.
mfloat* xdoty(const mfloat* x, const mfloat* y, int len);

/// matrix dot product.
mfloat* xdoty_mat(const mfloat* x, const mfloat* y, int m, int n);

/// z = <x, y>, inner product. (x^T y)
mfloat xTy(const mfloat* x, const mfloat* y, int len);

/// z = <1, x>, sum of x.
mfloat sum(const mfloat* x, int len);

/// compute the k-nearest neighbours of each column (x m-by-n). k_nearest and distance are output (k-by-n) including the center column.
void knnsearch(const mfloat* x, int k, int m, int n, int* k_nearest, mfloat* dist);

/** compute the weight (|E| vector) and construct the Node-Arc matrix (2-by-|E| array)
    input x (m-by-n), return the number of |E|
 */

/// Bmap: B'*X', x is m-by-n matrix, B is Node-Arc matrix (n-by-|E|)
void Bmap(const mfloat* XT, sparseRowMatrix BT, mfloat* XB);

/// BTmap: B*X', x is |E|-dim vector, B is Node-Arc matrix (n-by-|E|)
void BTmap(const mfloat* XT, sparseRowMatrix B, mfloat* XBT);

/// get dense transpose
Mat get_transpose(Mat X);

/// get dense transpose
void get_transpose(const mfloat* X, mfloat* output, int m, int n);

/// get the sparse row mode of Node-Arc matrix B (n-by-|E|, represented in 2-by-|E| array), for transpose use.
void get_spR(const int* NodeArcMatrix, sparseRowMatrix B, int num);

/// get the sparse row mode of Node-Arc matrix B (n-by-|E|, represented in 2-by-|E| array)
sparseRowMatrix get_BT(sparseRowMatrix BT);

/// sparse matrix vector times for row order matrix
void spMV_R(sparseRowMatrix A, const mfloat* x, int m, int n, mfloat* Ax);

/// sparse matrix vector transpose-times for column order matrix
void partial_spMV_2(sparseRowMatrix A, const mfloat* x, int m, int n, mfloat* Ax, spVec_int nzidx);

/// sparse matrix vector times for column order matrix
//void spMV_C(spC A, Mat x, int m, int n, mfloat* Ax);

/// sum column for m-by-n matrix
void sum(const mfloat* X, mfloat* output, int m, int n);

/// row norm for m-by-n matrix
void norm_R(const mfloat* X, mfloat* output, int m, int n, bool byRow = true);

/// row xdoty for m-by-n matrix
mfloat* xdoty_R(const mfloat* X, const mfloat* Y, int m, int n, bool byRow = true);

void create_find_step();

/// find the step length
void find_step();

/// arrange memory for PCG
void create_PCG();

/// Preconditioned Conjugate gradient method for AX = RHS, matrix level.
void PCG(const mfloat* x0, const mfloat* rhs, CG_struct* CG_space, mfloat* output);

/// Bi-conjugate gradient method for Ax = b, non-symmetric A.
void BiCG(const mfloat* x0, const mfloat* rhs, BiCG_struct* CG_space, mfloat* output);

/// initialize BiCG
void BiCG_init(BiCG_struct* CG_space);

/// arrange memory for My_CG
void create_My_CG();

/// Compute M*y for CG
void My_CG();

/// Compute Ax for BiCG
void BiCG_prod(const mfloat *x, mfloat* output, BiCG_struct *BiCG_space);

/// Compute A'y for BiCG
void My_BiCG_T(const mfloat *x, mfloat* output);

/// create dense m-by-n matrix
Mat creat_Mat(int m, int n);

/// print Matrix
void print_Mat(const mfloat* X, int m, int n);

void partial_spMV_R(sparseRowMatrix A, const mfloat* x, int m, int n, mfloat* Ax, spVec_int nzidx);

/// get submatrix of B
sparseRowMatrix B_sub_Row(spVec_int nzidx, sparseRowMatrix BT);

void create_SNAL();

void SNAL_new();

void create_SSNCG();

/// semismooth Newton
void SSNCG_new();

/// Ly for semismooth Newton subproblem
mfloat obj_val_SSN();

/// initialize SNAL
void initial_SNAL();

/// collect infeasibility
bool collect_infeasibility();

/// primal objective value for original problem
mfloat primal_obj();

/// dual objective value for original problem
mfloat dual_obj();

/// check termination for subproblem
bool check_termination_SSN();

/// update sigma
void update_sigma(QP_struct *QP_space);

/// print sparse matrix
void print_spM(sparseRowMatrix input);

/// create eigen sparse matrix
Eigen::SparseMatrix<mfloat> get_eigen_spMat(sparseRowMatrix BT);

/// compute the L^{-1} * r;
void precond_CG(CG_struct* CG_space);

/// compute the L^{-1} * r;
void precond_BiCG(const mfloat* input, mfloat* output, BiCG_struct* CG_space);

/// initialize Eigen variable for cholesky.
void initial_Eigen();

/// create ADMM variables
void create_ADMM();

/// ADMM for warm-start
void ADMM();

/// update sigma in ADMM
void update_sigma_admm();

void Runexperiment_xTy();

void writeCOO();

int PARDISO_init(PARDISO_var *PDS, sparseRowMatrix A);

int PARDISO_numerical_fact(PARDISO_var *PDS);

int PARDISO_solve(PARDISO_var *PDS, mfloat *rhs);

int PARDISO_release(PARDISO_var *PDS);
#endif //C_CODE_ALGORITHM_H