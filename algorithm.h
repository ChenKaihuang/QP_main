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
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>

#include "mkl_pardiso.h"
#include "mkl_types.h"
#include <mkl.h>

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
} linear_solver_chol;

typedef struct {
    bool isEmpty;
    Eigen::SparseLU<Eigen::SparseMatrix<mfloat>> LUsolver;
} linear_solver_lu;

typedef struct {
    int nRow;
    int nCol;
    mfloat* vec;
} Mat;

typedef struct sparse_vec_int{
    int number = 0;
    int* vec = nullptr;
} spVec_int;


typedef struct {
    int nRow;
    int nCol;
    mfloat* value;
    int* rowStart;
    int* rowLength;
    int* column;
} sparseRowMatrix;

typedef struct {
    int row;
    int col;
    mfloat value;
} sparseMatrixElement;

typedef struct {
    int nRow;
    int nCol;
    int nElements;
    sparseMatrixElement* elements;
} sparseMatrixCOO;

typedef struct BiCG_struct{
    mfloat *x,*r,*p,*Ap, *v, *r0, *p_hat, *s, *s_hat, *t, *x0, *Qx, *ATy;
    int dim_n, max_iter = 500, nCGs, dim_m;
    mfloat tol = 1e-6, cgTime, error;
//    linearSolver LLT;
    mfloat *BiCG_prod_temp1, *BiCG_prod_temp2;

} BiCG_struct;

typedef struct {
    mfloat *x,*r,*z,*p,*Ap,*Atp;
    int dim_n, dim_d, max_iter, nCGs;
    mfloat tol, cgTime;
    linear_solver_chol LLT;
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
    int max_ADMM_iter = 100;
    int max_ALM_iter = 10;
    mfloat sigma = 1.0;
    double kappa = 1.0;
    mfloat gamma = 1.618;
    int use_scale = 1;
    int use_mkl = 1;
} input_parameters;

typedef struct output_parameters {
    int iter_ADMM;
    int iter_pALM;
    mfloat primal_obj;
    mfloat dual_obj;
    mfloat time_ADMM;
    mfloat time_pALM;
} output_parameters;

typedef struct QP_struct {
    // input: 1/2 xQx + cx, s.t. Ax = b x \in C = [l, u]
    // quadratic matrix Q and constraint matrix A;
    sparseRowMatrix Q, A;
    // vector c
    mfloat *c, *b, *l, *u;
    int m, n;

    // work variables
    sparseRowMatrix AT, AAT, IQ;
    mfloat *w, *y, *z, *x, sigma, gamma, sigma_old, tau;
    PARDISO_var *PDS_IQ, *PDS_A, *PDS_Q2;
    mfloat *update_y_rhs, *Qw, *ATy, *Ax, *Rd, *Rp, *z_Qw_c, *A_times_z_Qw_c, *dz, *Qx, *Qz, *Az;
    mfloat *z_new, *y_old, *Qw_ATy_c;
    mfloat *w_bar, *y_bar;
    mfloat *update_w_rhs, *z_ATy_c;
    mfloat *update_z_unProjected, *update_z_projected;
    mfloat *pALM_z_unProjected, *pALM_z_projected;
    mfloat inf_p, inf_d, inf_Q, inf_C, inf_g, primal_obj, dual_obj, infty_z;
    mfloat *RQ, *RC;
    mfloat *dw, *w_old, *Qdw;
    mfloat normQ, kappa, SSN_tol;
    mfloat *SSN_grad_Q, *SSN_compute_obj_temp1, *SSN_grad, *SSN_dwdy;
    mfloat sGSADMM_tol = 1e-4, pALM_tol = 1e-6, stop_tol = 1e-6;
    bool sGSADMM_finish;
    bool pALM_finish = false;

//    input_parameters input_para;
    Eigen::SparseMatrix<mfloat> EigenIn, EigenIm, EigenQ, EigenIQ;
    Eigen::SparseMatrix<mfloat> upperMatQ;
    linear_solver_chol Eigen_linear_AAT, Eigen_linear_IQ;
    linear_solver_lu Eigen_linear_hessian_full_idx;
    int iter, max_ADMM_iter, max_ALM_iter;
    Eigen::VectorX<mfloat> Eigen_rhs_w, Eigen_rhs_y, Eigen_rhs_dwdy, Eigen_result_w, Eigen_result_y, Eigen_result_dwdy;
    bool *proj_idx;
    spVec_int U_idx;
    std::chrono::steady_clock::time_point time_solve_start;
    mfloat step_size;
    mfloat *Lorg, *Uorg;
//    Eigen::SparseMatrix<mfloat> EigenA;
    double normborg;
    double normcorg;
    double *cA;
    double *rA;
    double normb;
    double normc;
    double *xorg;
    double bscale;
    double cscale;
    double *zorg;
    double *Rp_org;
    double inf_p_org;
    double *Rd_org;
    double inf_d_org;
    double *Qx_org;
    double *Qw_org;
    double *RQ_org;
    double inf_Q_org;
    double *dzorg;
    double *RCorg;
    double inf_C_org;
    double primal_obj_org;
    double dual_obj_org;
    double inf_g_org;
    int gamma_reset_start = 200;
    bool gamma_test = true;
    mfloat *Qw_old;
    mfloat *Qx_old;
    mfloat *z_old;
    mfloat *x_old;
    int prim_win = 0;
    int dual_win = 0;
    bool fix_sigma = false;
    mfloat sigma_max = 5e6;
    mfloat sigma_min = 1e-3;
    double sigma_scale = 1.35;
    int sigma_change = 0;
    int rescale = 1;
    int sigma_update_iter = 50;
    double *temp_z;
    bool use_inforg = false;
    int max_iter_SSN = 100;
    double dwQdw;
    double x_diff_norm;
    double y_diff_norm;
    int precond = 0;
    mfloat *dQdA;
    mfloat *diag_AAT;
    mfloat *diag_Q;
    mfloat const0;
    /// 0: bicgstab; 1: chol;
    int SSN_method = 1;
    Eigen::SparseMatrix<double> EigenA;
    Eigen::SparseMatrix<double> EigenAT;
    Eigen::SparseMatrix<double> EigenAAT;
    Eigen::SparseMatrix<double> AQ;
    bool first_full_idx = true;
    Eigen::VectorXd diag_modifier, diagNUM;
    Eigen::SparseMatrix<double> hessian_1;
    Eigen::SparseMatrix<double> hessian_4;
    Eigen::SparseMatrix<double> hessian_full_idx;
    double *SSN_grad_invsigma;
    double *ATy0;
    double *ATdy;
    double *z_temp;
    double *x0;
    double *w_diff;
    mfloat *Qw_diff;
    double *line_search_temp1;
    double *y_diff;
    double *line_search_temp2;
    double obj_old;
    double *Rp_sub;
    double *Rd_sub;
    mfloat *w_new, *y_new, *Qw_new;
    std::vector<Eigen::Triplet<mfloat>> tripletList_Qsub;
//    sparseMatrixCOO Q_coo, Q_coo_sub;
//    int *Q_sub_counter;

    Eigen::SparseMatrix<mfloat> Eigen_Q_sub;
    bool *zero_idx;
    double *SSN_rhs, *SSN_rhs_compressed;
    Eigen::SparseMatrix<double> Eigen_A_sub;
    int *proj_idx_sum;
    double *SSN_sub_case1_temp1;
    double *SSN_sub_case1_temp2;
    double *SSN_sub_case1_temp3;
    double *SSN_r1_new;
    int max_iter_step = 40;
    int iter_step;
    double *normGrad_list;
    std::vector<mfloat> inf;
    double case1_time = 0.0;
    double get_eigen_sub_time = 0.0;
    double partial_CR_spMV_time = 0.0;
    double partial_axpy_time = 0.0;
    double axpy_time = 0.0;
    double eigen_matrix_mult_time = 0.0;
    double eigen_solver_time = 0.0;
    double solver_analyze_time = 0.0;
    double solver_factorize_time = 0.0;
    double solver_solve_time = 0.0;
//    bool use_eigen = false;
    int use_mkl = true;
    int use_scale = true;
    int fact_times = 0;
    sparseRowMatrix MKL_IQ;
} QP_struct;

void Eigen_init(QP_struct *QP_space);
void partial_axpby(const mfloat a, const mfloat* x, mfloat b, const mfloat* y, mfloat* z, const int len, bool *idx);
void simple_ruiz_test();

void QP_solve(sparseRowMatrix Q, sparseRowMatrix A, mfloat* b, mfloat* c, mfloat* l, mfloat* u, int m, int n, input_parameters para, output_parameters *para_out);

void sGSADMM_QP(QP_struct *QP_space);

void rescaling(QP_struct *qp);
void rescalingADMM(QP_struct *qp);

void save_solution(QP_struct *qp);
void read_solution(QP_struct *qp);

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

void pALM_QP_compute_status(QP_struct *QP_space);

void sGSADMM_QP_print_status(QP_struct* QP_space);

void sGSADMM_QP_refact_IQ(QP_struct* QP_space);

int sGSADMM_sigma_update_iter(int iter);

int sGSADMM_print_iter(int iter);

void pALM_QP_print_status(QP_struct* QP_space);

int pALM_print_iter(int iter);

void pALM_SSN_QP(QP_struct* QP_space);

void SSN_findStep(QP_struct* QP_space);

mfloat SSN_obj_val(QP_struct* QP_space);

void SSN_compute_grad(QP_struct* QP_space);

void SSN_findStep_update_variables(QP_struct* QP_space);

void pALM_update_variables(QP_struct* QP_space);

void Taylor_test(BiCG_struct* BiCG_space, QP_struct* QP_space);

/// support function of -x
mfloat support_function(const mfloat* x, const mfloat* l, const mfloat* u, int len, mfloat* infty_x);

/// read binary file
void read_bin(const char *lpFileName, mfloat* X, int len);
void write_bin(const char *lpFileName, int* X, int len);
void write_bin(const char *lpFileName, mfloat* X, int len);
/// projection on to an Unit Ball. (proximal mapping of conjugate of weighted l2 norm)
void proj_l2(const mfloat* input, mfloat weight, mfloat* output, int len, mfloat& norm_input, bool& rr);

/// projection on to an interval. (proximal mapping of indicate function)
void proj(const mfloat* input, mfloat* output, int len, const mfloat* l, const mfloat* u, bool *idx = nullptr);

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

/// z = y + a*x. a can be 1 and -1. z is output. z can be y. --partial
void partial_axpy(mfloat a, const mfloat* x, const mfloat* y, mfloat* z, int len, bool *idx);

/// z = b*y + a*x. a can be 1 and -1. z is output. z can be y.
void axpby(mfloat a, const mfloat* x, mfloat b, const mfloat* y, mfloat* z, int len);

void partial_axpby(const mfloat a, const mfloat* x, mfloat b, const mfloat* y, mfloat* z, const int len, bool *idx);

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
void xdoty(const mfloat* x, const mfloat* y, mfloat* z, int len, bool prod = true);

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

/// partial column and row sparse matrix times vector
void partial_CR_spMV(sparseRowMatrix A, const mfloat* x, int nRow, int nCol, mfloat* Ax, bool *row_idx, bool *col_idx);

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
void BiCG(const mfloat* x0, const mfloat* rhs, mfloat* output, BiCG_struct* CG_space, QP_struct *QP_space);

/// initialize BiCG
void BiCG_init(BiCG_struct* CG_space);

/// arrange memory for My_CG
void create_My_CG();

/// Compute M*y for CG
void My_CG();

/// Perform Ruiz equilibration for scaling a m by n matrix
void ruiz_equilibration(sparseRowMatrix A, sparseRowMatrix *A_new, sparseRowMatrix AT, sparseRowMatrix *AT_new, mfloat *D, mfloat *E);

/// pre-processing QP
void preprocessing(QP_struct *qp);

/// Compute Ax for BiCG
void BiCG_prod(const mfloat *x, mfloat* output, BiCG_struct *BiCG_space, QP_struct *QP_space);

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
void update_sigma_pALM(QP_struct* QP_space);

/// update gamma
void update_gamma(QP_struct *qp);

/// print sparse matrix
void print_spM(sparseRowMatrix input);

/// create eigen sparse matrix
Eigen::SparseMatrix<mfloat> get_eigen_spMat(sparseRowMatrix B);

/// create eigen sparse sub-matrix
void get_eigen_sub_spMat_Q(sparseRowMatrix B, bool *idx, std::vector<Eigen::Triplet<mfloat>> tripletList, int *idx_sum, Eigen::SparseMatrix<mfloat> *mat);
void get_eigen_sub_spMat_A(sparseRowMatrix B, bool *idx, std::vector<Eigen::Triplet<mfloat>> tripletList, int *idx_sum, Eigen::SparseMatrix<mfloat> *mat);

/// compute the L^{-1} * r;
void precond_CG(CG_struct* CG_space);

/// compute the L^{-1} * r;
void precond_BiCG(const mfloat* input, mfloat* output, QP_struct *qp);

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

//void run_bin(const char* file_name);

void line_search(QP_struct *qp);

int PARDISO_init(PARDISO_var *PDS, sparseRowMatrix A);
int PARDISO_init(PARDISO_var *PDS, Eigen::SparseMatrix<double> *A);

int PARDISO_numerical_fact(PARDISO_var *PDS);

int PARDISO_solve(PARDISO_var *PDS, mfloat *rhs);

int PARDISO_release(PARDISO_var *PDS);
#endif //C_CODE_ALGORITHM_H

Eigen::SparseMatrix<mfloat> concat_4_spMat(Eigen::SparseMatrix<mfloat> A, Eigen::SparseMatrix<mfloat> B, Eigen::SparseMatrix<mfloat> C, Eigen::SparseMatrix<mfloat> D);

void test_concat_4_spMat();

void solve_grb();

void CSR_to_COO(sparseRowMatrix Q, sparseMatrixCOO *Q_coo);

void printSparseMatrix(const sparseMatrixCOO* matrix);

void test_CSR_to_COO();

void test_eigen_sub();

void SSN_sub_case1(QP_struct *qp);

mfloat vec_max(const mfloat *x, int len);

void test_eigen_segment_set();

void test_MKL_pardiso();

void upper_sparseRowMatrix(sparseRowMatrix A, sparseRowMatrix *A_upper);

void load_para(QP_struct *qp, input_parameters para);

void vSet(mfloat *x, int len, mfloat val);

void MKL_init(QP_struct *qp);

void MKL_refact(QP_struct* qp);