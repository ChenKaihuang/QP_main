#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "algorithm.h"

using namespace std;

/* PARDISO prototype. */
extern "C" void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
extern "C" void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
                             double *, int    *,    int *, int *,   int *, int *,
                             int *, double *, double *, int *, double *);
extern "C" void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
extern "C" void pardiso_chkvec     (int *, int *, double *, int *);
extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                                    double *, int *);


int main( void )
{
    simple_QP_test();
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