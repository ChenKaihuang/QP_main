// QP_library_wrapper.cpp
#include <Python.h>
#include "algorithm.h"


static PyObject* py_QP_solve(PyObject* self, PyObject* args) {
    sparseRowMatrix Q, A;
    mfloat *b, *c, *l, *u;
    PyObject *Q_value, *Q_rowStart, *Q_column;
    PyObject *A_value, *A_rowStart, *A_column;
    PyObject *py_b, *py_c, *py_l, *py_u;
    PyObject *inputDict;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOO", &Q_value, &Q_rowStart, &Q_column, &A_value, &A_rowStart, &A_column, &py_b, &py_c, &py_l, &py_u, &inputDict)) {
        std::cout << "error" << std::endl;
        return NULL;
    }

    if (!PyList_Check(Q_value)) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must be lists");
        return NULL;
    }
    if (!PyList_Check(Q_rowStart)) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must be lists");
        return NULL;
    }
    if (!PyList_Check(Q_column)) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must be lists");
        return NULL;
    }

    if (!PyList_Check(A_value)) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must be lists");
        return NULL;
    }
    if (!PyList_Check(A_rowStart)) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must be lists");
        return NULL;
    }
    if (!PyList_Check(A_column)) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must be lists");
        return NULL;
    }

    if (!PyList_Check(py_b)) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must be lists");
        return NULL;
    }
    if (!PyList_Check(py_c)) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must be lists");
        return NULL;
    }
    if (!PyList_Check(py_l)) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must be lists");
        return NULL;
    }
    if (!PyList_Check(py_u)) {
        PyErr_SetString(PyExc_TypeError, "Input arrays must be lists");
        return NULL;
    }

    int m = PyList_Size(py_b);
    int n = PyList_Size(py_c);
    if (PyList_Size(py_l) != n) {
        PyErr_SetString(PyExc_TypeError, "length of l is not n");
    }
    if (PyList_Size(py_u) != n) {
        PyErr_SetString(PyExc_TypeError, "length of u is not n");
    }
    if (PyList_Size(Q_rowStart) != n+1) {
        PyErr_SetString(PyExc_TypeError, "length of Q_row is not n");
    }
    if (PyList_Size(Q_column) != PyList_Size(Q_value)) {
        PyErr_SetString(PyExc_TypeError, "length of Q_value is not equal to Q_column");
    }
    if (PyList_Size(A_rowStart) != m+1) {
        PyErr_SetString(PyExc_TypeError, "length of A_row is not m");
    }
    if (PyList_Size(A_column) != PyList_Size(A_value)) {
        PyErr_SetString(PyExc_TypeError, "length of A_value is not equal to A_column");
    }

    int Q_nnz = PyList_Size(Q_value);
    Q.nRow = n; Q.nCol = n;
    Q.rowStart = new int [n+1];
    Q.column = new int [Q_nnz];
    Q.value = new mfloat [Q_nnz];

    PyObject *item;
    for (int i = 0; i < n+1; ++i) {
        item = PyList_GetItem(Q_rowStart, i);
        Q.rowStart[i] = PyLong_AsLong(item);
//        std::cout << Q.rowStart[i] << std::endl;
    }
    for (int i = 0; i < Q_nnz; ++i) {
        item = PyList_GetItem(Q_column, i);
        Q.column[i] = PyLong_AsLong(item);
        item = PyList_GetItem(Q_value, i);
        Q.value[i] = PyFloat_AsDouble(item);
    }
//    std::cout << "n = " << n << std::endl;
//    print_spM(Q);
    int A_nnz = PyList_Size(A_column);
    A.nRow = m; A.nCol = n;
    A.rowStart = new int [m+1];
    A.column = new int [A_nnz];
    A.value = new mfloat [A_nnz];


    for (int i = 0; i < m+1; ++i) {
        item = PyList_GetItem(A_rowStart, i);
        A.rowStart[i] = PyLong_AsLong(item);
    }
    for (int i = 0; i < A_nnz; ++i) {
        item = PyList_GetItem(A_column, i);
        A.column[i] = PyLong_AsLong(item);
        item = PyList_GetItem(A_value, i);
        A.value[i] = PyFloat_AsDouble(item);
    }
//    print_spM(A);
    b = new mfloat [m];
    c = new mfloat [n];
    l = new mfloat [n];
    u = new mfloat [n];

    for (int i = 0; i < m; ++i) {
        item = PyList_GetItem(py_b, i);
        b[i] = PyFloat_AsDouble(item);
    }

    for (int i = 0; i < n; ++i) {
        item = PyList_GetItem(py_c, i);
        c[i] = PyFloat_AsDouble(item);
        item = PyList_GetItem(py_l, i);
        l[i] = PyFloat_AsDouble(item);
        item = PyList_GetItem(py_u, i);
        u[i] = PyFloat_AsDouble(item);
    }

    input_parameters para;
    PyObject* maxADMMiter = PyDict_GetItemString(inputDict, "maxADMMiter");
    if (maxADMMiter) {
        if (!PyArg_Parse(maxADMMiter, "i", &(para.max_ADMM_iter))) {
            return NULL;
        }
    }

    sGSADMM_QP(Q, A, b, c, l, u, m, n, para);
    return PyLong_FromLong(0);
}

static PyMethodDef QP_libraryMethods[] = {
        {"QP_solve", py_QP_solve, METH_VARARGS, "Solve the QP."},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef QP_libraryModule = {
        PyModuleDef_HEAD_INIT,
        "QP_library",   // Name of Python module
        NULL,            // Module documentation
        -1,              // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
        QP_libraryMethods
};

PyMODINIT_FUNC PyInit_QP_library(void) {
    return PyModule_Create(&QP_libraryModule);
}
