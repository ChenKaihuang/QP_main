// QP_library_wrapper.cpp
#include <Python.h>
#include "algorithm.h"


static PyObject* py_add(PyObject* self, PyObject* args) {
    sparseRowMatrix Q, A;
    mfloat *b, *c, *l, *u;
    PyObject *Q_value, *Q_rowStart, *Q_column;
    PyObject *A_value, *A_rowStart, *A_column;
    PyObject *py_b, *py_c, *py_l, *py_u;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOO", &Q_value, &Q_rowStart, &Q_column, &A_value, &A_rowStart, &A_column, *py_b, *py_c, *py_l, *py_u)) {

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
    if (PyList_Size(Q_rowStart) != n) {
        PyErr_SetString(PyExc_TypeError, "length of Q_row is not n");
    }
    if (PyList_Size(Q_column) != PyList_Size(Q_value)) {
        PyErr_SetString(PyExc_TypeError, "length of Q_value is not equal to Q_column");
    }
    if (PyList_Size(A_rowStart) != m) {
        PyErr_SetString(PyExc_TypeError, "length of A_row is not m");
    }
    if (PyList_Size(A_column) != PyList_Size(A_value)) {
        PyErr_SetString(PyExc_TypeError, "length of A_value is not equal to A_column");
    }


    return PyLong_FromLong(0);
}

static PyMethodDef QP_libraryMethods[] = {
        {"add", py_add, METH_VARARGS, "Add two integers."},
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
