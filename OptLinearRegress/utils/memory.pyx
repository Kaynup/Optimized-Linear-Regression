cimport cython
from libc.stdlib cimport malloc, free

cdef class DoubleArray:
    cdef double* data
    cdef int size

    def __cinit__(self, int size):
        self.size = size
        self.data = <double*>malloc(size * sizeof(double))
        if self.data == NULL:
            raise MemoryError("Allocation failed")

    def __dealloc__(self):
        if self.data != NULL:
            free(self.data)
