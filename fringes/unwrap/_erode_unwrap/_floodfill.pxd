ctypedef unsigned char uint8_t

cdef extern from "floodfill_unwrap.h":
    void _floodfill_unwrap 'floodfill_unwrap'(double* pp, uint8_t* mask, double* up,
		      size_t row, size_t col,
		      int mm, int nn)