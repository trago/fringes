#ifndef FLOODFILL_UNWRAP_H
#define FLOODFILL_UNWRAP_H

typedef unsigned char uint8_t;
struct pixel_t{
  size_t row;
  size_t col;
};

void floodfill_unwrap(double* pp, uint8_t* mask, double* up,
		      size_t row, size_t col,
		      int mm, int nn);
void get_neighborhood(pixel_t pixel, uint8_t* mask, uint8_t* visited,
		      pixel_t* p_neighbors, size_t* neighbors_dim,
		      size_t mm, size_t nn);

#endif
