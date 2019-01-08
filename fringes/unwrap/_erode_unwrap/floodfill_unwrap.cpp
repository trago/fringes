#include <list>
#include <cmath>
#include <iostream>
#include "floodfill_unwrap.h"

using namespace std;

inline double unwrap_value(double v1, double v2)
{
  double wrap_diff = round(v2 - v1);
  return v2 - wrap_diff;
}

void floodfill_unwrap(double* pp, uint8_t* mask, double* up,
		      size_t row, size_t col,
		      int mm, int nn)
{
  uint8_t* visited = new uint8_t[mm*nn];
  std::list<pixel_t> pixel_queue;
  pixel_t pixel;
  pixel_t neighbors[9];
  size_t neighbors_dim;
  pixel.col = col;
  pixel.row = row;

  for(int n=0; n<mm*nn; n++)
    visited[n] = 0;

  //cout << "(" << row << ", " << col << ")" << endl;
  up[pixel.row*nn + pixel.col] = pp[pixel.row*nn + pixel.col];
  do{
    get_neighborhood(pixel, mask, visited, neighbors, &neighbors_dim,
		     mm, nn);
    for(size_t n=0; n<neighbors_dim; n++){
      up[neighbors[n].row*nn + neighbors[n].col] =
	unwrap_value(up[pixel.row*nn + pixel.col],
		     pp[neighbors[n].row*nn + neighbors[n].col]);
      pixel_queue.push_back(neighbors[n]);
    }
    pixel = pixel_queue.front();
    pixel_queue.pop_front();
  }while(!pixel_queue.empty());
  delete [] visited;
}

void get_neighborhood(pixel_t pixel, uint8_t* mask, uint8_t* visited,
		      pixel_t* p_neighbors, size_t* neighbors_dim,
		      size_t mm, size_t nn)
{
  size_t col = pixel.col+1;
  size_t row = pixel.row+1;

  int dim = 0;
  if(col >=0 && col < nn && row>=0 && row < mm)
    if(mask[row*nn + col] && !visited[row*nn + col]){
      p_neighbors[dim].col = col;
      p_neighbors[dim].row = row;
      visited[row*nn + col] = 1;
      dim++;
    }
  col--;
  if(col >=0 && col < nn && row>=0 && row < mm)
    if(mask[row*nn + col] && !visited[row*nn + col]){
      p_neighbors[dim].col = col;
      p_neighbors[dim].row = row;
      visited[row*nn + col] = 1;
      dim++;
    }
  col--;
  if(col >=0 && col < nn && row>=0 && row < mm)
    if(mask[row*nn + col] && !visited[row*nn + col]){
      p_neighbors[dim].col = col;
      p_neighbors[dim].row = row;
      visited[row*nn + col] = 1;
      dim++;
    }
  row--;
  if(col >=0 && col < nn && row>=0 && row < mm)
    if(mask[row*nn + col] && !visited[row*nn + col]){
      p_neighbors[dim].col = col;
      p_neighbors[dim].row = row;
      visited[row*nn + col] = 1;
      dim++;
    }
  col++;
  if(col >=0 && col < nn && row>=0 && row < mm)
    if(mask[row*nn + col] && !visited[row*nn + col]){
      p_neighbors[dim].col = col;
      p_neighbors[dim].row = row;
      visited[row*nn + col] = 1;
      dim++;
    }
  col++;
  if(col >=0 && col < nn && row>=0 && row < mm)
    if(mask[row*nn + col] && !visited[row*nn + col]){
      p_neighbors[dim].col = col;
      p_neighbors[dim].row = row;
      visited[row*nn + col] = 1;
      dim++;
    }
  row--;
  if(col >=0 && col < nn && row>=0 && row < mm)
    if(mask[row*nn + col] && !visited[row*nn + col]){
      p_neighbors[dim].col = col;
      p_neighbors[dim].row = row;
      visited[row*nn + col] = 1;
      dim++;
    }
  col--;
  if(col >=0 && col < nn && row>=0 && row < mm)
    if(mask[row*nn + col] && !visited[row*nn + col]){
      p_neighbors[dim].col = col;
      p_neighbors[dim].row = row;
      visited[row*nn + col] = 1;
      dim++;
    }
  col--;
  if(col >=0 && col < nn && row>=0 && row < mm)
    if(mask[row*nn + col] && !visited[row*nn + col]){
      p_neighbors[dim].col = col;
      p_neighbors[dim].row = row;
      visited[row*nn + col] = 1;
      dim++;
    }

  *neighbors_dim = dim;
}
