
#ifndef GENERATE_INPUT_HPP_
#define GENERATE_INPUT_HPP_

#include "matrix.hpp"
#include "sparse_matrix_coo.hpp"
#include <iostream>
using namespace std; 

const int dim_mx = 20; 
const int val_mx = 1000000; 

int map_to_int(int i, int j, int nx, int ny) {
	if (i >= 0 && i < nx && j >= 0 && j < ny) {
		return i * ny + j; 
	}
	return -1; 
} 

int dx[] = {-1, 0, 1, 0}; 
int dy[] = {0, 1, 0, -1}; 

pair <Matrix <double>, Matrix <double>> generate_dense_matrix(int nx_max, int ny_max) {
	int nx = rand() % nx_max + 1; 
	int ny = rand() % ny_max + 1; 
	// cout << "nx: " << nx << ", " << "ny: " << ny << endl;
	// nx = 40; 
	// ny = 40;
	Matrix < double > b(nx * ny, 1);
	for(int i = 0; i < b.getRowSize(); ++i) {
		b.mat[i][0] = (rand() % val_mx) * 1.0;
	}
	Matrix < double > A(nx * ny, nx * ny);
	for(int i = 0; i < A.getRowSize(); ++i) {
		for(int j = 0; j < A.getColSize(); ++j) {
			A.mat[i][j] = 0;
		}
	} 
	for(int i = 0; i < nx; ++i) {
		for(int j = 0; j < ny; ++j) {
			int curr = map_to_int(i, j, nx, ny);
			for(int k = 0; k < 4; ++k) {
				int r = map_to_int(i + dx[k], j + dy[k], nx, ny);
				if (r != -1) A.mat[curr][r] = 1; 
			}
			A.mat[curr][curr] = -4; 
		}
	}
	return {A, b}; 
}

pair <Matrix_coo <double>, Matrix <double>> generate_sparse_matrix(int nx_max, int ny_max) {
	int nx = rand() % nx_max + 1; 
	int ny = rand() % ny_max + 1; 
	nx = 40, ny = 40;
	Matrix < double > b(nx * ny, 1);
	for(int i = 0; i < b.getRowSize(); ++i) {
		b.mat[i][0] = (rand() % val_mx) * 1.0;
	}
	Matrix_coo < double > A(nx * ny, nx * ny); 

	for(int i = 0; i < nx; ++i) {
		for(int j = 0; j < ny; ++j) {
			int curr = map_to_int(i, j, nx, ny);
			for(int k = 0; k < 4; ++k) {
				int r = map_to_int(i + dx[k], j + dy[k], nx, ny);
				if (r != -1) A.Insert_element(curr, r, 1.0); 
			}
			A.Insert_element(curr, curr, -4.0);
		}
	}
	return {A, b}; 
}

#endif