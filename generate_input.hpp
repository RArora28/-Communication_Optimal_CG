
#ifndef GENERATE_INPUT_HPP_
#define GENERATE_INPUT_HPP_

#include "Matrix_Classes/dense_matrix.hpp"
#include "Matrix_Classes/sparse_matrix_coo.hpp"
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

pair <Matrix <long double>, Matrix <long double>> generate_dense_matrix(int nx_max, int ny_max) {
	int nx = nx_max; 
	int ny = ny_max; 
	Matrix < long double > b(nx * ny, 1);
	for(int i = 0; i < b.getRowSize(); ++i) {
		b.mat[i][0] = (rand() % val_mx) * 1.0;
	}
	Matrix < long double > A(nx * ny, nx * ny);
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
pair <Matrix <long double>, Matrix <int> > generate_sparse_matrix(int nx_max, int ny_max) {
	int nx = nx_max; 
	int ny = ny_max; 
	Matrix < long double > b(nx * ny, 1);
	for(int i = 0; i < b.getRowSize(); ++i) {
		b.mat[i][0] = (rand() % val_mx) * 1.0;
	}
	Matrix < int > u(nx, ny); 
	for(int i = 0; i < nx; ++i) {
		for(int j = 0; j < ny; ++j) {
			u.mat[i][j] = map_to_int(i, j, nx, ny); 
		}
	}
	return {b, u}; 
}

#endif