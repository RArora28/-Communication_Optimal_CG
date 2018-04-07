
#ifndef SPARSE_MATRIX_COO_
#define SPARSE_MATRIX_COO_

#include <iostream>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "matrix.hpp"
using namespace std;

template < typename T > 
class Matrix_coo {
public:
	vector < int > row, col;
	vector < T > val; 
	int size, n, m;  
	
	Matrix_coo() {}
	
	Matrix_coo(int N, int M) {
		n = N, m = M; 
	}

	void Insert_element(int a, int b, T c) {
		row.push_back(a); 
		col.push_back(b); 
		val.push_back(c); 
		size = row.size();
		return; 
	}

 	void print() {
		cout << "...matrix contents(COO)..." << endl;
		for(int i = 0; i < size; ++i) {
			cout << "[" << row[i] << "," << col[i] << "]: " << val[i] << endl;  
		}
		return;
	}
}; 

template < class T>
Matrix <T> operator * (Matrix_coo <T> A, Matrix <T> b) {
	Matrix < T > ret(A.n, b.getColSize());
	for(size_t i = 0; i < A.size; ++i) {
		for(size_t j = 0; j < b.getColSize(); ++j) {
			ret.mat[A.row[i]][j] += A.val[i] * b.mat[A.col[i]][j];
		}
	}
	return ret; 
}	

template < class T>
Matrix_coo <T> operator * (const T& val, Matrix_coo <T> A) {
	Matrix_coo < T > ret(A.n, A.m);
	for(size_t i = 0; i < A.size; ++i) {
		ret.Insert_element(A.row[i], A.col[i], val * A.val[i]); 
	}
	return ret; 
}	
#endif 