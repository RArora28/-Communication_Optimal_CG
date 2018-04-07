
#ifndef PCG
#define PCG

#include <iostream>
#include <cmath>
#include <ctime>
#include "CG.hpp"
#include "matrix.hpp"
using namespace std;

template <class T>
Matrix <T> preconditioned_conjugate_gradient(Matrix<T> B, Matrix <T> A, Matrix <T> b, bool symmetric = false) {
  	clock_t begin = clock();
  	Matrix <T> R0(b.getRowSize(), 1), R1(b.getRowSize(), 1);
	Matrix <T> P0(b.getRowSize(), 1), P1(b.getRowSize(), 1);
	Matrix <T> X0(b.getRowSize(), 1), X1(b.getRowSize(), 1);
	Matrix <T> Z0(b.getRowSize(), 1), Z1(b.getRowSize(), 1);
	Matrix <T> br(b.getRowSize(), 1);
	int itr = 0;
	if (!symmetric) { 
		b = trans(A) * b;
		A = trans(A) * A; 
	}
	R0 = b - A * X0; 
	P0 = Z0 = B * R0; 
	cout << ".....Running Solver....." << endl;
	while(sqrt(dot(R0, R0)) >= EPS) {	
		++itr;
		T alpha = dot(R0, Z0) / Adot(P0, A, P0);
		X1 = X0  + alpha * P0; 
		R1 = R0 - alpha * A * P0;
		Z1 = B * R1; 
		T beta = dot(R1, Z1) / dot(R0, Z0);
		P1 = Z1 + beta * P0;
		swap(R0, R1); 
		swap(X0, X1); 
		swap(P0, P1);
		swap(Z0, Z1);
		br = b - A * X0; 
	}
	clock_t end = clock();
  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "...Statistics..." << endl;
 	cout << "Time Elapsed: " << elapsed_secs << " sec" << endl;
 	cout << "Iterations: " << itr << endl;
 	cout << "Error: " << dot(b - A * X0, b - A * X0) << endl;
 	cout << "X-values: ";
 	for(size_t i = 0; i < X0.getRowSize(); ++i) cout << X0.getVal(i, 0) << ' ';
 	cout << endl;
 	return X0;
} 

#endif 

