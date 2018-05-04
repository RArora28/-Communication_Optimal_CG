
#ifndef CG
#define CG

#include <iostream>
#include <cmath>
#include <ctime>
#include "../Matrix_Classes/dense_matrix.hpp"
#include "../Matrix_Classes/sparse_matrix_coo.hpp"
using namespace std;

template <class T>
T Adot(Matrix<T>& a, Matrix_coo<T>& A, Matrix<T> &b) {
	Matrix <T> ret = trans(a) * (A * b);
	return ret.mat[0][0];
}

const double EPS = 1e-10; 

template <class T>
Matrix <T> conjugate_gradient(Matrix_coo <T>& A, Matrix <T>& b) {
  	clock_t begin = clock();
  	Matrix <T> R0(b.getRowSize(), 1), R1(b.getRowSize(), 1);
	Matrix <T> P0(b.getRowSize(), 1), P1(b.getRowSize(), 1);
	Matrix <T> X0(b.getRowSize(), 1), X1(b.getRowSize(), 1);
	int itr = 0;
	cout << "..... Running Normal Solver ....." << endl;
	P0 = R0 = b - (A * X0); 
	while(sqrt(dot(R0, R0)) / sqrt(dot(b, b)) >= EPS) {
		++itr;
		T alpha = dot(R0, R0) / Adot(P0, A, P0);
		X1 = X0  + alpha * P0; 
		R1 = R0 - alpha * A * P0;
		T beta = dot(R1, R1) / dot(R0, R0);
		P1 = R1 + beta * P0;
		swap(R0, R1); 
		swap(X0, X1); 
		swap(P0, P1);
	}
	clock_t end = clock();
  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Time Elapsed: " << elapsed_secs << " sec" << endl;
 	cout << "Iterations: " << itr << endl;
 	cout << "Absolute Error: " << dot(b - A * X0, b - A * X0)<< endl;
 	cout << endl;
 	return X0;
} 

template <class T>
Matrix <T> preconditioned_conjugate_gradient(Matrix_coo<T>& B, Matrix_coo<T>& A, Matrix <T>& b) {
  	clock_t begin = clock();
  	Matrix <T> R0(b.getRowSize(), 1), R1(b.getRowSize(), 1);
	Matrix <T> P0(b.getRowSize(), 1), P1(b.getRowSize(), 1);
	Matrix <T> X0(b.getRowSize(), 1), X1(b.getRowSize(), 1);
	Matrix <T> Z0(b.getRowSize(), 1), Z1(b.getRowSize(), 1);
	Matrix <T> br(b.getRowSize(), 1);
	int itr = 0;
	R0 = b - A * X0; 
	P0 = Z0 = B * R0; 
	cout << "..... Running Preconditioned Solver ....." << endl;
	while((sqrt(dot(R0, R0)) / sqrt(dot(b, b))) >= EPS) {	
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
	cout << "Time Elapsed: " << elapsed_secs << " sec" << endl;
 	cout << "Iterations: " << itr << endl;
 	cout << "Absolute Error: " << dot(b - A * X0, b - A * X0) << endl;
 	cout << endl;
 	return X0;
} 
#endif 

