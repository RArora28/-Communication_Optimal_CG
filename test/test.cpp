#include "PCG.hpp"
#include "./generate_input.hpp"
#include "sparse_matrix_coo.hpp"
using namespace std;

int main() {
	// conjugate gradient - normal 
	vector < vector < double > > X = {{1.0, 1.0}, {2.0, -1.0}};
	vector < vector < double > > Y = {{3.0}, {1.0}};
	Matrix <double> A(2, 2), b(2, 1);
	A.mat = X; 
	b.mat = Y;
	A.mat = X; 
	b.mat = Y;
	Matrix <double> ret;
	// ret = conjugate_gradient(A, b);

	//conjugate gradient on a grid
	auto t = generate_sparse_matrix(200, 200);
	auto x = t.first; 
	auto y = t.second;
	conjugate_gradient(x, y, true);

	// //preconditioned conjugate gradient - Jacobi(Diagnol Matrix)
	// Matrix <double> z(x.n, x.n);
	// for(int i = 0; i < z.getRowSize(); ++i) {
	// 	// z.mat[i][i] = x.mat[i][i];
		
	// }
	// preconditioned_conjugate_gradient(z, x, y, true);
	return 0;
}

