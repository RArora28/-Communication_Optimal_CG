#include "../src/CG.hpp"
#include "../generate_input.hpp"
using namespace std;

int main() {
	
	//conjugate gradient on a grid
	auto t = generate_sparse_matrix(40, 40);
	auto x = t.first; 
	auto y = t.second;
	conjugate_gradient(x, y);

	// preconditioned conjugate gradient on a grid - Jacobi(Diagnol Matrix)
	Matrix_coo <double> z(x.n, x.m); 
	for(int i = 0; i < x.size; ++i) {
		if (x.row[i] == x.col[i]) {
			z.Insert_element(x.row[i], x.col[i], x.val[i]);
		}
	}
	preconditioned_conjugate_gradient(z, x, y);
	return 0;
}

