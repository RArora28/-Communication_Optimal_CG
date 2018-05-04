#include <iostream>
#include <cmath>
#include <ctime>
#include "../generate_input.hpp"
#include "parallel_operations.hpp"
#include <cassert>
using namespace std;


const long double EPS = 1e-10; 
#define MASTER 0


int main(int argc, char* argv[]) {
	
	int rank, size; 
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &size);

	if (rank == MASTER) {
		auto t = generate_sparse_matrix(40, 40); 
		auto b = t.first; auto u = t.second;
	  	
	  	Matrix <long double> R0(b), R1(b.getRowSize(), 1);
		Matrix <long double> P0(b), P1(b.getRowSize(), 1);
		Matrix <long double> X0(b.getRowSize(), 1), X1(b.getRowSize(), 1);
		Matrix <long double> temp(b.getRowSize(), 1), another_temp(b.getRowSize(), 1);
		
		int True = 1, False = 0, itr = 0, row_size = 1, n = X0.getRowSize(), sq = u.getRowSize(); 
		clock_t begin = clock();
		cout << "..... Running Solver ....." << endl;

		for(int i = 2; i * i <= sq; ++i) {
			if (sq % i == 0 && i * i + 1 <= size) {
				row_size = i; 
			}
		}

		while(sqrt(dot(R0, R0)) / sqrt(dot(b, b)) >= EPS) {	
			
			for(int i = 1; i < size; ++i) {
				MPI_Send(&True, 1, MPI_INT, i, 1234, MPI_COMM_WORLD);
			}
			
			++itr;
			cout << "#iteration: " << itr << endl;

			// alpha = (trans(R0) * R0) / (trans(P0) * A * P0);
			long double alpha = vector_dot_MASTER(R0, R0, n, size, 4); 
			matrix_vector_mult_MASTER(temp, u, P0, n, sq, row_size, sq / row_size, 5);
			alpha /= vector_dot_MASTER(temp, P0, n, size, 4);
		
			// X1 = X0 + alpha * P0; 
			vector_scalar_mult_MASTER(temp, P0, alpha, n, size, 3);
			vector_sum_MASTER(X1, X0, temp, n, size, 1);

			// R1 = R0 - alpha * A * P0;
			alpha *= -1.0;
			matrix_vector_mult_MASTER(another_temp, u, P0, n, sq, row_size, sq / row_size, 5); 
			vector_scalar_mult_MASTER(temp, another_temp, alpha, n, size, 3);
			vector_sum_MASTER(R1, R0, temp, n, size, 1); 
			
			// beta = (trans(R1) * R1) / (trans(R0) * R0);
			long double beta = vector_dot_MASTER(R1, R1, n, size, 4) / vector_dot_MASTER(R0, R0, n, size, 4);
			
			// P1 = R1 + beta * P0; 
			vector_scalar_mult_MASTER(temp, P0, beta, n, size, 3);
			vector_sum_MASTER(P1, R1, temp, n, size, 1);

			// for the next iteration 
			vector_swap_MASTER(R0, R1, n, size, 2);
			vector_swap_MASTER(X0, X1, n, size, 2); 
			vector_swap_MASTER(P0, P1, n, size, 2);
		}
		
 		for(int i = 1; i < size; ++i) {
 			MPI_Send(&False, 1, MPI_INT, i, 1234, MPI_COMM_WORLD);
 		}

		clock_t end = clock();
  		long double elapsed_secs = (long double)(end - begin) / CLOCKS_PER_SEC;
		cout << "Time Elapsed: " << elapsed_secs << " sec" << endl;
 		cout << "Num. Iterations: " << itr << endl;
 		cout << "Error: " << dot(R0, R0)<< endl;
 	} else {
 		int num_parallel_ops = 15;
 		int rank, operation, cont; 
		MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
 		
 		while(true) {
			MPI_Recv(&cont, 1, MPI_INT, 0, 1234, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (!cont) break; 
			for(int i = 1; i <= num_parallel_ops; ++i) {
			 	MPI_Recv(&operation, 1, MPI_INT, 0, 12345, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (operation == 1) { 
					vector_sum(rank); 
				} else if (operation == 2) {
					vector_swap(rank);
				} else if (operation == 3) {
					vector_scalar_mult(rank);
				} else if (operation == 4) {
					vector_dot(rank);
				} else if (operation == 5) {
					matrix_vector_mult(rank); 
				}
			}
		}
	}

 	MPI_Finalize();
	return 0;
}

