#include <iostream>
#include <cmath>
#include <ctime>
#include "../Matrix_Classes/dense_matrix.hpp"
#include "../Matrix_Classes/sparse_matrix_coo.hpp"
#include "../generate_input.hpp"
#include <mpi.h>
#include <cassert>
using namespace std;

const double EPS = 1e-10; 

template <class T>
T Adot(Matrix<T>& a, Matrix_coo<T>& A, Matrix<T> &b) {
	Matrix <T> ret = trans(a) * (A * b);
	return ret.mat[0][0];
}

void parallel_sum_rank0(Matrix<double>& C, Matrix <double>& A, Matrix <double>& B, int n, int size, int op) {
	int operation = op, subDivide = max((n / (size - 1)), 1), start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i) {
		if (i == size - 1) end = n - 1;
		MPI_Send(&operation, 1, MPI_INT, i, 12345, MPI_COMM_WORLD);
		MPI_Send(&start, 1, MPI_INT, i, 1e5, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_INT, i, 2e5, MPI_COMM_WORLD);
		for(int j = start; j <= end; j++) MPI_Send(&A.mat[j][0], 1, MPI_DOUBLE, i, j + 1e6, MPI_COMM_WORLD);
		for(int j = start; j <= end;j++) MPI_Send(&B.mat[j][0], 1, MPI_DOUBLE, i, j + 2e6, MPI_COMM_WORLD);
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
	start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i){
		if(i == size - 1) end = n - 1;
		for(int j = start; j <= end; j++) MPI_Recv(&C.mat[j][0], 1, MPI_DOUBLE, i, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
}
void parallel_sum_rankI(int rank) {
	int n, start, end;
	MPI_Recv(&start, 1, MPI_INT, 0, 1e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&end, 1, MPI_INT, 0, 2e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	Matrix < double > A(end - start + 1, 1), B(end - start + 1, 1), C(end - start + 1, 1); 
	for(int i = 0; i <= end - start; i++) MPI_Recv(&A.mat[i][0], 1, MPI_DOUBLE, 0, i + start + 1e6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	for(int i = 0; i <= end - start; i++) MPI_Recv(&B.mat[i][0], 1, MPI_DOUBLE, 0, i + start + 2e6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	for(int i = 0; i <= end - start; ++i) C.mat[i][0] = A.mat[i][0] + B.mat[i][0];
	for(int i = 0; i <= end - start; ++i) MPI_Send(&C.mat[i][0], 1, MPI_DOUBLE, 0, i + start, MPI_COMM_WORLD);
	return;
}

void parallel_swap_rank0(Matrix <double>& A, Matrix <double>& B, int n, int size, int op) {
	int operation = op; 
	int subDivide = max((n / (size - 1)), 1);
	int start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i) {
		if (i == size - 1) end = n - 1;
		MPI_Send(&operation, 1, MPI_INT, i, 12345, MPI_COMM_WORLD);
		MPI_Send(&start, 1, MPI_INT, i, 1e5, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_INT, i, 2e5, MPI_COMM_WORLD);
		for(int j = start; j <= end; j++) MPI_Send(&A.mat[j][0], 1, MPI_DOUBLE, i, j + 1e6, MPI_COMM_WORLD);
		for(int j = start; j <= end;j++) MPI_Send(&B.mat[j][0], 1, MPI_DOUBLE, i, j + 2e6, MPI_COMM_WORLD);
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
	start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i){
		if(i == size - 1) end = n - 1;
		for(int j = start; j <= end; j++) MPI_Recv(&A.mat[j][0], 1, MPI_DOUBLE, i, j + 2e7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		for(int j = start; j <= end; j++) MPI_Recv(&B.mat[j][0], 1, MPI_DOUBLE, i, j + 1e7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
}
void parallel_swap_rankI(int rank) {
	int n, start, end;
	MPI_Recv(&start, 1, MPI_INT, 0, 1e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&end, 1, MPI_INT, 0, 2e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	Matrix < double > A(end - start + 1, 1), B(end - start + 1, 1);
	for(int i = 0; i <= end - start; i++) MPI_Recv(&A.mat[i][0], 1, MPI_DOUBLE, 0, i + start + 1e6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	for(int i = 0; i <= end - start; i++) MPI_Recv(&B.mat[i][0], 1, MPI_DOUBLE, 0, i + start + 2e6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	for(int i = 0; i <= end - start; ++i) MPI_Send(&B.mat[i][0], 1, MPI_DOUBLE, 0, i + start + 2e7, MPI_COMM_WORLD);
	for(int i = 0; i <= end - start; ++i) MPI_Send(&A.mat[i][0], 1, MPI_DOUBLE, 0, i + start + 1e7, MPI_COMM_WORLD);
	return;
}

void parallel_scalar_mult0(Matrix<double>& C, Matrix <double>& A, double alpha, int n, int size, int op) {
	int operation = op, subDivide = max((n / (size - 1)), 1), start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i) {
		if (i == size - 1) end = n - 1;
		MPI_Send(&operation, 1, MPI_INT, i, 12345, MPI_COMM_WORLD);
		MPI_Send(&start, 1, MPI_INT, i, 1e5, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_INT, i, 2e5, MPI_COMM_WORLD);
		MPI_Send(&alpha, 1, MPI_DOUBLE, i, 4e5, MPI_COMM_WORLD);
		for(int j = start; j <= end; j++) MPI_Send(&A.mat[j][0], 1, MPI_DOUBLE, i, j + 1e6, MPI_COMM_WORLD);
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
	start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i){
		if(i == size - 1) end = n - 1;
		for(int j = start; j <= end; j++) MPI_Recv(&C.mat[j][0], 1, MPI_DOUBLE, i, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
}
void parallel_scalar_multI(int rank) {
	int n, start, end;
	double alpha;
	MPI_Recv(&start, 1, MPI_INT, 0, 1e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&end, 1, MPI_INT, 0, 2e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&alpha, 1, MPI_DOUBLE, 0, 4e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	Matrix < double > A(end - start + 1, 1), C(end - start + 1, 1); 
	for(int i = 0; i <= end - start; i++) MPI_Recv(&A.mat[i][0], 1, MPI_DOUBLE, 0, i + start + 1e6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	for(int i = 0; i <= end - start; ++i) C.mat[i][0] = alpha * A.mat[i][0];
	for(int i = 0; i <= end - start; ++i) MPI_Send(&C.mat[i][0], 1, MPI_DOUBLE, 0, i + start, MPI_COMM_WORLD);
	return;
}

double parallel_vector_vector_dot0(Matrix <double>& A, Matrix<double>& B, int n, int size, int op) {
	int operation = op, subDivide = max((n / (size - 1)), 1), start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i) {
		if (i == size - 1) end = n - 1;
		MPI_Send(&operation, 1, MPI_INT, i, 12345, MPI_COMM_WORLD);
		MPI_Send(&start, 1, MPI_INT, i, 1e5, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_INT, i, 2e5, MPI_COMM_WORLD);
		for(int j = start; j <= end; j++) MPI_Send(&A.mat[j][0], 1, MPI_DOUBLE, i, j + 1e6, MPI_COMM_WORLD);
		for(int j = start; j <= end; j++) MPI_Send(&B.mat[j][0], 1, MPI_DOUBLE, i, j + 2e6, MPI_COMM_WORLD);
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
	double dot = 0;
	start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i){
		if(i == size - 1) end = n - 1;
		double dot1; 
		MPI_Recv(&dot1, 1, MPI_DOUBLE, i, i * 10 + 123121, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		dot += dot1; 
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
	return dot; 
}
void parallel_vector_vector_dotI(int rank) {
	int n, start, end;
	double alpha;
	MPI_Recv(&start, 1, MPI_INT, 0, 1e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&end, 1, MPI_INT, 0, 2e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	Matrix < double > A(end - start + 1, 1), B(end - start + 1, 1); 
	for(int i = 0; i <= end - start; i++) MPI_Recv(&A.mat[i][0], 1, MPI_DOUBLE, 0, i + start + 1e6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	for(int i = 0; i <= end - start; i++) MPI_Recv(&B.mat[i][0], 1, MPI_DOUBLE, 0, i + start + 2e6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	double dot = 0;
	for(int i = 0; i <= end - start; ++i) dot += A.mat[i][0] * B.mat[i][0];
	MPI_Send(&dot, 1, MPI_DOUBLE, 0, rank * 10 + 123121, MPI_COMM_WORLD);
	return;
}


int main(int argc, char* argv[]) {
	
	auto t = generate_sparse_matrix(40, 40);
	auto A = t.first; 
	auto b = t.second;
  	
  	Matrix <double> R0(b.getRowSize(), 1), R1(b.getRowSize(), 1);
	Matrix <double> P0(b.getRowSize(), 1), P1(b.getRowSize(), 1);
	Matrix <double> X0(b.getRowSize(), 1), X1(b.getRowSize(), 1);
	Matrix <double> temp(b.getRowSize(), 1), another_temp(b.getRowSize(), 1);
	
	int rank, size; 
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &size);

	int True = 1, False = 0; 

	if (rank == 0) {
		
		clock_t begin = clock();
		int itr = 0;
		cout << "..... Running Solver ....." << endl;
		R0 = b - (A * X0); 
		P0 = b - (A * X0);
		int n = X0.getRowSize(); 

		while(sqrt(dot(R0, R0)) / sqrt(dot(b, b)) >= EPS) {	

			for(int i = 1; i < size; ++i) {
				MPI_Send(&True, 1, MPI_INT, i, 1234, MPI_COMM_WORLD);
			}
			++itr;
			cout << "iterations: " << itr << endl;	
			
			double alpha = parallel_vector_vector_dot0(R0, R0, n, size, 4) / Adot(P0, A, P0);
			
			// X1 = X0 + alpha * P0; 
			parallel_scalar_mult0(temp, P0, alpha, n, size, 3);
			parallel_sum_rank0(X1, X0, temp, n, size, 1);

			// R1 = R0 - alpha * A * P0;
			alpha *= -1.0; 
			another_temp = A * P0; 
			parallel_scalar_mult0(temp, another_temp, alpha, n, size, 3);
			parallel_sum_rank0(R1, R0, temp, n, size, 1); 
			
			double beta = parallel_vector_vector_dot0(R1, R1, n, size, 4) / dot(R0, R0);
			
			// P1 = R1 + beta * P0; 
			parallel_scalar_mult0(temp, P0, beta, n, size, 3);
			parallel_sum_rank0(P1, R1, temp, n, size, 1);

			// for the next iteration 
			parallel_swap_rank0(R0, R1, n, size, 2);
			parallel_swap_rank0(X0, X1, n, size, 2); 
			parallel_swap_rank0(P0, P1, n, size, 2);
		}

 		for(int i = 1; i < size; ++i) {
 			MPI_Send(&False, 1, MPI_INT, i, 1234, MPI_COMM_WORLD);
 		}

		clock_t end = clock();
  		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		cout << "Time Elapsed: " << elapsed_secs << " sec" << endl;
 		cout << "Iterations: " << itr << endl;
 		cout << "Absolute Error - 1: " << dot(R0, R0)<< endl;
 		cout << "Absolute Error - 2: " << dot(b - A * X0, b - A * X0)<< endl;
 	} else {
 		int rank, operation, cont; 
		MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
 		while(true) {
			MPI_Recv(&cont, 1, MPI_INT, 0, 1234, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (!cont) break; 
			for(int i = 1; i <= 11; ++i) {
			 	MPI_Recv(&operation, 1, MPI_INT, 0, 12345, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (operation == 1) { 
					parallel_sum_rankI(rank); 
				} else if (operation == 2) {
					parallel_swap_rankI(rank);
				} else if (operation == 3) {
					parallel_scalar_multI(rank);
				} else if (operation == 4) {
					parallel_vector_vector_dotI(rank);
				}
			}
		}
	}

 	MPI_Finalize();
	return 0;
}

