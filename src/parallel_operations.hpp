
#ifndef _PARALLEL_OPS
#define _PARALLEL_OPS 

#include <mpi.h>
#include "../Matrix_Classes/dense_matrix.hpp"
#include "../Matrix_Classes/sparse_matrix_coo.hpp"

void vector_sum_MASTER(Matrix<long double>& C, Matrix <long double>& A, Matrix <long double>& B, int n, int size, int op) {
	int operation = op, subDivide = max((n / (size - 1)), 1), start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i) {
		if (i == size - 1) end = n - 1;
		MPI_Send(&operation, 1, MPI_INT, i, 12345, MPI_COMM_WORLD);
		MPI_Send(&start, 1, MPI_INT, i, 1e5, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_INT, i, 2e5, MPI_COMM_WORLD);
		for(int j = start; j <= end; j++) {
			MPI_Send(&A.mat[j][0], 1, MPI_LONG_DOUBLE, i, j + 1e6, MPI_COMM_WORLD);
		}
		for(int j = start; j <= end;j++) {
			MPI_Send(&B.mat[j][0], 1, MPI_LONG_DOUBLE, i, j + 2e6, MPI_COMM_WORLD);
		}
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
	start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i) {
		if(i == size - 1) end = n - 1;
		for(int j = start; j <= end; j++) {
			MPI_Recv(&C.mat[j][0], 1, MPI_LONG_DOUBLE, i, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
}
void vector_sum(int rank) {
	int n, start, end;
	MPI_Recv(&start, 1, MPI_INT, 0, 1e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&end, 1, MPI_INT, 0, 2e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	Matrix < long double > A(end - start + 1, 1), B(end - start + 1, 1), C(end - start + 1, 1); 
	for(int i = 0; i < end - start + 1; i++) {
		MPI_Recv(&A.mat[i][0], 1, MPI_LONG_DOUBLE, 0, i + start + 1e6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	for(int i = 0; i < end - start + 1; i++) {
		MPI_Recv(&B.mat[i][0], 1, MPI_LONG_DOUBLE, 0, i + start + 2e6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	for(int i = 0; i < end - start + 1; ++i) {
		C.mat[i][0] = A.mat[i][0] + B.mat[i][0];
	}
	for(int i = 0; i < end - start + 1; ++i) {
		MPI_Send(&C.mat[i][0], 1, MPI_LONG_DOUBLE, 0, i + start, MPI_COMM_WORLD);
	}
	return;
}
void vector_swap_MASTER(Matrix <long double>& A, Matrix <long double>& B, int n, int size, int op) {
	int operation = op; 
	int subDivide = max((n / (size - 1)), 1);
	int start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i) {
		if (i == size - 1) end = n - 1;
		MPI_Send(&operation, 1, MPI_INT, i, 12345, MPI_COMM_WORLD);
		MPI_Send(&start, 1, MPI_INT, i, 1e5, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_INT, i, 2e5, MPI_COMM_WORLD);
		for(int j = start; j <= end; j++) {
			MPI_Send(&A.mat[j][0], 1, MPI_LONG_DOUBLE, i, j + 1e6, MPI_COMM_WORLD);
		}
		for(int j = start; j <= end;j++) {
			MPI_Send(&B.mat[j][0], 1, MPI_LONG_DOUBLE, i, j + 2e6, MPI_COMM_WORLD);
		}
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
	start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i){
		if(i == size - 1) end = n - 1;
		for(int j = start; j <= end; j++) {
			MPI_Recv(&A.mat[j][0], 1, MPI_LONG_DOUBLE, i, j + 2e7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		for(int j = start; j <= end; j++) {
			MPI_Recv(&B.mat[j][0], 1, MPI_LONG_DOUBLE, i, j + 1e7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
}
void vector_swap(int rank) {
	int n, start, end;
	MPI_Recv(&start, 1, MPI_INT, 0, 1e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&end, 1, MPI_INT, 0, 2e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	Matrix < long double > A(end - start + 1, 1), B(end - start + 1, 1);
	for(int i = 0; i <= end - start; i++) {
		MPI_Recv(&A.mat[i][0], 1, MPI_LONG_DOUBLE, 0, i + start + 1e6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	for(int i = 0; i <= end - start; i++) {
		MPI_Recv(&B.mat[i][0], 1, MPI_LONG_DOUBLE, 0, i + start + 2e6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	for(int i = 0; i <= end - start; ++i) {
		MPI_Send(&B.mat[i][0], 1, MPI_LONG_DOUBLE, 0, i + start + 2e7, MPI_COMM_WORLD);
	}
	for(int i = 0; i <= end - start; ++i) {
		MPI_Send(&A.mat[i][0], 1, MPI_LONG_DOUBLE, 0, i + start + 1e7, MPI_COMM_WORLD);
	}
	return;
}
void vector_scalar_mult_MASTER(Matrix<long double>& C, Matrix <long double>& A, long double alpha, int n, int size, int op) {
	int operation = op, subDivide = max((n / (size - 1)), 1), start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i) {
		if (i == size - 1) end = n - 1;
		MPI_Send(&operation, 1, MPI_INT, i, 12345, MPI_COMM_WORLD);
		MPI_Send(&start, 1, MPI_INT, i, 1e5, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_INT, i, 2e5, MPI_COMM_WORLD);
		MPI_Send(&alpha, 1, MPI_LONG_DOUBLE, i, 4e5, MPI_COMM_WORLD);
		for(int j = start; j <= end; j++) {
			MPI_Send(&A.mat[j][0], 1, MPI_LONG_DOUBLE, i, j + 1e6, MPI_COMM_WORLD);
		}
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
	start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i){
		if(i == size - 1) end = n - 1;
		for(int j = start; j <= end; j++) {
			MPI_Recv(&C.mat[j][0], 1, MPI_LONG_DOUBLE, i, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
}
void vector_scalar_mult(int rank) {
	int n, start, end;
	long double alpha;
	MPI_Recv(&start, 1, MPI_INT, 0, 1e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&end, 1, MPI_INT, 0, 2e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&alpha, 1, MPI_LONG_DOUBLE, 0, 4e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	Matrix < long double > A(end - start + 1, 1), C(end - start + 1, 1); 
	for(int i = 0; i <= end - start; i++) {
		MPI_Recv(&A.mat[i][0], 1, MPI_LONG_DOUBLE, 0, i + start + 1e6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	for(int i = 0; i <= end - start; ++i) {
		C.mat[i][0] = alpha * A.mat[i][0];
	}
	for(int i = 0; i <= end - start; ++i) {
		MPI_Send(&C.mat[i][0], 1, MPI_LONG_DOUBLE, 0, i + start, MPI_COMM_WORLD);
	}
	return;
}
long double vector_dot_MASTER(Matrix <long double>& A, Matrix<long double>& B, int n, int size, int op) {
	int operation = op, subDivide = max((n / (size - 1)), 1), start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i) {
		if (i == size - 1) end = n - 1;
		MPI_Send(&operation, 1, MPI_INT, i, 12345, MPI_COMM_WORLD);
		MPI_Send(&start, 1, MPI_INT, i, 1e5, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_INT, i, 2e5, MPI_COMM_WORLD);
		for(int j = start; j <= end; j++) {
			MPI_Send(&A.mat[j][0], 1, MPI_LONG_DOUBLE, i, j + 1e6, MPI_COMM_WORLD);
		}
		for(int j = start; j <= end; j++) {
			MPI_Send(&B.mat[j][0], 1, MPI_LONG_DOUBLE, i, j + 2e6, MPI_COMM_WORLD);
		}
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
	long double dot = 0;
	start = 0, end = start + subDivide;
	for(int i = 1; i < size; ++i){
		if(i == size - 1) end = n - 1;
		long double dot1; 
		MPI_Recv(&dot1, 1, MPI_LONG_DOUBLE, i, i * 10 + 123121, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		dot += dot1; 
		start = end + 1;
		end = min(n - 1, start + subDivide);
	}
	return dot; 
}
void vector_dot(int rank) {
	int n, start, end;
	long double alpha;
	MPI_Recv(&start, 1, MPI_INT, 0, 1e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&end, 1, MPI_INT, 0, 2e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	Matrix < long double > A(end - start + 1, 1), B(end - start + 1, 1); 
	for(int i = 0; i <= end - start; i++) {
		MPI_Recv(&A.mat[i][0], 1, MPI_LONG_DOUBLE, 0, i + start + 1e6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	for(int i = 0; i <= end - start; i++) {
		MPI_Recv(&B.mat[i][0], 1, MPI_LONG_DOUBLE, 0, i + start + 2e6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	long double dot = 0;
	for(int i = 0; i <= end - start; ++i) dot += A.mat[i][0] * B.mat[i][0];
	MPI_Send(&dot, 1, MPI_LONG_DOUBLE, 0, rank * 10 + 123121, MPI_COMM_WORLD);
	return;
}
void matrix_vector_mult_MASTER(Matrix <long double> &res, Matrix<int>& u, Matrix<long double>& b, int n, int dim, int size, int subDivide, int op) {
	int num_threads = size * size, Lr = 0, Lc = 0, Rr = subDivide - 1, Rc = subDivide - 1, zero = -1; 
	for(int j = 0; j < n; ++j) {
		res.mat[j][0] = 0.0;
	}
	for(int i = 1; i <= num_threads; ++i) {

		MPI_Send(&op, 1, MPI_INT, i, 12345, MPI_COMM_WORLD); 
		MPI_Send(&Lr, 1, MPI_INT, i, 1e5, MPI_COMM_WORLD); 
		MPI_Send(&Lc, 1, MPI_INT, i, 2e5, MPI_COMM_WORLD); 
		MPI_Send(&Rr, 1, MPI_INT, i, 3e5, MPI_COMM_WORLD); 
		MPI_Send(&Rc, 1, MPI_INT, i, 4e5, MPI_COMM_WORLD); 	
		MPI_Send(&n , 1, MPI_INT, i, 5e5, MPI_COMM_WORLD); 	
		
		for(int j = Lr - 1; j <= Rr + 1; ++j) {
			for(int k = Lc - 1; k <= Rc + 1; ++k) {
				if (j >= 0 && j < dim && k >= 0 && k < dim) { 
					MPI_Send(&u.mat[j][k], 1, MPI_INT, i, 1e4 * (j + 1) + (k + 1), MPI_COMM_WORLD);
				} else {
					MPI_Send(&zero, 1, MPI_INT, i, 1e4 * (j + 1) + (k + 1), MPI_COMM_WORLD);
				}
			}
		}

		for(int j = 0; j < n; ++j) {
			MPI_Send(&b.mat[j][0], 1, MPI_LONG_DOUBLE, i, 1e6 + j, MPI_COMM_WORLD);
		}
		Lc += subDivide; 
		Rc += subDivide; 
		if (Lc >= dim) {
			Lc = 0; 
			Rc = subDivide - 1;
			Lr += subDivide; 
			Rr += subDivide; 
		}
	}

	long double gather = 0;
	for(int i = 1; i <= num_threads; ++i) {
		for(int j = 0; j < n; ++j) {
			MPI_Recv(&gather, 1, MPI_LONG_DOUBLE, i, 1e8 + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
			res.mat[j][0] += gather; 
		}
	}

	return; 
}
void matrix_vector_mult(int rank) {
	int Lr, Lc, Rr, Rc, n; 
	
	MPI_Recv(&Lr, 1, MPI_INT, 0, 1e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&Lc, 1, MPI_INT, 0, 2e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&Rr, 1, MPI_INT, 0, 3e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&Rc, 1, MPI_INT, 0, 4e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&n , 1, MPI_INT, 0, 5e5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	Matrix < int > u(Rr - Lr + 3, Rc - Lc + 3); 
	Matrix < long double > b(n, 1);
	
	for(int i = 0; i < Rr - Lr + 3; ++i) {
		for(int j = 0; j < Rc - Lc + 3; ++j) {
			MPI_Recv(&u.mat[i][j], 1, MPI_INT, 0, 1e4 * (Lr + i) + (Lc + j), MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
		}
	} 
	for(int i = 0; i < n; ++i) {
		MPI_Recv(&b.mat[i][0], 1, MPI_LONG_DOUBLE, 0, 1e6 + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	Matrix < long double > res(n, 1);
	for(int i = 0; i < n; ++i) res.mat[i][0] = 0.0;
	for(int i = 1; i <= Rr - Lr + 1; ++i) {
		for(int j = 1; j <= Rc - Lc + 1; ++j) {
			int curr_pos = u.mat[i][j];
			res.mat[curr_pos][0] = 4.0 * b.mat[curr_pos][0];  
			if (u.mat[i - 1][j] >= 0) res.mat[curr_pos][0] -= b.mat[u.mat[i - 1][j]][0];
			if (u.mat[i + 1][j] >= 0) res.mat[curr_pos][0] -= b.mat[u.mat[i + 1][j]][0];
			if (u.mat[i][j - 1] >= 0) res.mat[curr_pos][0] -= b.mat[u.mat[i][j - 1]][0];
			if (u.mat[i][j + 1] >= 0) res.mat[curr_pos][0] -= b.mat[u.mat[i][j + 1]][0]; 	
		} 
	}
	for(int i = 0; i < n; ++i) {
		MPI_Send(&res.mat[i][0], 1, MPI_LONG_DOUBLE, 0, 1e8 + i, MPI_COMM_WORLD);
	}
	return; 
}


#endif 