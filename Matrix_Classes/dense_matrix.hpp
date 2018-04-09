/*  
	Boost Competency Test (Gsoc '17)
	Author : Rishabh Arora
	Institute: IIIT Hyderabad
*/


#ifndef _MATRIX_HPP
#define _MATRIX_HPP

#include <vector>
#include <algorithm>
#include <cassert>

template <class T>
class Matrix {
public:
	size_t row_size, col_size;
	std::vector< std::vector < T > > mat;

// public:
	Matrix ();
	Matrix (const size_t r_s, const size_t c_s);
	Matrix (const Matrix<T> &M);
	Matrix (Matrix<T> &&M);

	Matrix<T> &operator = (Matrix<T> &&M);
	Matrix<T> &operator = (Matrix<T> &M);

	T getVal(const size_t row, const size_t col);
	T setVal(const size_t row, const size_t col, const T value);
	size_t getRowSize();
	size_t getColSize();

	template <class ElType>
	friend Matrix<ElType> operator - (const Matrix<ElType> &M1);
	template <class ElType>
	friend Matrix<ElType> operator + (const Matrix<ElType> &M1, const Matrix<ElType> &M2);
	template <class ElType>
	friend Matrix<ElType> operator - (const Matrix<ElType> &M1, const Matrix<ElType> &M2);
	template <class ElType>
	friend Matrix<ElType> operator * (const Matrix<ElType> &M1, const Matrix<ElType> &M2);
	template <class ElType>
	friend void operator += (Matrix<ElType> &M1, const Matrix<ElType> &M2);
	template <class ElType>
	friend void operator -= (Matrix<ElType> &M1, const Matrix<ElType> &M2);
	template <class ElType>
	friend void operator *= (Matrix<ElType> &M1, const Matrix<ElType> &M2);
	
	~Matrix();
};

/* constructor declarations */
/* empty matrix initialised */
template <class T>
Matrix<T>::Matrix () {
	row_size = col_size = 0;
}

/*
	set the dimensions of the matrix 
	all values are set to 0
*/
template <class T>
Matrix<T>::Matrix (const size_t r_s, const size_t c_s) {
	row_size = r_s;
	col_size = c_s;
	mat.resize(row_size);
	for(size_t i = 0; i < row_size; i++) {
		mat[i].resize(col_size);
	}
	for(size_t i = 0; i < row_size; ++i) {
		for(size_t  j = 0; j < col_size; ++j) {
			mat[i][j] = 0;
		}
	}
}

/* 
	copy constructor 
	deepcopy performed
	Additional time of O(row*col) used
*/
template <class T> 
Matrix<T>::Matrix (const Matrix<T> &M)  {
	row_size = M.row_size;
	col_size = M.col_size;
	mat.resize(row_size);
	for(size_t i = 0; i < row_size; i++) {
		mat[i].resize(col_size);
	}
	for(size_t i = 0; i < row_size; i++) {
		for(size_t j = 0; j < col_size; j++) {
			mat[i][j] = M.mat[i][j];
		}
	}	 
}

/*
	move constructor: moves the rhs value to the matrix
	O(1) time complexity
*/
template <class T> 
Matrix<T>::Matrix (Matrix<T> &&M)  {
	row_size = std::move(M.row_size);
	col_size = std::move(M.col_size);
	mat = std::move(M.mat);	 
}

/* destructor declaration */
template <class T> 
Matrix<T>::~Matrix () {
	mat.clear();
} 

/* Class functions */
template <class T>
T Matrix<T>::getVal(const size_t row, const size_t col) {
	return mat[row][col];
}  
template <class T>
T Matrix<T>::setVal(const size_t row, const size_t col, const T value) {
	return mat[row][col] = value;
}
template <class T>
size_t Matrix<T>::getRowSize() {
	return row_size;
}
template <class T>
size_t Matrix<T>::getColSize() {
	return col_size;
}


/* 
	Move Assignment operator
	O(1) time complexity
*/
template<class T>
Matrix<T>& Matrix<T>::operator = (Matrix<T> &&M) {
    row_size = std::move(M.row_size);
    col_size = std::move(M.col_size);
    mat = std::move(M.mat);
    return *this;
}

/*
	Copy Assignment operator
	performs deepcopy : O(row*col) time complexity
*/
template<class T>
Matrix<T>& Matrix<T>::operator = (Matrix<T> &M) {
    row_size = M.row_size;
    col_size = M.col_size;
    for(size_t i = 0; i < row_size; i++) {
    	for(size_t j = 0; j < row_size; j++) {
    		mat[i][j] = M.mat[i][j];
    	}
    }
    return *this;
}

/* friend functions*/
/* returns -1*Matrix */
template <class T>
Matrix<T> operator- (const Matrix<T> &M) {
	Matrix<T> res(M.row_size, M.col_size);
	for(size_t i = 0; i < M.row_size; i++) {
		for(size_t j = 0; j < M.col_size; j++) {
			res.mat[i][j] = -M.mat[i][j];
		}
	}
	return res;
}

/* 
	operator overloaded for matrix addition
	returns Matrix1 + Matrix2
*/
template <class T>
Matrix<T> operator+ (const Matrix<T> &M1, const Matrix<T> &M2) {
	/*to check wheter the matrices have same dimensions*/
	if(M1.row_size != M2.row_size || M1.col_size != M2.col_size) {
		return M1;
	}
	Matrix<T> res(M1.row_size, M1.col_size);
	for(size_t i = 0; i < M1.row_size; i++) {
		for(size_t j = 0; j < M1.col_size; j++) {
			res.mat[i][j] = M1.mat[i][j] + M2.mat[i][j];
		}
	}
	return res;
}

/* 
	operator overloaded for matrix subtraction
	returns Matrix1 - Matrix2
*/
template <class T>
Matrix<T> operator- (const Matrix<T> &M1, const Matrix<T> &M2) {
	/*to check wheter the matrices have same dimensions*/
	if(M1.row_size != M2.row_size || M1.col_size != M2.col_size) {
		return M1;
	}
	Matrix<T> res(M1.row_size, M1.col_size);
	for(size_t i = 0; i < M1.row_size; i++) {
		for(size_t j = 0; j < M1.col_size; j++) {
			res.mat[i][j] = M1.mat[i][j] - M2.mat[i][j];
		}
	}
	return res;
}

/* 
	operator overloaded for matrix multiplication
	returns Matrix1 * Matrix2
*/
template <class T>
Matrix<T> operator* (const Matrix<T> &M1, const Matrix<T> &M2) {
	/* to check if matrix multiplication is dimensionally possible */
	if( M1.col_size != M2.row_size) {
		return M1;
	}
	/* O(M1.row * M1.col * M2.col) time complexity */
	Matrix<T> res(M1.row_size, M2.col_size);
	for (size_t i = 0; i < M1.row_size; i++) {
	    for (size_t j = 0; j < M2.col_size; j++) {
		    res.mat[i][j] = 0;
            for (size_t k = 0; k < M1.col_size; k++)
                res.mat[i][j] += M1.mat[i][k] * M2.mat[k][j];
        }
    }
    return res;
}

/* 
	operator overloaded for matrix self addition
	performs Matrix1 = Matrix1 + Matrix2
*/
template <class T>
void operator+= (Matrix<T> &M1, const Matrix<T> &M2) {
	/* to check wheter the matrices have same dimensions */
	if(M1.row_size != M2.row_size || M1.col_size != M2.col_size) {
		return;
	}
	for(size_t i = 0; i < M1.row_size; i++) {
		for(size_t j = 0; j < M1.col_size; j++) {
			M1.mat[i][j] = M1.mat[i][j] + M2.mat[i][j];
		}
	}
	return;
}

/* 
	operator overloaded for matrix self subtraction
	performs Matrix1 = Matrix1 - Matrix2
*/
template <class T>
void operator-= (Matrix<T> &M1, const Matrix<T> &M2) {
	/* to check wheter the matrices have same dimensions */
	if(M1.row_size != M2.row_size || M1.col_size != M2.col_size) {
		return;
	}
	for(size_t i = 0; i < M1.row_size; i++) {
		for(size_t j = 0; j < M1.col_size; j++) {
			M1.mat[i][j] = M1.mat[i][j] - M2.mat[i][j];
		}
	}
	return;
}

/* 
	operator overloaded for matrix self multiplication
	performs Matrix1 = Matrix1 * Matrix2
*/
template <class T>
inline void operator*=(Matrix<T> &M1, const Matrix<T> &M2) {
	/* to check wheter the matrices have same dimensions */
	if(M1.col_size != M2.row_size) {
		return;
	}
	/* alocating O(row*col) additional memory to find the resultant */
	Matrix<T> res(M1.row_size, M2.col_size);
	for (size_t i = 0; i < M1.row_size; i++) {
	    for (size_t j = 0; j < M2.col_size; j++) {
		    res.mat[i][j] = 0;
            for (size_t k = 0; k < M1.col_size; k++)
                res.mat[i][j] += M1.mat[i][k] * M2.mat[k][j];
        }
    }
    size_t r = M1.row_size, c = M2.col_size;
    for(size_t i = 0; i < M1.row_size; i++) {
    	M1.mat[i].clear();
    }
    M1.mat.clear();
    M1.mat.resize(r);
    M1.row_size = r;
    M1.col_size = c;
    for(size_t i = 0; i < r; i++) {
    	M1.mat[i].resize(c);
    }
    for(size_t i = 0; i < M1.row_size; i++) {
    	for(size_t j = 0; j < M1.col_size; j++) {
    		M1.mat[i][j] = res.mat[i][j];
    	}
    }
	return;
}
/* returns Trace: sum of the body diagnol elements of a square matrix */
template <class T> 
T trace(Matrix<T> &M) {
	auto t = 0;
	for(size_t i = 0; i < std::min(M.getRowSize(), M.getColSize()) ; i++) {
		t += M.getVal(i, i);
	}
	return t;
}

template <class T>
Matrix<T> trans(Matrix<T>& X) {
	Matrix <T> ret(X.getColSize(), X.getRowSize());
	for(size_t i = 0; i < X.getRowSize(); ++i) {
		for(size_t j = 0; j < X.getColSize(); ++j) {
			ret.setVal(j, i, X.getVal(i, j));
		}
	}
	return ret; 
}

template <class T> 
T dot(Matrix<T> a, Matrix<T> b) {
	assert(a.getRowSize() == b.getRowSize() && a.getColSize() == b.getColSize());
	T ret = 0; 
	for(size_t i = 0; i < a.getRowSize(); ++i) {
		for(size_t j = 0; j < b.getColSize(); ++j) {
			ret += a.getVal(i, j) * b.getVal(i, j);
		}
	}
	return ret; 
}

template <class T>
Matrix <T> operator * (const T& val, Matrix <T>& A) {
	Matrix <T> ret(A.getRowSize(), A.getColSize()); 
	for(size_t i = 0; i < A.getRowSize(); ++i) {
		for(size_t j = 0; j < A.getColSize(); ++j) {
			ret.setVal(i, j, val * A.getVal(i, j));
		}
	}
	return ret; 
}

#endif


