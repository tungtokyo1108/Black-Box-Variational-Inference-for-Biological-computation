/**
 *  Big Data Technology
 *
 *  Created on: July 14, 2019
 *  Data Scientist: Tung Dang
 */
#pragma once

#include <iostream>
#include <cstddef>
#include <cmath>
#include <vector>
#include <limits>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <string>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers> 
#include <unsupported/Eigen/IterativeSolvers> 

class MatrixReplacement;
using Eigen::SparseMatrix;

namespace Eigen {
    namespace internal {
        template <>
        struct traits <MatrixReplacement> : public Eigen::internal::traits<Eigen::SparseMatrix<double>>
        {};
    }
}

class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement> {
    public:
        typedef double Scalar;
        typedef double RealScalar;
        typedef int StorageIndex;

        enum {
            ColsAtCompileTime = Eigen::Dynamic,
            MaxColsAtCompileTime = Eigen::Dynamic,
            IsRowMajor = false
        };

        Index rows() const {return mp_mat->rows(); }
        Index cols() const {return mp_mat->cols(); }

        template <typename Rhs>
        Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
            return Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
        }

        MatrixReplacement() : mp_mat(0) {}

        void attachMyMatrix(const SparseMatrix<double> &mat) {
            mp_mat = &mat;
        }

        const SparseMatrix<double> my_matrix() const {return *mp_mat;}

    private:
        const SparseMatrix<double> *mp_mat;
};

namespace Eigen {
    namespace internal {

        template <typename Rhs>
        struct generic_product_impl<MatrixReplacement, Rhs, SparseShape, DenseShape, GemvProduct>
        : generic_product_impl_base<MatrixReplacement, Rhs, generic_product_impl<MatrixReplacement, Rhs>>
        {
            typedef typename Product<MatrixReplacement,Rhs>::Scalar Scalar;
            template <typename Dest>
            static void scaleAndAddTo(Dest& dst, const MatrixReplacement& lhs, const Rhs& rhs, const Scalar& alpha)
            {
                assert(alpha==Scalar(1) && "scaling is not implemented");
                EIGEN_ONLY_USED_FOR_DEBUG(alpha);

                for (Index i=0; i<lhs.cols(); ++i)
                {
                    dst += rhs(i) * lhs.my_matrix().col(i);
                }
            }
        };

        enum class MatrixLayout
        {
            columnMajor,
            rowMajor
        };

        template <typename ElementType>
        class CommonMatrixBase
        {
            public:
                size_t NumRows() const {return _numRows;}
                size_t NumColumns() const {return _numColumns;}
                size_t Size() const {return NumRows() * NumColumns();}
                size_t GetIncrement() const {return _increment;}

            protected: 
                CommonMatrixBase(const ElementType* pData, size_t numRows, size_t numColumns, size_t increment);
                const ElementType* _pData;
                size_t _numRows;
                size_t _numColumns;
                size_t _increment;
        };

        template <typename ElementType, MatrixLayout layout>
        class MatrixBase;

        template <typename ElementType>
        class MatrixBase<ElementType, MatrixLayout::columnMajor> : public CommonMatrixBase<ElementType>
        {
            public:
                size_t GetMinorSize() const {return this->NumColumns();}
                size_t GetMajorSize() const {return this->NumRows();}
                size_t GetRowIncrement() const {return 1;}
                size_t GetColumnIncrement() const {return this->GetIncrement();}
            
            protected:
                MatrixBase(const ElementType* pData, size_t numRows, size_t numColumns);
                MatrixBase(const ElementType* pData, size_t numRows, size_t numColumns, size_t increment);
        };

        template <typename ElementType>
        class MatrixBase<ElementType, MatrixLayout::rowMajor> : public CommonMatrixBase<ElementType>
        {
            public: 
                size_t GetMinorSize() const {return this->NumRows();}
                size_t GetMajorSize() const {return this->NumColumns();}
                size_t GetColumnIncrement() const {return 1;}
                size_t GetRowIncrement() const {return this->GetIncrement();}
            
            protected: 
                MatrixBase(const ElementType* pData, size_t numRows, size_t numColumns);
                MatrixBase(const ElementType* pData, size_t numRows, size_t numColumns, size_t increment);
        };

        template <typename ElementType, MatrixLayout layout>
        class ConstMatrixReference : public MatrixBase<ElementType, layout>
        {
            public:
                ConstMatrixReference(const ElementType* pData, size_t numRows, size_t numColumns);
                ConstMatrixReference(const ElementType* pData, size_t numRows, size_t numColumns, size_t increment);
        };

        template <typename ElementType, MatrixLayout layout>
        class MatrixReference : public ConstMatrixReference<ElementType, layout>
        {
            public: 
                MatrixReference(const ElementType* pData, size_t numRows, size_t numColumns);
                MatrixReference(const ElementType* pData, size_t numRows, size_t numColumns, size_t increment);
        };

        template <typename ElementType, MatrixLayout layout>
        class Matrix : public MatrixReference<ElementType, layout>
        {
            public:
                Matrix(size_t numRows, size_t numColumns);
                Matrix(std::initializer_list<std::initializer_list<ElementType>> list);
                Matrix(size_t numRows, size_t numColumns, const std::vector<ElementType>& data);
                Matrix(size_t numRows, size_t numColumns, std::vector<ElementType>&& data);
                Matrix(Matrix<ElementType, layout>&& other);
                Matrix(const Matrix<ElementType, layout>& other);
                Matrix(ConstMatrixReference<ElementType, layout>& other);
            private: 
                std::vector<ElementType> _data;
        };
    }
}
