/**
 *  Big Data Technology
 *
 *  Created on: July 15, 2019
 *  Data Scientist: Tung Dang
 */

#include "SparseRandomMatrix.h"

namespace Eigen 
{
    namespace internal
    {
        template <typename ElementType>
        CommonMatrixBase<ElementType>::CommonMatrixBase(const ElementType* pData, size_t numRows, size_t numColumns, size_t increment):
            _pData(pData),
            _numRows(numRows),
            _numColumns(numColumns),
            _increment(increment)
        {}

        template <typename ElementType>
        MatrixBase<ElementType, MatrixLayout::columnMajor>::MatrixBase(const ElementType* pData, size_t numRows, size_t numColumns):
            CommonMatrixBase<ElementType>(pData, numRows, numColumns, numRows)
        {}

        template <typename ElementType>
        MatrixBase<ElementType, MatrixLayout::columnMajor>::MatrixBase(const ElementType* pData, size_t numRows, size_t numColumns, size_t increment):
            CommonMatrixBase<ElementType>(pData, numRows, numColumns, increment)
        {}

        template <typename ElementType>
        MatrixBase<ElementType, MatrixLayout::rowMajor>::MatrixBase(const ElementType* pData, size_t numRows, size_t numColumns):
            CommonMatrixBase<ElementType>(pData, numRows, numColumns, numColumns)
        {}

        template <typename ElementType>
        MatrixBase<ElementType, MatrixLayout::rowMajor>::MatrixBase(const ElementType* pData, size_t numRows, size_t numColumns, size_t increment):
            CommonMatrixBase<ElementType>(pData, numRows, numColumns, increment)
        {}

        template <typename ElementType, MatrixLayout layout>
        ConstMatrixReference<ElementType, layout>::ConstMatrixReference(const ElementType* pData, size_t numRows, size_t numColumns):
            MatrixBase<ElementType, layout>(pData, numRows, numColumns)
        {}

        template <typename ElementType, MatrixLayout layout>
        ConstMatrixReference<ElementType, layout>::ConstMatrixReference(const ElementType* pData, size_t numRows, size_t numColumns, size_t increment):
            MatrixBase<ElementType, layout>(pData, numRows, numColumns, increment)
        {}

        template <typename ElementType, MatrixLayout layout>
        MatrixReference<ElementType, layout>::MatrixReference(const ElementType* pData, size_t numRows, size_t numColumns):
            ConstMatrixReference<ElementType, layout>(pData, numRows, numColumns)
        {}

        template <typename ElementType, MatrixLayout layout>
        MatrixReference<ElementType, layout>::MatrixReference(const ElementType* pData, size_t numRows, size_t numColumns, size_t increment):
            ConstMatrixReference<ElementType, layout>(pData, numRows, numColumns, increment)
        {}

        template <typename ElementType, MatrixLayout layout>
        Matrix<ElementType, layout>::Matrix(size_t numRows, size_t numColumns) : 
            MatrixReference<ElementType, layout>(nullptr, numRows, numColumns),
            _data(numRows * numColumns)
        {
            this->_pData = _data.data();
        }

        template <typename ElementType, MatrixLayout layout>
        Matrix<ElementType, layout>::Matrix(std::initializer_list<std::initializer_list<ElementType>> list) : 
            MatrixReference<ElementType, layout>(nullptr, list.size(), list.begin()->size()),
            _data(list.size() * list.begin()->size())
        {
            this->_pData = _data.data();
            auto numColumns = list.begin()->size();
            size_t i = 0;
            for (auto rowIter = list.begin(); rowIter < list.end(); ++rowIter)
            {
                size_t j = 0;
                for (auto elementIter = rowIter->begin(); elementIter < rowIter->end(); ++elementIter)
                {
                    (*this)(i, j) = *elementIter;
                    ++j;
                }
                ++i;
            }
        }

        template <typename ElementType, MatrixLayout layout>
        Matrix<ElementType, layout>::Matrix(size_t numRows, size_t numColumns, const std::vector<ElementType>& data) : 
            MatrixReference<ElementType, layout>(nullptr, numRows, numColumns),
            _data(data)
        {
            this->_pData = _data.data();
        }

        template <typename ElementType, MatrixLayout layout>
        Matrix<ElementType, layout>::Matrix(size_t numRows, size_t numColumns, std::vector<ElementType>&& data) :
            MatrixReference<ElementType, layout>(nullptr, numRows, numColumns),
            _data(std::move(data))
        {
            this->_pData = _data.data();
        }

        template <typename ElementType, MatrixLayout layout>
        Matrix<ElementType, layout>::Matrix(Matrix<ElementType, layout>&& other) : 
            MatrixReference<ElementType, layout>(nullptr, other.NumRows(), other.NumColumns()),
            _data(std::move(other._data))
        {
            this->_pData = _data.data();
        }

        template <typename ElementType, MatrixLayout layout>
        Matrix<ElementType, layout>::Matrix(const Matrix<ElementType, layout>& other) : 
            MatrixReference<ElementType, layout>(nullptr, other.NumRows(), other.NumColumns()),
            _data(other._data)
        {
            this->_pData = _data.data();
        }

        template <typename ElementType, MatrixLayout layout>
        Matrix<ElementType, layout>::Matrix(ConstMatrixReference<ElementType, layout>& other) : 
            MatrixReference<ElementType, layout>(nullptr, other.NumRows(), other.NumColumns()),
            _data(other.NumRows() * other.NumColumns())
        {
            this->_pData = _data.data();
            for (size_t i=0; i < this->NumRows(); ++i)
            {
                for (size_t j=0; j < this->NumColumns(); ++j)
                {
                    (*this)(i,j) = other(i,j);
                }
            }
        }
    }
}
