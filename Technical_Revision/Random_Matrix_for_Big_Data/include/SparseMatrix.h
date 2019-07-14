/**
 *  Big Data Technology
 *
 *  Created on: July 14, 2019
 *  Data Scientist: Tung Dang
 */

#include <iostream>
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
        
    }
}
