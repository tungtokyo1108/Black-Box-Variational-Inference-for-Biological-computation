/**
 *  Big Data Technology
 *
 *  Created on: July 15, 2019
 *  Data Scientist: Tung Dang
 */

#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

namespace RandSVD {
    namespace Internal {
        /* Eigen::IOFormat CleanFmt(3, 0, ",", "\n", "", "");

        enum class SubspaceIterationConditioner {
            NoConditioner,
            // Full pivoted LU decomposition, fast, acceptable numerical stability
            LuConditioner,
            // Modified Gram-Schmidt orthonormalization, slow, better numerical stability 
            MgsConditioner,
            // QR decomposition, slowest, best numerical stability 
            QrConditioner,
        };

        void print_eigen(const std::string name, const Eigen::MatrixXd& mat) {
            if (mat.cols() == 1)
            {
                std::cout << name << ":" << mat.transpose().format(CleanFmt) << std::endl;
            }
            else 
            {
                std::cout << name << ":\n" << mat.format(CleanFmt) << std::endl;
            }
        }*/

        typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SMatrixXd;

        template <typename MatrixType>
        class Util 
        {
            public:
                SMatrixXd *fill_sparse_matrix(SMatrixXd *A, int nnz, int *I, int *J, double *V);
                void sampleGaussianMat(MatrixType& mat);
                void processGramSchmidt(MatrixType& mat);
                void modifiedGramSchmit(MatrixType& mat);
                double getSec();

            protected:
                void sampleTwoGaussian(double& f1, double& f2);
        };

    } // namespace Internal
} // namespace RandSVD

#pragma region implementation

namespace RandSVD {
    namespace Internal {

        const double SVD_EPS = 0.0001f;

        template <typename MatrixType>
        void Util<MatrixType>::sampleTwoGaussian(double& f1, double& f2)
        {
            double v1 = (double)(std::rand() + 1.f) / ((double)RAND_MAX + 2.f);
            double v2 = (double)(std::rand() + 1.f) / ((double)RAND_MAX + 2.f);
            double len = std::sqrt(-2.f * std::log(v1));
            f1 = len * std::cos(2.f * M_PI * v2);
            f2 = len * std::sin(2.f * M_PI * v2);
        }

        template <typename MatrixType>
        void Util<MatrixType>::sampleGaussianMat(MatrixType& mat) 
        {
            for (int i=0; i < mat.rows(); ++i)
            {
                int j=0;
                for (; j+1 < mat.cols(); j += 2) 
                {
                    double f1, f2;
                    sampleTwoGaussian(f1,f2);
                    mat(i,j) = f1;
                    mat(i,j+1) = f2;
                }
                for (; j < mat.cols(); j++)
                {
                    double f1, f2;
                    sampleTwoGaussian(f1,f2);
                    mat(i,j) = f1;
                }
            }
        }

        template <typename MatrixType>
        void Util<MatrixType>::processGramSchmidt(MatrixType& mat)
        {
            for (int i=0; i < mat.cols(); ++i)
            {
                for (int j=0; j < i; ++j)
                {
                    double r = mat.col(i).dot(mat.col(j));
                    mat.col(i) -= r * mat.col(j);
                }
                double norm = mat.col(i).norm();
                if (norm < SVD_EPS)
                {
                    for (int k=i; k < mat.cols(); ++k)
                    {
                        mat.col(k).setZero();
                    }
                    return;
                }
                mat.col(i) *= (1.f / norm);
            }
        }

        template <typename MatrixType>
        void Util<MatrixType>::modifiedGramSchmit(MatrixType& mat) 
        {
            using RealType = typename Eigen::NumTraits<typename MatrixType::Scalar>::Real;

            RealType largestNormSeen = 0;
            const RealType tol = 100 * std::numeric_limits<RealType>::epsilon();

            assert(mat.cols() <= mat.rows());

            Eigen::Index currCol;

            for (currCol = 0; currCol < mat.cols(); ++currCol)
            {
                for (Eigen::Index prevCol = 0; prevCol < currCol; ++prevCol)
                {
                    mat.col(currCol) -= mat.col(prevCol).dot(mat.col(currCol)) * mat.col(prevCol);
                }

                const auto currColNorm = mat.col(currCol).norm();
                if (currColNorm < tol * largestNormSeen)
                {
                    mat.col(currCol).setZero();
                }
                else
                {
                    mat.col(currCol) /= currColNorm;
                    largestNormSeen = std::max(largestNormSeen, currColNorm);
                }
            }
        }
    }
}

#pragma endregion implementation