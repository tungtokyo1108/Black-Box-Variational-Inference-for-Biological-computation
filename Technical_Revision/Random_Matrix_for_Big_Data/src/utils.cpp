/**
 *  Big Data Technology
 *
 *  Created on: July 16, 2019
 *  Data Scientist: Tung Dang
 */

#include <iostream>
#include <sys/time.h>
#include <algorithm>
#include <limits>
#include "utils.h"

namespace RandSVD {
    namespace Internal {
        
        /* *const double SVD_EPS = 0.0001f;
        
        void Util::sampleTwoGaussian(double& f1, double& f2) { 
            double v1 = (double)(std::rand() + 1.f) / ((double)RAND_MAX + 2.f);
            double v2 = (double)(std::rand() + 1.f) / ((double)RAND_MAX + 2.f);
            double len = std::sqrt(-2.f * std::log(v1));
            f1 = len * std::cos(2.f * M_PI * v2);
            f2 = len * std::sin(2.f * M_PI * v2);
        }

        void Util::sampleGaussianMat(Eigen::MatrixXd& mat) {
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

        void Util::processGramSchmidt(Eigen::MatrixXd& mat) {
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
        }*/
    }
}
