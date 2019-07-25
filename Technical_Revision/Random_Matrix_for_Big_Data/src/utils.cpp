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
        
        void TestsampleGauss()
        {
            Eigen::MatrixXd m1(10,10);

            RandSVD::Internal::Util<Eigen::MatrixXd> randmat;
            randmat.sampleGaussianMat(m1);
            std::cout << "Here is the Gaussian matrix with double type: \n" << m1 << std::endl;
            // randmat.processGramSchmidt(m1);
            randmat.modifiedGramSchmit(m1);
            std::cout << "Here is the GramSchmidt process with double type: \n" << m1 << "\n" << std::endl;

            Eigen::MatrixXf m2(10,10);
            RandSVD::Internal::Util<Eigen::MatrixXf> randmat_f;
            randmat_f.sampleGaussianMat(m2);
            std::cout << "Here is the Gaussian matrix with float type: \n" << m2 << std::endl;
            randmat_f.modifiedGramSchmit(m2);
            std::cout << "Here is the GramSchmidt process with float type: \n" << m2 << std::endl;
        }
    }
}
