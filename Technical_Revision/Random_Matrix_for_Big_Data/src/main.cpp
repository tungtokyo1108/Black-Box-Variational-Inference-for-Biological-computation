/**
 *  Big Data Technology
 *
 *  Created on: July 14, 2019
 *  Data Scientist: Tung Dang
 */

#include "SparseRandomMatrix.h"
#include "SparseRandomMatrix.cpp"
#include "utils.h"
// #include "utils.cpp"
#include <time.h>

void Test_Matrix_free_solver() {
    std::cout << "****************************************************************" << std::endl;
    std::cout << "Test Matrix Free Solver" << std::endl;
    int n = 10;
    Eigen::SparseMatrix<double> S = Eigen::MatrixXd::Random(n,n).sparseView(0.5,1);
    S = S.transpose() * S;

    MatrixReplacement A;
    A.attachMyMatrix(S);

    Eigen::VectorXd b(n), x;
    b.setRandom();

    {
        Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
        cg.compute(A);
        x = cg.solve(b);
        std::cout << "CG:       #iteration:   " << cg.iterations() << ",   estimated error: " << cg.error() << std::endl;
    }
}

void Test_L2norm() {
    std::cout << "****************************************************************" << std::endl;
    std::cout << "Checking is a solution really exists" << std::endl;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(100,100);
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(100,50);
    Eigen::MatrixXd x = A.fullPivHouseholderQr().solve(b);
    double relative_error = (A*x - b).norm() / b.norm();
    std::cout << "The relative error is: " << relative_error << std::endl;
}

void Test_SVD() {
    std::cout << "****************************************************************" << std::endl;
    std::cout << "Testing SVD algorithm" << std::endl;
    int rows;
    std::cout << "Please input the number of rows: ";
    std::cin >> rows;
    std::cout << "\n";
    int cols;
    std::cout << "Please input the number of columns: ";
    std::cin >> cols;
    std::cout << "\n";
    int choise;
    std::cout << "There are two options for computing SVD. \n";
    std::cout << "- 0 is option for BDCSVD algorithm \n";
    std::cout << "- 1 is option for JacobiSVD algorithm \n";
    std::cout << "Your option: ";
    std::cin >> choise;
    std::cout << "\n";

    clock_t tStart = clock();
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(rows,cols);
    // std::cout << "Here is the matrix A: \n" << A << std::endl;
    Eigen::VectorXf b = Eigen::VectorXf::Random(rows);
    // std::cout << "Here is the right hand site b: \n" << b << std::endl;
    if (choise == 0)
    {
        Eigen::BDCSVD<Eigen::MatrixXf> svd_bd(A, Eigen::ComputeThinU || Eigen::ComputeThinV);
        std::cout << "Its singular values are :" << std::endl << svd_bd.singularValues() << std::endl;
    }
    else 
    {
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        std::cout << "Its singular values are :" << std::endl << svd.singularValues() << std::endl;
        // std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
        // std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;
    }
    std::cout << "\n";
    std::cout << "Time Computation: " << (double)(clock() - tStart)/CLOCKS_PER_SEC << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////
template <typename ElementType, Eigen::internal::MatrixLayout layout>
void TestMatrixNumRows()
{
    Eigen::internal::Matrix<ElementType, layout> M {
        {1,0,4,0},
        {0,0,0,0},
        {0,0,0,7}
    };

    std::cout << M.NumRows();
}

template <typename ElementType, Eigen::internal::MatrixLayout layout>
void RunLayoutMatrixTests()
{
    TestMatrixNumRows<ElementType, layout>();
}

template <typename ElementType>
void RunMatrixTest()
{
    RunLayoutMatrixTests<ElementType, Eigen::internal::MatrixLayout::columnMajor>();
    RunLayoutMatrixTests<ElementType, Eigen::internal::MatrixLayout::rowMajor>();
}
/////////////////////////////////////////////////////////////////////////////////////

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

int main() {
    try
    {
        // Test_Matrix_free_solver();
        // Test_L2norm();
        // Test_SVD();
        TestsampleGauss();
        // RunMatrixTest<int>();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        throw;
    }
    
    return 0;
}
