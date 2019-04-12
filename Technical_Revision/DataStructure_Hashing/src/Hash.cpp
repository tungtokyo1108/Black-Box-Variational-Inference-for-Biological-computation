#include "Hash.h"

using namespace hash;

void TestPrintBinaryTreeHash()
{
    Node<double> *root = newNode(1.2); 
    root->left = newNode(2.3); 
    root->right = newNode(3.1); 
    root->left->left = newNode(4.2); 
    root->left->right = newNode(5.3); 
    root->right->left = newNode(6.5); 
    root->right->right = newNode(7.6); 
    root->right->left->right = newNode(8.6); 
    root->right->right->right = newNode(9.8); 

    HashAlgorithm<double> hs;
    hs.printVerticalOrder(root);
}

int main (int argc, char const* argv[])
{
    try
    {
        TestPrintBinaryTreeHash();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        throw;
    }
    
    return 0;
}
