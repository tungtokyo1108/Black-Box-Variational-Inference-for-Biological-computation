#include "Hash.h"
#include <iostream>

using namespace hash;

void TestPrintBinaryTreeHash()
{
    std::cout << " Algorithm for printing Binary Tree " << std::endl;

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

void TestUnionIntersetLL()
{
    std::cout << "\n" << std::endl;
    std::cout << "**************************************************";
    std::cout << "\n" << std::endl;
    std::cout << "Algorithm for Union and Intersection LL \n" << std::endl;
    std::cout << "- Case 1: Integer Number \n" << std::endl;
    NodeLinkList<int> *head1 = NULL;
    NodeLinkList<int> *head2 = NULL;

    LinkList_HashMap<int> LLHM;
    LLHM.pushLinkList(&head1, 1);
    LLHM.pushLinkList(&head1, 2);
    LLHM.pushLinkList(&head1, 3);
    LLHM.pushLinkList(&head1, 4);
    LLHM.pushLinkList(&head1, 5);

    LLHM.pushLinkList(&head2, 1);
    LLHM.pushLinkList(&head2, 3);
    LLHM.pushLinkList(&head2, 5);
    LLHM.pushLinkList(&head2, 6);

    std::cout << " Link List 1 is : ";
    LLHM.printLinkList(head1);
    std::cout << "\n" << std::endl;

    std::cout << " Link List 2 is : ";
    LLHM.printLinkList(head2);
    std::cout << "\n" << std::endl;

    LLHM.printUnionInterest(head1, head2);

    /***************************************************************/

    std::cout << "- Case 2: Double Number \n" << std::endl;
    NodeLinkList<double> *head3 = NULL;
    NodeLinkList<double> *head4 = NULL;

    LinkList_HashMap<double> LLHM_d;
    LLHM_d.pushLinkList(&head3, 1.1);
    LLHM_d.pushLinkList(&head3, 2.2);
    LLHM_d.pushLinkList(&head3, 3.3);
    LLHM_d.pushLinkList(&head3, 4.4);
    LLHM_d.pushLinkList(&head3, 5.5);

    LLHM_d.pushLinkList(&head4, 1.1);
    LLHM_d.pushLinkList(&head4, 3.3);
    LLHM_d.pushLinkList(&head4, 5.5);
    LLHM_d.pushLinkList(&head4, 6.6);

    std::cout << " Link List 3 is : ";
    LLHM_d.printLinkList(head3);
    std::cout << "\n" << std::endl;

    std::cout << " Link List 4 is : ";
    LLHM_d.printLinkList(head4);
    std::cout << "\n" << std::endl;

    LLHM_d.printUnionInterest(head3, head4);

    /***************************************************************/

    std::cout << "- Case 3: Character \n" << std::endl;
    NodeLinkList<char> *head5 = NULL;
    NodeLinkList<char> *head6 = NULL;

    LinkList_HashMap<char> LLHM_c;
    LLHM_c.pushLinkList(&head5, 'A');
    LLHM_c.pushLinkList(&head5, 'B');
    LLHM_c.pushLinkList(&head5, 'C');
    LLHM_c.pushLinkList(&head5, 'D');
    LLHM_c.pushLinkList(&head5, 'E');

    LLHM_c.pushLinkList(&head6, 'E');
    LLHM_c.pushLinkList(&head6, 'F');
    LLHM_c.pushLinkList(&head6, 'C');
    LLHM_c.pushLinkList(&head6, 'N');

    std::cout << " Link List 5 is : ";
    LLHM_c.printLinkList(head5);
    std::cout << "\n" << std::endl;

    std::cout << " Link List 6 is : ";
    LLHM_c.printLinkList(head6);
    std::cout << "\n" << std::endl;

    LLHM_c.printUnionInterest(head5, head6);


    std::cout << "**************************************************";
    std::cout << "\n" << std::endl;
}

int main (int argc, char const* argv[])
{
    try
    {
        TestPrintBinaryTreeHash();
        TestUnionIntersetLL();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        throw;
    }
    
    return 0;
}
