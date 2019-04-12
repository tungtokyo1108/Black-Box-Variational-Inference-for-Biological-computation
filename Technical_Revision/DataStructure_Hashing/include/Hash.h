#include <vector>
#include <map>
#include <iostream>

namespace hash 
{
    template <typename ElementType>
    struct Node 
    {
        ElementType key;
        Node *left;
        Node *right;
    };

    template <typename ElementType>
    struct Node<ElementType>* newNode(ElementType key)
    {
        struct Node<ElementType>* root = new Node<ElementType>;
        root->key = key;
        root->left = NULL;
        root->right = NULL;
        return root;
    };

    template <typename ElementType>
    class HashAlgorithm 
    {
        public: 
            
            void GetVerticalOrder(Node<ElementType>* root, ElementType hd, std::map<ElementType, std::vector<ElementType>> &map);
            void printVerticalOrder(Node<ElementType> *root);
    };
}

#pragma region implementation 

namespace hash
{
    template <typename ElementType>
    void HashAlgorithm<ElementType>::GetVerticalOrder(Node<ElementType>* root, ElementType hd, std::map<ElementType, std::vector<ElementType>> &map)
    {
        try
        {
            if (root == NULL)
            {
                throw "Not input data for Binary Tree";
            }

            if (hd < 0)
            {
                throw "Distance is negative";
            }

            map[hd].push_back(root->key);
            GetVerticalOrder(root->left, hd-1, map);
            GetVerticalOrder(root->right, hd+1, map);
        }
        catch(const char* msg)
        {
            std::cout << "An exception occurred. Exception is: " << msg << std::endl;
        }
        
    }

    template <typename ElementType>
    void HashAlgorithm<ElementType>::printVerticalOrder(Node<ElementType>* root)
    {
        
        std::map<ElementType, std::vector<ElementType>> map;
        ElementType hd = -1;
        GetVerticalOrder(root, hd, map);
        std::map<double, std::vector<double>>::iterator it;
        for (it=map.begin(); it!=map.end(); it++)
        {
            for (int i=0; i < it->second.size(); i++)
            {
                std::cout << it->second[i] << " ";
            }
            std::cout << std::endl;
        }
    }
}

#pragma endregion implementation 
