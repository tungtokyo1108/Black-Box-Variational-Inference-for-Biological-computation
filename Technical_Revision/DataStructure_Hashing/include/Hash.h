#include <vector>
#include <map>
#include <iostream>
#include <bits/stdc++.h>

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
    struct NodeLinkList 
    {
        ElementType data;
        struct NodeLinkList* next;
    };

    template <typename ElementType>
    class HashAlgorithm 
    {
        public: 
            
            void GetVerticalOrder(Node<ElementType>* root, ElementType hd, std::map<ElementType, std::vector<ElementType>> &map);
            void printVerticalOrder(Node<ElementType> *root);

    };

    template <typename ElementType>
    class LinkList_HashMap
    {
        public:
        // LinkList and Hashmap 
            void pushLinkList(NodeLinkList<ElementType>** head_ref, ElementType new_data);
            void storeEle(NodeLinkList<ElementType>* head1, NodeLinkList<ElementType>* head2, 
                            std::unordered_map<ElementType, ElementType> &hashmap);
            NodeLinkList<ElementType> *getUnion(std::unordered_map<ElementType, ElementType> hashmap);
            NodeLinkList<ElementType> *getIntersection(std::unordered_map<ElementType, ElementType> hashmap);
            void printLinkList(NodeLinkList<ElementType>* node);
            void printUnionInterest(NodeLinkList<ElementType>* head1, NodeLinkList<ElementType>* head2);
    };

    template <typename ElementType>
    class FindElements
    {
        public:
            void sumPairs(ElementType arr[], ElementType size_arr, ElementType sum);
            void mostFreq(ElementType arr[], ElementType size_arr);
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

            /* if (hd < 0)
            {
                throw "Distance is negative";
            }*/

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
        ElementType hd = 0;
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

    ///////////////////////////// LinkList and HashMap //////////////////////////////////////////////
    
    template <typename ElementType>
    void LinkList_HashMap<ElementType>::pushLinkList(NodeLinkList<ElementType>** head_ref, ElementType new_data)
    {
        struct NodeLinkList<ElementType>* new_node = (struct NodeLinkList<ElementType>*) malloc(sizeof(struct NodeLinkList<ElementType>));
        new_node->data = new_data;
        new_node->next = (*head_ref);
        (*head_ref) = new_node;
    }

    template <typename ElementType>
    void LinkList_HashMap<ElementType>::storeEle(NodeLinkList<ElementType>* head1, NodeLinkList<ElementType>* head2, 
                                                    std::unordered_map<ElementType, ElementType>& hashmap)
    {
        struct NodeLinkList<ElementType> *ptr1 = head1;
        struct NodeLinkList<ElementType> *ptr2 = head2;
        while (ptr1 != NULL || ptr2 != NULL)
        {
            if (ptr1 != NULL)
            {
                hashmap[ptr1->data]++;
                ptr1 = ptr1->next;
            }

            if (ptr2 != NULL)
            {
                hashmap[ptr2->data]++;
                ptr2 = ptr2->next;
            }
        }
    }

    template <typename ElementType>
    NodeLinkList<ElementType>* LinkList_HashMap<ElementType>::getUnion(std::unordered_map<ElementType, ElementType> hashmap)
    {
        struct NodeLinkList<ElementType>* result = NULL;
        for (auto it=hashmap.begin(); it!=hashmap.end(); it++)
        {
            pushLinkList(&result, it->first); 
        }
        return result;
    }

    template <typename ElementType>
    NodeLinkList<ElementType>* LinkList_HashMap<ElementType>::getIntersection(std::unordered_map<ElementType, ElementType> hashmap)
    {
        struct NodeLinkList<ElementType>* result = NULL;
        for (auto it=hashmap.begin(); it!=hashmap.end(); it++)
        {
            if (it->second == 2)
            {
                pushLinkList(&result, it->first);
            }
        }
        return result;
    }

    template <typename ElementType>
    void LinkList_HashMap<ElementType>::printLinkList(NodeLinkList<ElementType>* node)
    {
        while (node != NULL)
        {
            std::cout << node->data << " -> ";
            node = node->next;
            if (node == NULL)
            {
                std::cout << "NULL" << std::endl;
            }
        }
    }

    template <typename ElementType>
    void LinkList_HashMap<ElementType>::printUnionInterest(NodeLinkList<ElementType>* head1, NodeLinkList<ElementType>* head2)
    {
        std::unordered_map<ElementType, ElementType> hashmap;
        storeEle(head1, head2, hashmap);
        NodeLinkList<ElementType>* interset_list = getIntersection(hashmap);
        NodeLinkList<ElementType>* union_list = getUnion(hashmap);

        std::cout << " The result of Intersetion : ";
        printLinkList(interset_list);
        std::cout << "\n" << std::endl;

        std::cout << " The result of union : ";
        printLinkList(union_list);
        std::cout << "\n" << std::endl;
    }

    ///////////////////////////// Find Elements in Hashing //////////////////////////////////////////////

    template <typename ElementType>
    void FindElements<ElementType>::sumPairs(ElementType arr[], ElementType size_arr, ElementType sum)
    {
        std::unordered_set<ElementType> set;
        for (int i=0; i < size_arr; i++)
        {
            ElementType temp = sum - arr[i];
            if (temp != 0 && set.find(temp) != set.end())
            {
                std::cout << "Pair with given sum " << sum << " is " << arr[i] << " + " << temp << std::endl;
            }
            set.insert(arr[i]);
        }
    }
}

#pragma endregion implementation 
