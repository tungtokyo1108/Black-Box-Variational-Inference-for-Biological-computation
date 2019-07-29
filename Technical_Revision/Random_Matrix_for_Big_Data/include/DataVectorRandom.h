/**
 *  Big Data Technology
 *
 *  Created on: July 25, 2019
 *  Data Scientist: Tung Dang
 */

#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <vector>
#include <Eigen/Dense>

namespace RandSVD {
    namespace data {

        enum class IterationPolity
        {
            all,
            skipZeros
        };

        class IDataVector 
        {

            public: 

                enum class Type 
                {
                    DoubleDataVector,
                    FloatDataVector,
                    ShortDataVector,
                    ByteDataVector,
                    SparseDoubleDataVector,
                    SparseFloatDataVector,
                    SparseShortDataVector,
                    SparseByteDataVector,
                    SparseBinaryDataVector,
                    AutoDataVector
                };

                virtual ~IDataVector() = default;

                virtual Type GetType() const = 0;
                virtual void AppendElement(size_t index, double value) = 0;
                virtual size_t PrefixLength() const = 0;
                virtual double Norm2Squared() const = 0;
                virtual double Dot(Eigen::VectorXd& vector) const = 0;
                virtual float Dot(Eigen::VectorXf& vector) const = 0;
                virtual void AddTo(Eigen::VectorXf& vector) const = 0;

                template <IterationPolity policy, typename TransformationType>
                void AddTransformedTo(Eigen::VectorXd& vector, TransformationType transformation) const;

                virtual std::vector<double> ToArray() const = 0;
                virtual std::vector<double> ToArray(size_t size) const = 0;

                template <typename ReturnType>
                ReturnType CopyAs() const;

                template <IterationPolity policy, typename ReturnType, typename TransformationType>
                ReturnType TransformAs(TransformationType transformation, size_t size) const;

                template <IterationPolity policy, typename ReturnType, typename TransformationType>
                ReturnType TransformAs(TransformationType transformation) const;

                virtual void Print(std::ostream& os) const = 0; 

            private: 
                template <typename ReturnType, typename GenericLambdaType>
                ReturnType InvokeWithThis(GenericLambdaType lambda) const;
        };

        struct IndexValue
        {
            size_t index;
            double value;
        };

        struct IIndexValueIterator
        {
            /* data */
        };

        // A helper definition used to define the IsDataVector concept
        template <typename T>
        using IsDataVector = typename std::enable_if_t<std::is_base_of<IDataVector, T>::value, bool>;

        // Helper type for concepts
        template <typename IteratorType>
        using IsIndexValueIterator = typename std::enable_if_t<std::is_base_of<IIndexValueIterator, IteratorType>::value, bool>;

        void operator+=(Eigen::VectorXd& vector, const IDataVector& dataVector);

        template <class DerivedType>
        class DataVectorBase : public IDataVector 
        {
            template <typename IndexValueIteratorType, IsIndexValueIterator<IndexValueIteratorType> Concept = true>
            void AppendElements(IndexValueIteratorType indexValueIterator);
            void AppendElements(std::initializer_list<IndexValue> vec);
            void AppendElements(const std::vector<double>& vec);
            void AppendElements(const std::vector<float>& vec);
            double Norm2Squared() const override;
            double Dot(Eigen::VectorXd& vector) const override;
            float Dot(Eigen::VectorXf& vector) const override;
            void AddTo(Eigen::VectorXd& vector) const override;

            template <IterationPolity policy, typename TransformationType>
            void AddTransformedTo(Eigen::VectorXd& vector, TransformationType transformation) const;

            auto GetValueIterator() {return static_cast<DerivedType*>(this)->GetValueIterator(PrefixLength()); }
            std::vector<double> ToArray() const override { return ToArray(PrefixLength()); }
            std::vector<double> ToArray(size_t size) const override;

            template <typename ReturnType>
            ReturnType CopyAs() const;

            template <IterationPolity policy, typename ReturnType, typename TransformationType>
            ReturnType TransformAs(TransformationType transformation, size_t size) const;

            template <IterationPolity policy, typename ReturnType, typename TransformationType>
            ReturnType TransformAs(TransformationType transformation) const;

            void Print(std::ostream& os) const override;  
        };

        // Wrapper for AddTransformedTo that hides the template specifier 
        template <typename DataVectorType, IterationPolity policy, typename TransformationType>
        static void AddTransformedTo(const DataVectorType& dataVector, Eigen::VectorXd& vector, 
                                        TransformationType transformation);
        
        // Wrapper for GetIterator that hides the template specifier
        template <typename DataVectorType, IterationPolity policy>
        static auto GetIterator(DataVectorType& vector);

        template <typename DataVectorType, IterationPolity policy>
        static auto GetIterator(const DataVectorType& vector);

        template <typename DataVectorType, IterationPolity policy>
        static auto GetIterator(DataVectorType& vector, size_t size);

        template <typename DataVectorType, IterationPolity policy>
        static auto GetIterator(const DataVectorType& vector, size_t size);

        template <typename DataVectorType, typename ReturnType>
        static ReturnType CopyAs(DataVectorType& vector);

        template <typename DataVectorType, typename ReturnType>
        static ReturnType CopyAs(const DataVectorType& vector);

        template <typename DataVectorType, IterationPolity policy, typename ReturnType, typename TransformationType>
        static ReturnType TransformAs(DataVectorType& vector, TransformationType transformation, size_t size);

        template <typename DataVectorType, IterationPolity policy, typename ReturnType, typename TransformationType>
        static ReturnType TransformAs(const DataVectorType& vector, TransformationType transformation, size_t size);

        template <typename DataVectorType, IterationPolity policy, typename ReturnType, typename TransformationType>
        static ReturnType TransformAs(DataVectorType& vector, TransformationType transformation);

        template <typename DataVectorType, IterationPolity policy, typename ReturnType, typename TransformationType>
        static ReturnType TransformAs(const DataVectorType& vector, TransformationType transformation);
    }
}

#pragma region implementation

namespace RandSVD {
    namespace data {
        template <typename ReturnType, typename GenericLambdaType>
        ReturnType IDataVector::InvokeWithThis(GenericLambdaType lambda) const 
        {
            auto type = GetType();
            switch (type)
            {
            case Type::DoubleDataVector:
                return lambda(static_cast<const DoubleDataVector*>(this));

            case Type::FloatDataVector:
                return lambda(static_cast<const FloatDataVector*>(this));

            case Type::ShortDataVector: 
                return lambda(static_cast<const ShortDataVector*>(this));

            case Type::ByteDataVector: 
                return lambda(static_cast<const ByteDataVector*>(this));

            case Type::SparseFloatDataVector: 
                return lambda(static_cast<const SparseFloatDataVector*>(this));

            case Type::SparseDoubleDataVector: 
                return lambda(static_cast<const SparseDoubleDataVector*>(this));

            case Type::SparseShortDataVector: 
                return lambda(static_cast<const SparseShortDataVector*>(this));

            case Type::SparseByteDataVector: 
                return lambda(static_cast<const SparseByteDataVector*>(this));

            case Type::SparseBinaryDataVector: 
                return lambda(static_cast<const SparseBinaryDataVector*>(this));
            
            default:
                break;
            }
        }

        template <IterationPolity policy, typename TransformationType>
        void IDataVector::AddTransformedTo(Eigen::VectorXd& vector, TransformationType transformation) const
        {
            InvokeWithThis<void>([vector, transformation](const auto* pThis) {
                pThis->template AddTransformedTo<policy>(vector, transformation);
            });
        }

        template <typename ReturnType>
        ReturnType IDataVector::CopyAs() const 
        {
            return InvokeWithThis<ReturnType>([](const auto* pThis) {
                return ReturnType(pThis->template GetIterator<IterationPolity::skipZeros>());
            });
        }

        
    }
}

#pragma endregion implementation
