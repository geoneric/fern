#pragma once
#include <functional>
#include <map>


namespace fern {
namespace python {

using BinaryAlgorithm = std::function<PyObject*(PyObject*, PyObject*)>;
template<
    typename BinaryAlgorithmKey>
using BinaryOperationMap = std::map<BinaryAlgorithmKey, BinaryAlgorithm>;

} // namespace python
} // namespace fern
