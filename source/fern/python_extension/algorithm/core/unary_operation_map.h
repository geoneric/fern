#pragma once
#include <functional>
#include <map>


namespace fern {
namespace python {

using UnaryAlgorithm = std::function<PyObject*(PyObject*)>;
template<
    typename UnaryAlgorithmKey>
using UnaryOperationMap = std::map<UnaryAlgorithmKey, UnaryAlgorithm>;

} // namespace python
} // namespace fern
