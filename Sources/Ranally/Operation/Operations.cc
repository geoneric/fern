#include "Ranally/Operation/Operations.h"



namespace ranally {
namespace operation {

Operations::~Operations()
{
}



bool Operations::empty() const
{
  return _operations.empty();
}



size_t Operations::size() const
{
  return _operations.size();
}



bool Operations::hasOperation(
  String const& name) const
{
  return _operations.find(name) != _operations.end();
}



OperationPtr const& Operations::operation(
  String const& name) const
{
  std::map<String, OperationPtr>::const_iterator it =
    _operations.find(name);
  assert(it != _operations.end());
  return (*it).second;
}

} // namespace operation
} // namespace ranally

