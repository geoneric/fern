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

} // namespace operation
} // namespace ranally

