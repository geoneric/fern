#include "ranally/core/stack.h"


namespace ranally {

boost::any const& Stack::top() const
{
  return _stack.top();
}


void Stack::pop()
{
    _stack.pop();
}


size_t Stack::size() const
{
    return _stack.size();
}


bool Stack::empty() const
{
    return _stack.empty();
}

} // namespace ranally
