// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/stack.h"
#include <cassert>


namespace fern {

//! Return the top value from the stack.
/*!
*/
boost::any const& Stack::top() const
{
    return std::stack<boost::any>::top();
}


//! Pop the top value from the stack.
/*!
  \warning   The stack must not be empty.
*/
void Stack::pop()
{
    assert(!empty());
    std::stack<boost::any>::pop();
}


//! Return the number of values in the stack.
/*!
*/
size_t Stack::size() const
{
    return std::stack<boost::any>::size();
}


//! Return whether the stack is empty.
/*!
*/
bool Stack::empty() const
{
    return std::stack<boost::any>::empty();
}

} // namespace fern
