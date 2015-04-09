// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <stack>
#include <boost/any.hpp>


namespace fern {

//! Stack for storing any type of value.
/*!
  This is a regular stack class, but it supports storing any type of value. So
  you can push integers, strings, and whatever onto the same stack.
*/
class Stack:
    private std::stack<boost::any>
{

public:

                   Stack               ()=default;

                   Stack               (Stack&&)=default;

    Stack&         operator=           (Stack&&)=default;

                   Stack               (Stack const&)=default;

    Stack&         operator=           (Stack const&)=default;

                   ~Stack              ()=default;

  template<class T>
  void             push                (T const& value);

  template<class T>
  T const&         top                 () const;

  boost::any const& top                () const;

  void             pop                 ();

  size_t           size                () const;

  bool             empty               () const;

private:

};


//! Push \a value on the stack.
/*!
  \tparam    T Type of value to add.
  \param     value Value to push.
*/
template<class T>
inline void Stack::push(
    T const& value)
{
    std::stack<boost::any>::push(value);
}


//! Return the top value from the stack.
/*!
  \tparam    T Type of value to return.
  \exception boost::bad_any_cast In case the value at the top of the stack
             is not of type \a T.
*/
template<class T>
inline T const& Stack::top() const
{
    return boost::any_cast<T const&>(top());
}

} // namespace fern
