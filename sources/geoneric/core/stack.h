#pragma once
#include <stack>
#include <boost/any.hpp>


namespace geoneric {

//! Stack for storing any type of value.
/*!
  This is a regular stack class, but it supports storing any type of value. So
  you can push integers, strings, and whatever onto the same stack.
*/
class Stack
{

public:

                   Stack               ()=default;

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

  std::stack<boost::any> _stack;

};


template<class T>
inline void Stack::push(
    T const& value)
{
    _stack.push(value);
}


template<class T>
inline T const& Stack::top() const
{
    return boost::any_cast<T const&>(_stack.top());
}

} // namespace geoneric
