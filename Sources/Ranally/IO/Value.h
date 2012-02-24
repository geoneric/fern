#ifndef INCLUDED_RANALLY_VALUE
#define INCLUDED_RANALLY_VALUE

#include <boost/noncopyable.hpp>



namespace ranally {

//! A value is a property of an attribute's domain.
/*!
  \sa        .
*/
class Value:
  private boost::noncopyable
{

  friend class ValueTest;

public:

  virtual          ~Value              ();

protected:

                   Value               ();

private:

};

} // namespace ranally

#endif
