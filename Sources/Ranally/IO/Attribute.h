#ifndef INCLUDED_RANALLY_ATTRIBUTE
#define INCLUDED_RANALLY_ATTRIBUTE

#include <boost/noncopyable.hpp>



namespace ranally {

//! Class for Attribute instances.
/*!
  \sa        .
*/
class Attribute:
  private boost::noncopyable
{

  friend class AttributeTest;

public:

  virtual          ~Attribute          ();

protected:

                   Attribute           ();

private:

};

} // namespace ranally

#endif
