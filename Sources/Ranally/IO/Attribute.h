#ifndef INCLUDED_RANALLY_ATTRIBUTE
#define INCLUDED_RANALLY_ATTRIBUTE

#include <boost/noncopyable.hpp>
#include "Ranally/Util/String.h"



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

  String const&    name                () const;

protected:

                   Attribute           (String const& name);

private:

  String           _name;

};

} // namespace ranally

#endif
