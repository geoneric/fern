#ifndef INCLUDED_RANALLY_ATTRIBUTE
#define INCLUDED_RANALLY_ATTRIBUTE

#include <unicode/unistr.h>
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

  UnicodeString const& name            () const;

protected:

                   Attribute           (UnicodeString const& name);

private:

  UnicodeString    _name;

};

} // namespace ranally

#endif
