#ifndef INCLUDED_RANALLY_DOMAIN
#define INCLUDED_RANALLY_DOMAIN

#include <boost/noncopyable.hpp>



namespace ranally {

//! A Domain positions a Feature's Attribute in the spatio-temporal coordinate space.
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Domain:
  private boost::noncopyable
{

  friend class DomainTest;

public:

  virtual          ~Domain             ();

protected:

                   Domain              ();

private:

};

} // namespace ranally

#endif
