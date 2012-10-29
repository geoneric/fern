#pragma once
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

  enum Type {
    PointDomain,
    PolygonDomain
  };

  virtual          ~Domain             ();

  Type             type                () const;

  virtual bool     isSpatial           () const=0;

  virtual bool     isTemporal          () const=0;

protected:

                   Domain              (Type type);

private:

  Type             _type;

};

} // namespace ranally
