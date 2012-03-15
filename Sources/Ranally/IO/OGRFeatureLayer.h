#ifndef INCLUDED_RANALLY_IO_OGRFEATURELAYER
#define INCLUDED_RANALLY_IO_OGRFEATURELAYER

#include <unicode/unistr.h>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>



class OGRLayer;

namespace ranally {

class Domain;

namespace io {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OGRFeatureLayer:
  private boost::noncopyable
{

  friend class OGRFeatureLayerTest;

public:

                   OGRFeatureLayer     (OGRLayer* const layer);

                   ~OGRFeatureLayer    ();

  UnicodeString    name                () const;

  Domain const&    domain              () const;

  template<class Domain>
  boost::shared_ptr<Domain> domain     () const;

private:

  OGRLayer* const  _layer;

  boost::shared_ptr<Domain> _domain;

};



template<class Domain>
boost::shared_ptr<Domain> OGRFeatureLayer::domain() const
{
  return boost::dynamic_pointer_cast<Domain>(_domain);
}

} // namespace io
} // namespace ranally

#endif
