#pragma once
#include <boost/noncopyable.hpp>



namespace ranally {
namespace io {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OGRClient:
  private boost::noncopyable
{

  friend class OGRClientTest;

public:

  virtual          ~OGRClient          ();

protected:

                   OGRClient           ();

private:

};

} // namespace io
} // namespace ranally
