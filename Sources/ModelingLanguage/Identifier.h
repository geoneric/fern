#ifndef INCLUDED_RANALLY_IDENTIFIER
#define INCLUDED_RANALLY_IDENTIFIER

#include <boost/noncopyable.hpp>



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Identifier: private boost::noncopyable
{

  friend class IdentifierTest;

private:

protected:

public:

                   Identifier          ();

  /* virtual */    ~Identifier         ();

};

} // namespace ranally

#endif
