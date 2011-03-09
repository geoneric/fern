#ifndef INCLUDED_RANALLY_LANGUAGE_PURIFYVISITOR
#define INCLUDED_RANALLY_LANGUAGE_PURIFYVISITOR

#include <boost/noncopyable.hpp>



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class PurifyVisitor: private boost::noncopyable
{

  friend class PurifyVisitorTest;

public:

                   PurifyVisitor       ();

  /* virtual */    ~PurifyVisitor      ();

protected:

private:

};

} // namespace language
} // namespace ranally

#endif
