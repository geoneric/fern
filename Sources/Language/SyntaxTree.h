#ifndef INCLUDED_RANALLY_SYNTAXTREE
#define INCLUDED_RANALLY_SYNTAXTREE

#include <boost/noncopyable.hpp>



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class SyntaxTree: private boost::noncopyable
{

  friend class SyntaxTreeTest;

private:

protected:

public:

                   SyntaxTree               ();

  /* virtual */    ~SyntaxTree              ();

};

} // namespace ranally

#endif
