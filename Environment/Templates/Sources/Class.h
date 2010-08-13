#ifndef INCLUDED_PACK_CLASS
#define INCLUDED_PACK_CLASS

#include <boost/noncopyable.hpp>



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Class: private boost::noncopyable
{

  friend class ClassTest;

private:

protected:

public:

                   Class               ();

  /* virtual */    ~Class              ();

};

} // namespace ranally

#endif
