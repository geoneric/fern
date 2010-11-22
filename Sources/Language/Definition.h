#ifndef INCLUDED_RANALLY_DEFINITION
#define INCLUDED_RANALLY_DEFINITION

#include <unicode/unistr.h>



namespace ranally {
namespace language {

//! Definition instances contain information about an identifier.
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Definition
{

  friend class DefinitionTest;

private:

  UnicodeString    _name;

protected:

public:

                   Definition          (UnicodeString const& name);

                   Definition          (Definition const& rhs);

  /* virtual */    ~Definition         ();

  UnicodeString const& name            () const;

};

} // namespace language
} // namespace ranally

#endif
