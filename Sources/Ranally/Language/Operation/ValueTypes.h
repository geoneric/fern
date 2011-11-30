#ifndef INCLUDED_RANALLY_LANGUAGE_OPERATION_VALUETYPES
#define INCLUDED_RANALLY_LANGUAGE_OPERATION_VALUETYPES

#include <bitset>



namespace ranally {
namespace language {
namespace operation {

enum ValueType {
  Number,
  NrValueTypes
};



//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ValueTypes:
  public std::bitset<NrValueTypes>
{

  friend class ValueTypesTest;

public:

                   ValueTypes          ();

                   ~ValueTypes         ();

protected:

private:

};

} // namespace operation
} // namespace language
} // namespace ranally

#endif
