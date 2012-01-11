#ifndef INCLUDED_RANALLY_DESCRIBECOMMAND
#define INCLUDED_RANALLY_DESCRIBECOMMAND

#include "Command.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class DescribeCommand:
  public Command
{

  friend class DescribeCommandTest;

public:

                   DescribeCommand     (int argc,
                                        char** argv);

                   ~DescribeCommand    ();

  int              execute             ();

private:

  void             describe            (UnicodeString const& xml);

};

} // namespace ranally

#endif
