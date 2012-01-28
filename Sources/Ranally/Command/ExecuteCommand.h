#ifndef INCLUDED_RANALLY_EXECUTECOMMAND
#define INCLUDED_RANALLY_EXECUTECOMMAND

#include "Command.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ExecuteCommand:
  public Command
{

  friend class ExecuteCommandTest;

public:

                   ExecuteCommand      (int argc,
                                        char** argv);

                   ~ExecuteCommand     ();

  int              execute             ();

private:

  void             execute             (UnicodeString const& xml);

};

} // namespace ranally

#endif