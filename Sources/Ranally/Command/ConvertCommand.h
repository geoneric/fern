#ifndef INCLUDED_RANALLY_CONVERTCOMMAND
#define INCLUDED_RANALLY_CONVERTCOMMAND

#include "Command.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ConvertCommand:
  public Command
{

  friend class ConvertCommandTest;

public:

                   ConvertCommand      (int argc,
                                        char** argv);

                   ~ConvertCommand     ();

  int              execute             ();

private:

  int              convertToRanally    (int argc,
                                        char** argv);

  int              convertToCpp        (int argc,
                                        char** argv);

  String           convertToDotAst     (String const& xml,
                                        int modes);

  int              convertToDotAst     (int argc,
                                        char** argv);

  String           convertToDotFlowgraph(
                                        String const& xml);

  int              convertToDotFlowgraph(
                                        int argc,
                                        char** argv);

  int              convertToDot        (int argc,
                                        char** argv);

  int              convertToPython     (int argc,
                                        char** argv);

  int              convertToXml        (int argc,
                                        char** argv);

};

} // namespace ranally

#endif
