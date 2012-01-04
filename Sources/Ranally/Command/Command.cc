#include "Command.h"



namespace ranally {

Command::Command(
  int argc,
  char** argv)

  : _argc(argc),
    _argv(argv)

{
}



Command::~Command()
{
}



int Command::argc() const
{
  return _argc;
}



char** Command::argv() const
{
  return _argv;
}

} // namespace ranally

