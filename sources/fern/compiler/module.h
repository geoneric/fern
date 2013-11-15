#pragma once
#include <vector>
#include "fern/compiler/argument.h"


namespace fern {

class Module
{

public:

    virtual void   run                 ();

protected:

                   Module              (std::vector<Argument> const& arguments,
                                        int argc,
                                        char** argv);

    virtual        ~Module             ()=default;

                   Module              (Module&&)=delete;

    Module&        operator=           (Module&&)=delete;

                   Module              (Module const&)=delete;

    Module&        operator=           (Module const&)=delete;

private:

    std::vector<Argument> const _arguments;

    int const      _argc;

    char** const   _argv;

    void           parse_command_line  ();

};

} // namespace fern
