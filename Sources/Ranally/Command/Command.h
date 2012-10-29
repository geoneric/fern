#pragma once
#include <boost/noncopyable.hpp>
#include "Ranally/Util/String.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Command:
    private boost::noncopyable
{

    friend class CommandTest;

public:

    virtual        ~Command            ();

    virtual int    execute             ()=0;

protected:

                   Command             (int argc,
                                        char** argv);

    int            argc                () const;

    char**         argv                () const;

    String         read                (std::string const& fileName);

    void           write               (String const& contents,
                                        std::string const& fileName);

private:

    int            _argc;

    char**         _argv;

};

} // namespace ranally
