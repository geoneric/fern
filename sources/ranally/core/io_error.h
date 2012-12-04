#pragma once
#include "ranally/core/exception.h"


namespace ranally {

class IOError:
    public Exception
{
public:

                   IOError             (String const& source_name,
                                        String const& message);

                   IOError             (String const& source_name,
                                        int errno_);

                   IOError             (IOError const&)=default;

    IOError&       operator=           (IOError const&)=default;

                   IOError             (IOError&&)=default;

    IOError&       operator=           (IOError&&)=default;

                   ~IOError            () noexcept(true);

    String         message             () const;

private:

    String         _source_name;

    String         _message;

};

} // namespace ranally
