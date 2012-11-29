#pragma once
#include "ranally/core/exception.h"


namespace ranally {

class IOError:
    public Exception
{
public:

                   IOError             (String const& filename,
                                        String const& message);

                   IOError             (String const& filename,
                                        int errno_);

                   IOError             (IOError const&)=default;

    IOError&       operator=           (IOError const&)=default;

                   IOError             (IOError&&)=default;

    IOError&       operator=           (IOError&&)=default;

                   ~IOError            () noexcept(true);

    String         message             () const;

private:

    String         _filename;

    String         _message;

};

} // namespace ranally
