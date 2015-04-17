// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <algorithm>
#include <type_traits>
#include <utility>


namespace fern {

/*!
    @brief      ScopeExit instances store a function to be called when the
                instance goes out of scope.

    This class is useful when you have the responsibility to perform some
    cleanup action, like closing a file, when an exception is thrown.

    @code
    auto status = open_file("blah.dat", &file_id);

    // Check status. Throw exception if needed.
    // ...

    // File is opened successfuly. Make sure we close it when an exception
    // is thrown from now on.
    auto file_closer = makeScopeExit([&file_id]() {
        static_cast<void>(close_file(file_id));
    });

    // Do something with the file. This code may throw an exception.
    // ...

    // Release the CloseExit instance of its responsibility.
    file_closer.release();

    // Close the file ourselves and handle errors.
    status = close_file(file_id);

    // Check status. Throw exception if needed.
    // ...
    @endcode

    If you need to be able to handle any errors raised by the cleanup action,
    you should release the ScopeExit instance of its responsibility and
    perform the cleanup action yourself.
*/
template<
    typename Function>
class ScopeExit
{

private:

    //! Function to call upon destruction.
    Function       _function;

    //! Are we still responsible for calling the function?
    bool           _execute_on_destruction;

public:

    explicit       ScopeExit           (Function&& function) noexcept;

                   ScopeExit           (ScopeExit const& other)=delete;

                   ScopeExit           (ScopeExit&& other)=default;

                   ~ScopeExit          () noexcept(noexcept(_function));

    ScopeExit&     operator=           (ScopeExit const& other)=delete;

    ScopeExit&     operator=           (ScopeExit&& other)=delete;

    void           release             ();

};


/*!
    @brief      Construct an instance storing the @a function passed in.

    As long the instance is not @ref release() d of its responsibility, it
    will call the function upon destruction.
*/
template<
    typename Function>
inline ScopeExit<Function>::ScopeExit(
    Function&& function) noexcept

    : _function(std::move(function)),
      _execute_on_destruction(true)

{
}


/*!
    @brief      Destruct the instance.

    If the instance is not @ref release() d of its responsibility, it
    will call the layered function.
*/
template<
    typename Function>
inline ScopeExit<Function>::~ScopeExit() noexcept(noexcept(_function))
{
    if(_execute_on_destruction) {
        _function();
    }
}


/*!
    @brief      Release the instance of its responsibility to call the layered
                function upon destruction.
*/
template<
    typename Function>
inline void ScopeExit<Function>::release()
{
    _execute_on_destruction = false;
}


/*!
    @brief      Return a ScopeExit instance based on the @a function
                passed in.
*/
template<
    typename Function>
inline auto makeScopeExit(
    Function&& function) noexcept
{
    return ScopeExit<std::remove_reference_t<Function>>(
        std::forward<Function>(function));
}

} // namespace fern
