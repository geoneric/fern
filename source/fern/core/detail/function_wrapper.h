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
#include <memory>
#include "fern/configure.h"


namespace fern {
namespace detail {

class FunctionWrapper
{

public:

                   FunctionWrapper     ()=default;

                   FunctionWrapper     (FunctionWrapper const&)=delete;

                   FunctionWrapper     (FunctionWrapper&&)=default;

    template<
        class Function>
                   FunctionWrapper     (Function&& function);

                   ~FunctionWrapper    ()=default;

    FunctionWrapper& operator=         (FunctionWrapper const&)=delete;

    FunctionWrapper& operator=         (FunctionWrapper&&)=default;

    void           operator()          ();

private:

    struct Concept
    {

        virtual ~Concept()
        {
        }

        virtual void call()=0;

    };

    template<
        class Function>
    struct Model:
        public Concept
    {

        Model(
            Function&& function)

            : _function(std::move(function))

        {
        }

        void call()
        {
            _function();
        }

        Function   _function;

    };

    std::unique_ptr<Concept> _concept;

};


template<
    class Function>
inline FunctionWrapper::FunctionWrapper(
    Function&& function)

    : _concept{std::make_unique<Model<Function>>(std::move(function))}

{
}


inline void FunctionWrapper::operator()()
{
    _concept->call();
}

} // namespace detail
} // namespace fern
