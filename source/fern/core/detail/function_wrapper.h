#pragma once
#include <algorithm>
#include <memory>
#include "fern/configure.h"
#ifdef FERN_COMPILER_DOES_NOT_HAVE_MAKE_UNIQUE
#include "fern/core/memory.h"
#endif


namespace fern {
namespace detail {

class FunctionWrapper
{

public:

                   FunctionWrapper     ()=default;

                   FunctionWrapper     (FunctionWrapper&& other);

                   FunctionWrapper     (FunctionWrapper const& other)=delete;

    template<
        class Function>
                   FunctionWrapper     (Function&& function);

    FunctionWrapper& operator=         (FunctionWrapper&& other);

    FunctionWrapper& operator=         (FunctionWrapper const& other)=delete;

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

    : _concept(std::make_unique<Model<Function>>(std::move(function)))

{
}

} // namespace detail
} // namespace fern
