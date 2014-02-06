#pragma once
#include <memory>
#include <type_traits>


namespace fern {

template<
    class Result>
struct Array
{

    static_assert(std::is_arithmetic<Result>::value, "Type must be numeric");

    typedef Result value_type;

    typedef Array<Result> result_type;

    template<
        class Container>
    Array(
        Container const& container)
        : _self(std::make_shared<Model<Container>>(container))
    {
    }

    struct Concept
    {

        // virtual const_iterator begin() const=0;

        // virtual const_iterator end() const=0;

    };

    template<
        class Container>
    struct Model:
        Concept
    {

        Model(
            Container const& container)
            : container(container)
        {
        }

        Container const& container;

    };

    std::shared_ptr<Concept> _self;

};

} // namespace fern
