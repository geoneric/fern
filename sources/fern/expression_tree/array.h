#pragma once
#include <memory>


namespace fern {

template<
    class Result>
struct Array
{

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

        Container container;

    };

    std::shared_ptr<Concept> _self;

};

} // namespace fern
