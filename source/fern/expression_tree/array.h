#pragma once
#include <memory>
#include <type_traits>


namespace fern {
namespace expression_tree {

template<
    class Result>
struct Array
{

    static_assert(std::is_arithmetic<Result>::value, "Type must be numeric");

    using value_type = Result;

    using result_type = Array<Result>;

    template<
        class Container>
    Array(
        Container const& container)
        : _self(std::make_shared<Model<Container>>(container))
    {
    }

    template<
        class Container>
    Container const& container() const
    {
        // If this assertion fails, you are expecting a different type of
        // container than the one that is stored in the Model.
        assert(dynamic_cast<Model<Container>*>(_self.get()));
        return dynamic_cast<Model<Container> const*>(_self.get())->container;
    }

    struct Concept
    {

        virtual ~Concept()=default;

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

} // namespace expression_tree
} // namespace fern
