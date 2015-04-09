// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
