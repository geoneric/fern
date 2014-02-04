#pragma once
#include <memory>
#include <boost/variant/get.hpp>
#include "fern/expression_tree/arity.h"
#include "fern/expression_tree/data.h"


namespace fern {

struct Implementation
{

    template<
        class Algorithm>
    Implementation(
        Algorithm const& algorithm)
        : _self(std::make_shared<Model<Algorithm, typename Algorithm::Arity>>(
              algorithm))
    {
    }

    Data evaluate(
        std::vector<Data> const& data) const
    {
        return _self->evaluate(data);
    }

    struct Concept
    {

        virtual ~Concept()=default;

        virtual Data evaluate(std::vector<Data> const& data) const=0;

    };

    template<
        class Algorithm,
        class Arity>
    struct Model
    {
    };

    template<
        class Algorithm>
    struct Model<Algorithm, arity::Unary>:
        Concept
    {

        Model(
            Algorithm const& algorithm)
            : algorithm(algorithm)
        {
        }

        Data evaluate(
            std::vector<Data> const& data) const
        {
            assert(data.size() == 1u);
            return algorithm(
                boost::get<typename Algorithm::A const&>(data[0]));
        }

        Algorithm algorithm;

    };

    template<
        class Algorithm>
    struct Model<Algorithm, arity::Binary>:
        Concept
    {

        Model(
            Algorithm const& algorithm)
            : algorithm(algorithm)
        {
        }

        Data evaluate(
            std::vector<Data> const& data) const
        {
            assert(data.size() == 2u);
            return algorithm(
                boost::get<typename Algorithm::A1 const&>(data[0]),
                boost::get<typename Algorithm::A2 const&>(data[1]));
        }

        Algorithm algorithm;

    };

    std::shared_ptr<Concept> _self;
};

} // namespace fern
