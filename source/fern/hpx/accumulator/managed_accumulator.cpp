#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include "server/managed_accumulator.hpp"


// Add factory registration functionality.
HPX_REGISTER_COMPONENT_MODULE()


typedef hpx::components::managed_component<
    examples::server::managed_accumulator
> accumulator_type;


HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(accumulator_type, managed_accumulator)


// Serialization support for managed_accumulator actions.
HPX_REGISTER_ACTION(
    accumulator_type::wrapped_type::reset_action,
    managed_accumulator_reset_action)
HPX_REGISTER_ACTION(
    accumulator_type::wrapped_type::add_action,
    managed_accumulator_add_action)
HPX_REGISTER_ACTION(
    accumulator_type::wrapped_type::query_action,
    managed_accumulator_query_action)
