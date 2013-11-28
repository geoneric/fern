#define BOOST_TEST_MODULE fern compiler
#include <boost/test/unit_test.hpp>
#include "fern/interpreter/data_sources.h"
#include "fern/compiler/module.h"


class Support // :
    // public fern::IOClient
{

public:

    Support()
        // : fern::IOClient()
    {
    }

};


class AbsModule:
    public fern::Module
{

public:

    AbsModule()
        // std::vector<std::shared_ptr<fern::DataSource>> const& data_sources)

        : fern::Module(
              {
                  fern::DataDescription("argument_value")
              },
              {
                  fern::DataDescription("result_value")
              })

    {
    }


    // AbsModule(
    //     int argc,
    //     char** argv)

    //     : fern::Module(
    //           {
    //               fern::DataDescription("Attribute")
    //           }, argc, argv)

    // {
    // }

};


BOOST_FIXTURE_TEST_SUITE(module, Support)

BOOST_AUTO_TEST_CASE(number_of_arguments)
{
    // // Abs takes one argument.
    // // Not enough arguments.
    // BOOST_CHECK_THROW(AbsModule({}), std::invalid_argument);

    // // OK
    // BOOST_CHECK_NO_THROW(AbsModule(
    //     {
    //         std::shared_ptr<fern::DataSource>(
    //             new fern::ConstantSource<int>(-5))
    //     }));

    // // Too many arguments.
    // BOOST_CHECK_THROW(AbsModule(
    //     {
    //         std::shared_ptr<fern::DataSource>(
    //             new fern::ConstantSource<int>(-5)),
    //         std::shared_ptr<fern::DataSource>(
    //             new fern::ConstantSource<int>(-6)),
    //     }), std::invalid_argument);
}


BOOST_AUTO_TEST_CASE(type_of_arguments)
{
    // TODO
}


BOOST_AUTO_TEST_CASE(run)
{
    // AbsModule module(
    //     {
//!
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   .
  \sa        .
*/
    //         std::shared_ptr<fern::DataSource>(
    //             new fern::ConstantSource<int>(-5))
    //     });
    // module.run();
}

BOOST_AUTO_TEST_SUITE_END()
