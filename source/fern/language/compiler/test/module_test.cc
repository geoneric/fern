// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern compiler
#include <boost/test/unit_test.hpp>
#include "fern/language/interpreter/data_sources.h"
#include "fern/language/compiler/module.h"


namespace fl = fern::language;


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
    public fl::Module
{

public:

    AbsModule()
        // std::vector<std::shared_ptr<fern::DataSource>> const& data_sources)

        : fl::Module(
              {
                  fl::DataDescription("argument_value")
              },
              {
                  fl::DataDescription("result_value")
              })

    {
    }


    // AbsModule(
    //     int argc,
    //     char** argv)

    //     : fl::Module(
    //           {
    //               fl::DataDescription("Attribute")
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
