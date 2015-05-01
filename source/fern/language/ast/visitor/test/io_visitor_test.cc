// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern ast
#include <boost/test/unit_test.hpp>
#include "fern/language/script/algebra_parser.h"
#include "fern/language/ast/core/vertices.h"
#include "fern/language/ast/xml/xml_parser.h"
#include "fern/language/ast/visitor/io_visitor.h"


namespace fl = fern::language;


class Support
{

public:

    Support()
        : _algebra_parser(),
          _xml_parser(),
          _visitor()
    {
    }

protected:

    fl::AlgebraParser _algebra_parser;

    fl::XmlParser _xml_parser;

    fl::IOVisitor _visitor;

};


BOOST_FIXTURE_TEST_SUITE(module_visitor, Support)


BOOST_AUTO_TEST_CASE(visit_inputs)
{
    std::string xml;
    std::shared_ptr<fl::ModuleVertex> tree;

    // The tree contains four inputs: b, c, a, d.
    tree = _xml_parser.parse_string(_algebra_parser.parse_string(u8R"(
a = foo(b, c, a, d, b)
)"));

    fl::IOVisitor visitor;
    tree->Accept(visitor);

    BOOST_CHECK_EQUAL(tree->scope()->statements().size(), 1u);

    auto inputs = visitor.inputs();
    BOOST_REQUIRE_EQUAL(inputs.size(), 4u);

    BOOST_CHECK_EQUAL(inputs[0], "b");
    BOOST_CHECK_EQUAL(inputs[1], "c");
    BOOST_CHECK_EQUAL(inputs[2], "a");
    BOOST_CHECK_EQUAL(inputs[3], "d");
}


BOOST_AUTO_TEST_CASE(visit_outputs)
{
    std::string xml;
    std::shared_ptr<fl::ModuleVertex> tree;

    // The tree contains five statements. There are four outputs: a, b, c, d.
    // a is overwritten, and only the last a counts as an output. Verify that
    // the last one is indeed reported, instead of the first one.
    tree = _xml_parser.parse_string(_algebra_parser.parse_string(u8R"(
a = 5
b = 6
c = 7
a = 8
d = 9
)"));

    fl::IOVisitor visitor;
    tree->Accept(visitor);

    BOOST_CHECK_EQUAL(tree->scope()->statements().size(), 5u);

    fl::AssignmentVertex const* assignment;
    fl::NameVertex const* a_1 = nullptr;
    fl::NameVertex const* a_2 = nullptr;
    fl::NameVertex const* b = nullptr;
    fl::NameVertex const* c = nullptr;
    fl::NameVertex const* d = nullptr;

    assignment = dynamic_cast<fl::AssignmentVertex const*>(
        &(*tree->scope()->statements()[0]));
    a_1 = dynamic_cast<fl::NameVertex const*>(&(*assignment->target()));
    BOOST_REQUIRE(a_1);

    assignment = dynamic_cast<fl::AssignmentVertex const*>(
        &(*tree->scope()->statements()[1]));
    b = dynamic_cast<fl::NameVertex const*>(&(*assignment->target()));
    BOOST_REQUIRE(b);

    assignment = dynamic_cast<fl::AssignmentVertex const*>(
        &(*tree->scope()->statements()[2]));
    c = dynamic_cast<fl::NameVertex const*>(&(*assignment->target()));
    BOOST_REQUIRE(c);

    assignment = dynamic_cast<fl::AssignmentVertex const*>(
        &(*tree->scope()->statements()[3]));
    a_2 = dynamic_cast<fl::NameVertex const*>(&(*assignment->target()));
    BOOST_REQUIRE(a_2);

    assignment = dynamic_cast<fl::AssignmentVertex const*>(
        &(*tree->scope()->statements()[4]));
    d = dynamic_cast<fl::NameVertex const*>(&(*assignment->target()));
    BOOST_REQUIRE(d);

    auto outputs = visitor.outputs();
    BOOST_REQUIRE_EQUAL(outputs.size(), 4u);

    BOOST_CHECK_EQUAL(outputs[0], b);
    BOOST_CHECK_EQUAL(outputs[1], c);
    BOOST_CHECK_EQUAL(outputs[2], a_2);
    BOOST_CHECK_EQUAL(outputs[3], d);
}

BOOST_AUTO_TEST_SUITE_END()
