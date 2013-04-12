#define BOOST_TEST_MODULE ranally language
#include <boost/test/unit_test.hpp>
#include "ranally/script/algebra_parser.h"
#include "ranally/language/assignment_vertex.h"
#include "ranally/language/identify_visitor.h"
#include "ranally/language/if_vertex.h"
#include "ranally/language/function_vertex.h"
#include "ranally/language/name_vertex.h"
#include "ranally/language/operator_vertex.h"
#include "ranally/language/script_vertex.h"
#include "ranally/language/xml_parser.h"


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

    ranally::AlgebraParser _algebra_parser;

    ranally::XmlParser _xml_parser;

    ranally::IdentifyVisitor _visitor;
};


BOOST_FIXTURE_TEST_SUITE(identify_visitor, Support)

BOOST_AUTO_TEST_CASE(visit_empty_script)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        ranally::String("")));
    tree->Accept(_visitor);
}


BOOST_AUTO_TEST_CASE(visit_name)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        ranally::String("a")));

    ranally::NameVertex const* vertex_a =
        dynamic_cast<ranally::NameVertex const*>(&(*tree->statements()[0]));

    BOOST_CHECK(vertex_a->definitions().empty());
    BOOST_CHECK(vertex_a->uses().empty());

    tree->Accept(_visitor);

    BOOST_CHECK(vertex_a->definitions().empty());
    BOOST_CHECK(vertex_a->uses().empty());
}


BOOST_AUTO_TEST_CASE(visit_assignment)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String("a = b")));

        ranally::AssignmentVertex const* assignment =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[0]));
        ranally::NameVertex const* vertex_a =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment->target()));
        ranally::NameVertex const* vertex_b =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment->expression()));

        BOOST_CHECK(vertex_a->definitions().empty());
        BOOST_CHECK(vertex_a->uses().empty());

        BOOST_CHECK(vertex_b->definitions().empty());
        BOOST_CHECK(vertex_b->uses().empty());

        tree->Accept(_visitor);

        BOOST_REQUIRE_EQUAL(vertex_a->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertex_a->definitions()[0], vertex_a);
        BOOST_CHECK(vertex_a->uses().empty());

        BOOST_CHECK(vertex_b->definitions().empty());
        BOOST_CHECK(vertex_b->uses().empty());
    }

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String(
                "a = b\n"
                "d = f(a)\n"
        )));

        ranally::AssignmentVertex const* assignment_1 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[0]));
        ranally::NameVertex const* vertex_a1 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment_1->target()));
        ranally::NameVertex const* vertex_b =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment_1->expression()));

        ranally::AssignmentVertex const* assignment_2 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[1]));
        ranally::FunctionVertex const* function =
            dynamic_cast<ranally::FunctionVertex const*>(
                &(*assignment_2->expression()));
        ranally::NameVertex const* vertex_a2 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*function->expressions()[0]));
        ranally::NameVertex const* vertex_d =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment_2->target()));

        BOOST_CHECK(vertex_a1->definitions().empty());
        BOOST_CHECK(vertex_a1->uses().empty());

        BOOST_CHECK(vertex_b->definitions().empty());
        BOOST_CHECK(vertex_b->uses().empty());

        BOOST_CHECK(vertex_d->definitions().empty());
        BOOST_CHECK(vertex_d->uses().empty());

        BOOST_CHECK(vertex_a2->definitions().empty());
        BOOST_CHECK(vertex_a2->uses().empty());

        tree->Accept(_visitor);

        BOOST_REQUIRE_EQUAL(vertex_a1->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertex_a1->definitions()[0], vertex_a1);
        BOOST_REQUIRE_EQUAL(vertex_a1->uses().size(), 1u);
        BOOST_CHECK_EQUAL(vertex_a1->uses()[0], vertex_a2);

        BOOST_CHECK(vertex_b->definitions().empty());
        BOOST_CHECK(vertex_b->uses().empty());

        BOOST_REQUIRE_EQUAL(vertex_d->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertex_d->definitions()[0], vertex_d);
        BOOST_CHECK(vertex_d->uses().empty());

        BOOST_REQUIRE_EQUAL(vertex_a2->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertex_a2->definitions()[0], vertex_a1);
        BOOST_CHECK(vertex_a2->uses().empty());
    }
}


BOOST_AUTO_TEST_CASE(visit_if)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String(
                "a = b\n"
                "if True:\n"
                "    a = c\n"
                "d = a\n"
        )));

        ranally::AssignmentVertex const* assignment_1 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[0]));
        ranally::NameVertex const* vertex_a1 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment_1->target()));

        ranally::IfVertex const* ifVertex =
            dynamic_cast<ranally::IfVertex const*>(
                &(*tree->statements()[1]));
        ranally::AssignmentVertex const* assignment_2 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*ifVertex->true_statements()[0]));
        ranally::NameVertex const* vertex_a2 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment_2->target()));

        ranally::AssignmentVertex const* assignment_3 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[2]));
        ranally::NameVertex const* vertex_a3 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment_3->expression()));

        tree->Accept(_visitor);

        BOOST_REQUIRE_EQUAL(vertex_a1->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertex_a1->definitions()[0], vertex_a1);
        BOOST_REQUIRE_EQUAL(vertex_a1->uses().size(), 1u);
        BOOST_CHECK_EQUAL(vertex_a1->uses()[0], vertex_a3);

        BOOST_REQUIRE_EQUAL(vertex_a2->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertex_a2->definitions()[0], vertex_a2);
        // TODO
        // BOOST_REQUIRE_EQUAL(vertex_a2->uses().size(), 1u);
        // BOOST_CHECK_EQUAL(vertex_a2->uses()[0], vertex_a3);

        // BOOST_REQUIRE_EQUAL(vertex_a3->definitions().size(), 2u);
        // BOOST_CHECK_EQUAL(vertex_a3->definitions()[0], vertex_a1);
        // BOOST_CHECK_EQUAL(vertex_a3->definitions()[0], vertex_a2);
        // BOOST_CHECK(vertex_a3->uses().empty());
    }
}


BOOST_AUTO_TEST_CASE(visit_reuse_of_identifiers)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String(
                "a = \"MyRaster\"\n"
                "b = abs(a)\n"
                "c = abs(b)\n"
                "b = c + b\n"
        )));

        ranally::AssignmentVertex const* assignment_1 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[0]));
        ranally::NameVertex const* vertex_a1 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment_1->target()));
        BOOST_REQUIRE(assignment_1);
        BOOST_REQUIRE(vertex_a1);

        ranally::AssignmentVertex const* assignment_2 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[1]));
        ranally::NameVertex const* vertex_b1 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment_2->target()));
        ranally::FunctionVertex const* vertex_abs1 =
            dynamic_cast<ranally::FunctionVertex const*>(
                &(*assignment_2->expression()));
        ranally::NameVertex const* vertex_a2 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*vertex_abs1->expressions()[0]));
        BOOST_REQUIRE(assignment_2);
        BOOST_REQUIRE(vertex_b1);
        BOOST_REQUIRE(vertex_abs1);
        BOOST_REQUIRE(vertex_a2);

        ranally::AssignmentVertex const* assignment_3 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[2]));
        ranally::NameVertex const* vertex_c1 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment_3->target()));
        ranally::FunctionVertex const* vertex_abs2 =
            dynamic_cast<ranally::FunctionVertex const*>(
                &(*assignment_3->expression()));
        ranally::NameVertex const* vertex_b2 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*vertex_abs2->expressions()[0]));
        BOOST_REQUIRE(assignment_3);
        BOOST_REQUIRE(vertex_c1);
        BOOST_REQUIRE(vertex_abs2);
        BOOST_REQUIRE(vertex_b2);

        ranally::AssignmentVertex const* assignment_4 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[3]));
        ranally::NameVertex const* vertex_b4 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment_4->target()));
        ranally::OperatorVertex const* vertex_plus1 =
            dynamic_cast<ranally::OperatorVertex const*>(
                &(*assignment_4->expression()));
        ranally::NameVertex const* vertex_c2 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*vertex_plus1->expressions()[0]));
        ranally::NameVertex const* vertex_b3 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*vertex_plus1->expressions()[1]));
        BOOST_REQUIRE(assignment_4);
        BOOST_REQUIRE(vertex_b4);
        BOOST_REQUIRE(vertex_plus1);
        BOOST_REQUIRE(vertex_c2);
        BOOST_REQUIRE(vertex_b3);

        tree->Accept(_visitor);

        BOOST_REQUIRE_EQUAL(vertex_a1->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertex_a1->definitions()[0], vertex_a1);
        BOOST_REQUIRE_EQUAL(vertex_a1->uses().size(), 1u);
        BOOST_CHECK_EQUAL(vertex_a1->uses()[0], vertex_a2);

        BOOST_REQUIRE_EQUAL(vertex_b1->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertex_b1->definitions()[0], vertex_b1);
        BOOST_REQUIRE_EQUAL(vertex_b1->uses().size(), 2u);
        BOOST_CHECK_EQUAL(vertex_b1->uses()[0], vertex_b2);
        BOOST_CHECK_EQUAL(vertex_b1->uses()[1], vertex_b3);

        BOOST_REQUIRE_EQUAL(vertex_c1->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertex_c1->definitions()[0], vertex_c1);
        BOOST_REQUIRE_EQUAL(vertex_c1->uses().size(), 1u);
        BOOST_CHECK_EQUAL(vertex_c1->uses()[0], vertex_c2);

        BOOST_REQUIRE_EQUAL(vertex_b4->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertex_b4->definitions()[0], vertex_b4);
        BOOST_REQUIRE_EQUAL(vertex_b4->uses().size(), 0u);
    }
}

BOOST_AUTO_TEST_SUITE_END()
