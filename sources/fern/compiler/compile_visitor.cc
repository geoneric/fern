#include "fern/compiler/compile_visitor.h"
#include "fern/core/path.h"
#include "fern/ast/core/vertices.h"


namespace fern {

CompileVisitor::CompileVisitor(
    String const& header_filename)

    : AstVisitor(),
      _header_filename(header_filename),
      _header(),
      _module()

{
}


CompileVisitor::~CompileVisitor()
{
}


String const& CompileVisitor::header() const
{
    return _header;
}


String const& CompileVisitor::module() const
{
    return _module;
}


void CompileVisitor::Visit(
    AssignmentVertex& /* vertex */)
{
}


void CompileVisitor::Visit(
    IfVertex& /* vertex */)
{
}


void CompileVisitor::Visit(
    WhileVertex& /* vertex */)
{
}


void CompileVisitor::Visit(
    NameVertex& /* vertex */)
{
}


#define VISIT_NUMBER_VERTEX(                                                   \
    type)                                                                      \
void CompileVisitor::Visit(                                                    \
    NumberVertex<type>&)                                                       \
{                                                                              \
}

VISIT_NUMBER_VERTICES(VISIT_NUMBER_VERTEX)

#undef VISIT_NUMBER_VERTEX


void CompileVisitor::Visit(
    OperationVertex& /* vertex */)
{
}


void CompileVisitor::Visit(
    ModuleVertex& vertex)
{
    _header.clear();
    _module.clear();

    String model_name = Path(vertex.source_name()).filename().stem();
    String class_name = model_name;

    // TODO Change location of module header to geoneric/module.h.
    _header = String((boost::format(
        "#pragma once\n"
        "#include \"geoneric/compiler/module.h\"\n"
        "\n"
        "\n"
        "class %1%:\n"
        "    public fern::Module\n"
        "{\n"
        "\n"
        "public:\n"
        "\n"
        "                   %1%                 (int argc,\n"
        "                                        char** argv);\n"
        "\n"
        "                   ~%1%                ();\n"
        "\n"
        "};\n"
    ) % class_name).str());

    _module = String((boost::format(
        "#include \"%1%\"\n"
        "\n"
        "\n"
        "%2%::%2%(\n"
        "    int argc,\n"
        "    char** argv)\n"
        "\n"
        "    : fern::Module(std::vector<fern::Argument>(), argc, argv)\n"
        "\n"
        "{\n"
        "}\n"
        "\n"
        "\n"
        "%2%::~%2%()\n"
        "{\n"
        "}\n"
    ) % _header_filename % class_name).str());

}


void CompileVisitor::Visit(
    StringVertex& /* vertex */)
{
}


void CompileVisitor::Visit(
    SubscriptVertex& /* vertex */)
{
}

} // namespace fern
