// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/compiler/compile_visitor.h"
#include "fern/core/path.h"
#include "fern/language/ast/core/vertices.h"
#include "fern/language/ast/visitor/io_visitor.h"


namespace fern {
namespace language {

CompileVisitor::CompileVisitor(
    OperationsPtr const& operations,
    String const& header_filename)

    : AstVisitor(),
      _operations(operations),
      _header_filename(header_filename),
      _header(),
      _statement(),
      _body(),
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
    AssignmentVertex& vertex)
{
    _statement.clear();
    _statement +=
        "std::vector<std::shared_ptr<fern::Argument>> results(";
    vertex.expression()->Accept(*this);
    _body.emplace_back(_statement);
    _body.emplace_back(boost::format(
        "std::shared_ptr<fern::Argument> %1%(results[0])"
        ) % vertex.target()->name());
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
    NameVertex& vertex)
{
    _statement += vertex.name();
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
    OperationVertex& vertex)
{
    Operation const& operation(*_operations->operation(vertex.name()));

    _statement += boost::format(
        "_operations->operation(\"%1%\")->execute(std::vector<std::shared_ptr<fern::Argument>>{"
        ) % operation.name();
    visit_expressions(vertex.expressions());

    _statement +=
        "}))";
}


void CompileVisitor::Visit(
    StatementVertex& vertex)
{
    _statement.clear();
    AstVisitor::Visit(vertex);
    // _statement += ";";
}


void CompileVisitor::Visit(
    ModuleVertex& vertex)
{
    _header.clear();
    _statement.clear();
    _body.clear();
    _module.clear();

    // Determine inputs and outputs of the module.
    IOVisitor visitor;
    vertex.Accept(visitor);
    String arguments_initializer_list;
    String results_initializer_list;

    {
        std::vector<String> initializers;
        for(auto const& input: visitor.inputs()) {
            initializers.emplace_back(boost::format(
                "fern::DataDescription(\"%1%\")"
            ) % input);
        }
        arguments_initializer_list = join(initializers, ", ");

        initializers.clear();
        for(auto const& output: visitor.outputs()) {
            initializers.emplace_back(boost::format(
                "fern::DataDescription(\"%1%\")"
            ) % output->name());
        }
        results_initializer_list = join(initializers, ", ");
    }


    // Prepare the module's inputs.
    for(size_t i = 0; i < visitor.inputs().size(); ++i) {
        _body.emplace_back((boost::format(
            "std::shared_ptr<fern::Argument> %1%(data_sources[%2%]->read());"
            ) % visitor.inputs()[i] % i).str());
    }

    // Add the module's C++ code to the body.
    AstVisitor::Visit(vertex);

    // Write out the results, if needed.
    for(size_t i = 0; i < visitor.outputs().size(); ++i) {
        _body.emplace_back((boost::format(
            "if(%1% < data_syncs.size()) {"
            ) % i).str());
        _body.emplace_back((boost::format(
            "    data_syncs[%1%]->write(*%2%);"
            ) % i % visitor.outputs()[i]->name()).str());
        _body.emplace_back(
            "}");
    }

    // Now put the new module's code in the header and module strings.
    String model_name =
        Path(vertex.source_name()).filename().stem().generic_string();
    String class_name = model_name;

    _header = String((boost::format(
"#pragma once\n"
"#include \"fern/language/compiler/module.h\"\n"
"#include \"fern/language/operation/std/operations.h\"\n"
"\n"
"\n"
"class %1%:\n"
"    public fern::Module\n"
"{\n"
"\n"
"public:\n"
"\n"
"                   %1%                 ();\n"
"\n"
"                   ~%1%                ();\n"
"\n"
"    void           run                 (std::vector<std::shared_ptr<\n"
"                                            fern::DataSource>> const&\n"
"                                                data_sources,\n"
"                                        std::vector<std::shared_ptr<\n"
"                                            fern::DataSync>> const&\n"
"                                                data_syncs) const;\n"
"\n"
"private:\n"
"\n"
"    std::shared_ptr<fern::Operations> _operations;\n"
"\n"
"};\n"
    ) % class_name).str());

    _module = String((boost::format(
"#include \"%1%\"\n"
"\n"
"\n"
"%2%::%2%()\n"
"\n"
"    : fern::Module(\n"
"          {\n"
"              // Inputs.\n"
"              %3%\n"
"          },\n"
"          {\n"
"              // Outputs.\n"
"              %4%\n"
"          }),\n"
"      _operations(fern::operations())\n"
"\n"
"{\n"
"}\n"
"\n"
"\n"
"%2%::~%2%()\n"
"{\n"
"}\n"
"\n"
"\n"
"void %2%::run(\n"
"    std::vector<std::shared_ptr<fern::DataSource>> const& data_sources,\n"
"    std::vector<std::shared_ptr<fern::DataSync>> const& data_syncs) const\n"
"{\n"
"    check_sources_and_syncs(data_sources, data_syncs);\n"
"    %5%\n"
"}\n"
    ) % _header_filename % class_name
      % arguments_initializer_list
      % results_initializer_list
      % (join(_body, ";\n    ") +
            (!_body.empty() ? String(";") : String("")))).str());
}


void CompileVisitor::Visit(
    StringVertex& /* vertex */)
{
}


void CompileVisitor::Visit(
    SubscriptVertex& /* vertex */)
{
}

} // namespace language
} // namespace fern
