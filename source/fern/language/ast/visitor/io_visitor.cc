// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/visitor/io_visitor.h"
#include "fern/language/ast/core/vertices.h"


namespace fern {
namespace language {

IOVisitor::IOVisitor()

    : AstVisitor(),
      _mode(Mode::Using),
      _inputs(),
      _outputs(),
      _output_id_by_name()

{
}


IOVisitor::~IOVisitor()
{
}


std::vector<String> const& IOVisitor::inputs() const
{
    return _inputs;
}


std::vector<NameVertex const*> const& IOVisitor::outputs() const
{
    return _outputs;
}


void IOVisitor::Visit(
    AssignmentVertex& vertex)
{
    _mode = Mode::Using;
    vertex.expression()->Accept(*this);

    _mode = Mode::Defining;
    vertex.target()->Accept(*this);

    _mode = Mode::Using;
}


void IOVisitor::Visit(
    NameVertex& vertex)
{
    switch(_mode) {
        case Mode::Using: {
            if(std::find(_inputs.begin(), _inputs.end(), vertex.name()) ==
                    _inputs.end()) {
                _inputs.emplace_back(vertex.name());
            }
            break;
        }
        case Mode::Defining: {
            // TODO Assert that the scope is global.
            if(_output_id_by_name.find(vertex.name()) !=
                    _output_id_by_name.end()) {
                // This identifier is already stored as a result.
                // Remove it first. We want to store this latter occurence.
                _outputs.erase(_outputs.begin() + _output_id_by_name[
                    vertex.name()]);
            }

            _output_id_by_name[vertex.name()] = _outputs.size();
            _outputs.emplace_back(&vertex);
            break;
        }
    }
}


void IOVisitor::Visit(
    ModuleVertex& vertex)
{
    _inputs.clear();
    _outputs.clear();
    _output_id_by_name.clear();
    _mode = Mode::Using;

    AstVisitor::Visit(vertex);

    assert(_output_id_by_name.size() == _outputs.size());
}

} // namespace language
} // namespace fern
