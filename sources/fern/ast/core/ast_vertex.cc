#include "fern/ast/core/ast_vertex.h"
// #include "fern/ast/copy_visitor.h"


namespace fern {

AstVertex::AstVertex()

    : Loki::BaseVisitable<>(),
      _line(0),
      _col(0),
      _successors()

{
}


AstVertex::AstVertex(
    int line_nr,
    int col_id)

    : Loki::BaseVisitable<>(),
      _line(line_nr),
      _col(col_id),
      _successors()

{
}


// AstVertex::AstVertex(
//   AstVertex const& other)
// 
//   : Loki::BaseVisitable<>(other),
//     _line(other._line),
//     _col(other._col)
// 
// {
//   // BOOST_FOREACH(AstVertex* vertex, other._successors) {
//   //   CopyVisitor visitor;
//   //   vertex->Accept(visitor);
//   //   _successors.push_back(visitor.vertex());
//   // }
// 
//   CopyVisitor visitor;
//   other.Accept(visitor);
//   _successors = visitor.syntax_vertices();
// }



void AstVertex::set_position(
    int line_nr,
    int col_id)
{
    _line = line_nr;
    _col = col_id;
}


int AstVertex::line() const
{
    return _line;
}


int AstVertex::col() const
{
    return _col;
}


AstVertex::AstVertices const& AstVertex::successors() const
{
    return _successors;
}


bool AstVertex::has_successor() const
{
    return !_successors.empty();
}


//! Return the successor in the control flow graph of this vertex.
/*!
  \return    Pointer to the successor.
  \warning   It is assumed that this vertex has only one successor.
*/
AstVertex const* AstVertex::successor() const
{
    return successor(0);
}


/*!
  \overload
*/
AstVertex* AstVertex::successor()
{
    return successor(0);
}


//! Return one of the successors in the control flow graph of this vertex.
/*!
  \param     index Index of successor to return.
  \return    Pointer to the successor.
*/
AstVertex const* AstVertex::successor(
    size_type index) const
{
    assert(index < _successors.size());
    return _successors[index];
}


/*!
  \overload
*/
AstVertex* AstVertex::successor(
    size_type index)
{
    assert(index < _successors.size());
    return _successors[index];
}


//! Adds a successor to the control flow graph of this vertex.
/*!
  \param     successor Successor to set.
  \sa        set_successor(AstVertex*)

  Prefer set_successor(AstVertex*) over this method.
*/
void AstVertex::add_successor(
    AstVertex* successor)
{
    assert(successor);
    _successors.push_back(successor);
}


//! Adds a successor to the control flow graph of this vertex.
/*!
  \param     successor Successor to set.
  \sa        add_successor(AstVertex*)

  Prefer this method over add_successor(AstVertex*) because it checks
  whether a successor is already set or not. Most of the times, a vertex
  has a single successor.  Only call add_successor for those special
  cases that a vertex can have more than one successor (eg: IfVertex).
*/
void AstVertex::set_successor(
    AstVertex* successor)
{
    assert(_successors.empty());
    add_successor(successor);
}

} // namespace fern
