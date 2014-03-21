Vector algebra  {#vector_algebra}
==============

Operations to implement:

- (lax)
- gradx, grady
- laplacian
- dot product
- diversion
- gradient


Theory
------
The *directed line segment* from point \f$P\f$ to \f$Q\f$, denoted by \f$\overrightarrow{PQ}\f$, is the straight-line segment that extents from \f$P\f$ to \f$Q\f$. \f$\overrightarrow{PQ}\f$ and \f$\overleftarrow{QP}\f$ are different since they point in opposite directions.

Point \f$P\f$ is called the *initial point* of the segment and point \f$Q\f$ is called the *terminal point*. Two important properties of a directed line segment are its magnitude (length) and its direction. If two directed line segments \f$\overrightarrow{PQ}\f$ and \f$\overrightarrow{RS}\f$ have the same magnitude and direction, they are said to be *equivalent*, no matter where they are located with respect to the origin.

The set of all directed line segments equivalent to a given directed line segment is called a *vector*.

A *vector* \f$\mathbf{v}\f$ in the xy-plane is an ordered pair of real numbers \f$(a, b)\f$. The number \f$a\f$ and \f$b\f$ are called the *components* of the vector \f$\mathbf{v}\f$. The *zero vector* is the vector \f$(0, 0)\f$ and is denoted by \f$\mathbf{0}\f$. Two vectors are equal if their corresponding components are equal. That is, \f$(a, b) == (c, d)\f$ if \f$a == c\f$ and \f$b == d\f$.

\f$|\mathbf{v}| = \f$ magnitude of \f$\mathbf{v} = \sqrt{a^2 + b^2}\f$.

The *direction* of the vector \f$\mathbf{v} = (a, b)\f$ is the angle \f$\theta\f$, measured in radians, that the vector makes with the positive x-axis. By convention \f$0 \le \theta < 2\pi\f$.

If \f$a \ne 0\f$, then \f$tan \theta = \frac{b}{a}\f$






See also
--------
- [Wikipedia on vector calculus](https://en.wikipedia.org/wiki/Vector_calculus)
- Laplace
    - [Laplace operator](http://en.wikipedia.org/wiki/Laplace_operator)
    - [Discrete laplace operator](http://en.wikipedia.org/wiki/Discrete_Laplace_operator)
