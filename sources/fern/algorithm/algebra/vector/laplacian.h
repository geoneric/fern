#pragma once


// https://en.wikipedia.org/wiki/Vector_calculus

namespace fern {

// domain
// range
// algorithm

}



/* calculates the laplacian (div dot grad)
 * of a scalar field f(x,y): laplacian(f) = div · grad(f) = d^2(f)/dx^2
 * */
extern int vf_laplacian(MAP_REAL8 *result,
                  const MAP_REAL8 *scalar)
{
    int nrows, ncols;
    double dx, value, neighbour, gg;
    nrows  = result->NrRows(result);
    ncols  = result->NrCols(result);
    dx     = scalar->CellLength(scalar);

    for(int r = 0; r < nrows; ++r) {
        for(int c = 0; c < ncols; ++c) {
            gg = 0;

            // gg becomes sum of:
            //     2 * (north-west or center)
            //     2 * (north-east or center)
            //     2 * (south-west or center)
            //     2 * (south-east or center)
            //     3 * (north or center)
            //     3 * (west or center)
            //     3 * (east or center)
            //     3 * (south or center)
            //
            // These are 20 values.
            //
            // result becomes
            // (gg - 20 * center) / (dx * dx)

            if(scalar->Get(&value, r + 0, c + 0, scalar)) {
                // North-west cell.
                if(scalar->Get(&neighbour, r - 1, c - 1, scalar)) {
                    gg = gg + 2 * neighbour;
                }
                else {
                    gg = gg + 2 * value;
                }

                // North cell.
                if(scalar->Get(&neighbour, r - 1, c + 0, scalar)) {
                    gg = gg + 3 * neighbour;
                }
                else  gg = gg + 3 * value;

                // North-east cell.
                if(scalar->Get(&neighbour, r - 1, c + 1, scalar)) {
                    gg = gg + 2 * neighbour;
                }
                else {
                    gg = gg + 2 * value;
                }

                // West cell.
                if(scalar->Get(&neighbour, r + 0, c - 1, scalar)) {
                    gg = gg + 3 * neighbour;
                }
                else {
                    gg = gg + 3 * value;
                }

                // East cell.
                if(scalar->Get(&neighbour, r + 0, c + 1, scalar)) {
                    gg = gg + 3 * neighbour;
                }
                else {
                    gg = gg + 3 * value;
                }

                // South-west cell.
                if(scalar->Get(&neighbour, r + 1, c - 1, scalar)) {
                    gg = gg + 2*neighbour;
                }
                else {
                    gg = gg + 2 * value;
                }

                // South cell.
                if(scalar->Get(&neighbour, r + 1, c + 0, scalar)) {
                    gg = gg + 3 * neighbour;
                }
                else {
                    gg = gg + 3 * value;
                }

                // South-east cell.
                if(scalar->Get(&neighbour, r + 1, c + 1, scalar)) {
                    gg = gg + 2 * neighbour;
                }
                else {
                    gg = gg + 2 * value;
                }

                result->Put((gg - 20 * value) / (dx * dx), r, c, result);
            }
            else {
                result->PutMV(r, c, result);
            }
        }
   }

   return 0;
}
