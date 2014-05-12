#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_ITERATIONS 1000
#define NX 20
#define NY 10

#define HX (2 / (double) NX)
#define HY (1 / (double) NY)

#define XPOS(X) (X) * HX
#define YPOS(Y) (Y) * HY

#define ZIDX(X, Y) (unsigned int) (((X) * (NY + 1)) + (Y))

#define K 2 * M_PI


double *current_grid, *previous_grid;

double f(double x, double y) {
    return 4 * M_PI * M_PI * sin(2 * M_PI * x) * sinh(2 * M_PI * y);
}

void borders() {
    unsigned int ix, iy;

    for (ix=0; ix<=NX; ++ix) {
        current_grid[ZIDX(ix, 0)] = previous_grid[ZIDX(ix, 0)] = 0;
        current_grid[ZIDX(ix, NY)] = previous_grid[ZIDX(ix, NY)] = sin(2*M_PI*XPOS(ix)) * sinh(2*M_PI);
    }

    for (iy=0; iy<=NY; ++iy) {
        current_grid[ZIDX(0, iy)] = previous_grid[ZIDX(0, iy)] = 0;
        current_grid[ZIDX(NX, iy)] = previous_grid[ZIDX(NX, iy)] = 0;
    }
}

void jacobi() {
    unsigned int it, ix, iy;
    double *tmp;
    double sum;

    for (it=0; it<NUM_ITERATIONS; ++it) {
        tmp = previous_grid;
        previous_grid = current_grid;
        current_grid = tmp;

        for (ix=1; ix<NX; ++ix) {
            for (iy=1; iy<NY; ++iy) {
                sum = 0;

                sum += -(previous_grid[ZIDX(ix-1, iy)] / HX*HX);
                sum += -(previous_grid[ZIDX(ix+1, iy)] / HX*HX);
                sum += -(previous_grid[ZIDX(ix, iy-1)] / HY*HY);
                sum += -(previous_grid[ZIDX(ix, iy+1)] / HY*HY);

                current_grid[ZIDX(ix, iy)] = (f(XPOS(ix), YPOS(iy)) - sum) / (2 / HX*HX + 2 / HY*HY + K*K);
            }
        }
    }
}

void cleanup() {
    free(current_grid);
    free(previous_grid);
}

int main(int argc, char **argv) {
    unsigned int ix, iy;

    FILE *solution = fopen("solution.txt", "w");

    if (solution == NULL) {
        fprintf(stderr, "error: cannot open 'solution.txt' for writing!\n");
        exit(1);
    }

    current_grid = (double*) malloc((NX+1) * (NY+1) * sizeof(double));
    previous_grid = (double*) malloc((NX+1) * (NY+1) * sizeof(double));

    for (ix=0; ix<=NX; ++ix) {
        for (iy=0; iy<=NY; ++iy) {
            current_grid[ZIDX(ix, iy)] = previous_grid[ZIDX(ix, iy)] = 0;
        }
    }

    borders();

    jacobi();

    for (ix=0; ix<=NX; ++ix) {
        for (iy=0; iy<=NY; ++iy) {
            fprintf(solution, "%lf %lf %lf\n", XPOS(ix), YPOS(iy), current_grid[ZIDX(ix, iy)]);
        }
        fprintf(solution, "\n");
    }

    cleanup();
    fclose(solution);
    exit(0);
}
