#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define USAGE "Usage:\n\trbgs nx ny t c [j|g]\n\nWhere:\n \
 nx,ny\tnumber of discretization intervals in x and y axis, repectively\n \
 t\tnumber of threads\n \
 c\tnumber of iterations\n \
 j|q\tmethod to be used (j for Jacobi, g for Gaussian)\n\n"

#define K 2 * M_PI

typedef enum { JACOBI, GAUSS } method_t;

double *current_grid, *previous_grid;

double f(double x, double y) {
    return 4 * M_PI * M_PI * sin(2 * M_PI * x) * sinh(2 * M_PI * y);
}

void borders(int nx, int ny) {
    int ix, iy;

    double hx = 2 / (double) nx;

    for (ix=0; ix<=nx; ++ix) {
        current_grid[ix*(ny+1) + 0] = previous_grid[ix*(ny+1) + 0] = 0;
        current_grid[ix*(ny+1) + ny] = previous_grid[ix*(ny+1) + ny] = sin(2*M_PI*(ix*hx)) * sinh(2*M_PI);
    }

    for (iy=0; iy<=ny; ++iy) {
        current_grid[0*(ny+1) + iy] = previous_grid[0*(ny+1) + iy] = 0;
        current_grid[nx*(ny+1) + iy] = previous_grid[nx*(ny+1) + iy] = 0;
    }
}

void jacobi(int nx, int ny, int num_threads, int num_iterations) {
    int it, ix, iy;
    double *tmp;
    double sum;

    double hx = 2 / (double) nx;
    double hy = 1 / (double) ny;

    for (it=0; it<num_iterations; ++it) {
        tmp = previous_grid;
        previous_grid = current_grid;
        current_grid = tmp;

        for (ix=1; ix<nx; ++ix) {
            for (iy=1; iy<ny; ++iy) {
                sum = 0;

                sum += -(previous_grid[(ix-1)*(ny+1) + iy] / hx*hx);
                sum += -(previous_grid[(ix+1)*(ny+1) + iy] / hx*hx);
                sum += -(previous_grid[ix*(ny+1) + (iy-1)] / hy*hy);
                sum += -(previous_grid[ix*(ny+1) + (iy+1)] / hy*hy);

                current_grid[ix*(ny+1) + iy] = (f(ix*hx, iy*hy) - sum) / (2 / hx*hx + 2 / hy*hy + K*K);
            }
        }
    }
}

int main(int argc, char **argv) {
    int ix, iy, nx, ny, num_threads, num_iterations;
    double hx, hy;
    method_t method;
    FILE *solution;

    if (argc != 6) {
        fprintf(stderr, USAGE);
        exit(1);
    }

    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    num_threads = atoi(argv[3]);
    num_iterations = atoi(argv[4]);

    if (strcmp(argv[5], "j") == 0) {
        method = JACOBI;
    }
    else if (strcmp(argv[5], "g") == 0) {
        method = GAUSS;
    }
    else {
        fprintf(stderr, "error: '%s' is not a valid calculation method (valid options are 'j' for Jacobi or 'g' for Gauss)\n", argv[5]);
        exit(1);
    }

    if ((nx < 1) || (ny < 1)) {
        fprintf(stderr, "error: %dx%d are not valid dimensions for discretization\n", nx, ny);
        exit(1);
    }

    if (num_threads < 1) {
        fprintf(stderr, "error: %d is not a valid number of threads\n", num_threads);
        exit(1);
    }

    if (num_iterations < 1) {
        fprintf(stderr, "error: %d is not a valid number of iterations\n", num_iterations);
        exit(1);
    }

    hx = 2 / (double) nx;
    hy = 1 / (double) ny;

    // open solution.txt file to write the resulting solution
    if ((solution = fopen("solution.txt", "w")) == NULL) {
        fprintf(stderr, "error: failed to open 'solution.txt' for writing\n");
        exit(1);
    }

    // allocate grids
    current_grid = (double*) malloc((nx+1) * (ny+1) * sizeof(double));
    previous_grid = (double*) malloc((nx+1) * (ny+1) * sizeof(double));
    if ((current_grid == NULL) || (previous_grid == NULL)) {
        fprintf(stderr, "error: memory allocation failed\n");
        exit(1);
    }

    // initialize grids
    for (ix=0; ix<=nx; ++ix) {
        for (iy=0; iy<=ny; ++iy) {
            current_grid[ix*(ny+1) + iy] = previous_grid[ix*(ny+1) + iy] = 0;
        }
    }

    // initialize borders
    borders(nx, ny);

    // calculate
    jacobi(nx, ny, num_threads, num_iterations);

    // write solution to file
    for (ix=0; ix<=nx; ++ix) {
        for (iy=0; iy<=ny; ++iy) {
            fprintf(solution, "%lf %lf %lf\n", ix*hx, iy*hy, current_grid[ix*(ny+1) + iy]);
        }
        fprintf(solution, "\n");
    }

    free(current_grid);
    free(previous_grid);
    fclose(solution);
    exit(0);
}
