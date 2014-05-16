#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define USAGE "Usage:\n\trbgs nx ny t c [j|g]\n\nWhere:\n \
 nx,ny\tnumber of discretization intervals in x and y axis, repectively\n \
 t\tnumber of threads\n \
 c\tnumber of iterations\n \
 j|q\tmethod to be used (j for Jacobi, g for Gaussian)\n\n"

#define K 2 * M_PI

#define F(X, Y) 4 * M_PI * M_PI * sin(2 * M_PI * X) * sinh(2 * M_PI * Y)

#define L(X, Y) ((X) * step + (int) floor((Y) / 2.0))

int nx, ny, num_threads, num_iterations;
double *current_grid, *previous_grid;
double *f;
double hx, hy;

void init_grid() {
    int ix, iy;

    for (ix=0; ix<=nx; ++ix) {
        current_grid[ix*(ny+1) + 0] = previous_grid[ix*(ny+1) + 0] = 0;
        current_grid[ix*(ny+1) + ny] = previous_grid[ix*(ny+1) + ny] = sin(2*M_PI*(ix*hx)) * sinh(2*M_PI);
    }

    for (iy=0; iy<=ny; ++iy) {
        current_grid[0*(ny+1) + iy] = previous_grid[0*(ny+1) + iy] = 0;
        current_grid[nx*(ny+1) + iy] = previous_grid[nx*(ny+1) + iy] = 0;
    }

    for (ix=1; ix<nx; ++ix) {
        for (iy=1; iy<ny; ++iy) {
            // current_grid[ix*(ny+1) + iy] = previous_grid[ix*(ny+1) + iy] = 0;

            // f[ix*(ny+1) + iy] = F(ix*hx, iy*hy);
        }
    }
}

void jacobi() {
    int it, ix, iy;
    double *tmp;
    double sum;

    for (it=0; it<num_iterations; ++it) {
        tmp = previous_grid;
        previous_grid = current_grid;
        current_grid = tmp;

        // #pragma omp parallel for collapse(2) private(sum)
        for (ix=1; ix<nx; ++ix) {
            for (iy=1; iy<ny; ++iy) {
                sum = 0;

                sum -= previous_grid[(ix-1)*(ny+1) + iy] / (hx*hx);
                sum -= previous_grid[(ix+1)*(ny+1) + iy] / (hx*hx);
                sum -= previous_grid[ix*(ny+1) + (iy-1)] / (hy*hy);
                sum -= previous_grid[ix*(ny+1) + (iy+1)] / (hy*hy);

                current_grid[ix*(ny+1) + iy] = (f[ix*(ny+1) + iy] - sum) /
                                                    (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
            }
        }
    }
}

void gauss() {
    double *grid_red, *grid_black;
    int size, step;

    step = ceil((nx+1) / 2.0);
    size = step * (ny+1);

    grid_red = (double*) malloc(size * sizeof(double));
    grid_black = (double*) malloc(size * sizeof(double));

    if ((grid_red == NULL) || (grid_black == NULL)) {
        fprintf(stderr, "error: memory allocation failed\n");
        exit(1);
    }

    #pragma omp parallel
    {
        int it, ix, iy, dy, i;
        double sum;

        for (it=0; it<=num_iterations; ++it) {
            #pragma omp for
            for (ix=1; ix<nx; ++ix) {
                dy = ((ix % 2) == 0) ? 2 : 1;

                for (iy=dy; iy<ny; iy += 2) {
                    i = L(ix, iy);

                    if (it == 0) {
                        grid_red[i] = 0;
                        f[ix*(ny+1) + iy] = F(ix*hx, iy*hy);
                        continue;
                    }

                    sum = 0;

                    sum -= grid_black[L(ix-1, iy)] / (hx*hx);
                    sum -= grid_black[L(ix+1, iy)] / (hx*hx);
                    sum -= grid_black[L(ix, iy-1)] / (hy*hy);
                    sum -= grid_black[L(ix, iy+1)] / (hy*hy);

                    grid_red[i] = (f[ix*(ny+1) + iy] - sum) /
                                (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                }
            }

            #pragma omp barrier

            #pragma omp for
            for (ix=1; ix<nx; ++ix) {
                dy = ((ix % 2) != 0) ? 2 : 1;

                for (iy=dy; iy<ny; iy += 2) {
                    i = L(ix, iy);

                    if (it == 0) {
                        grid_black[i] = 0;
                        f[ix*(ny+1) + iy] = F(ix*hx, iy*hy);
                        continue;
                    }

                    sum = 0;

                    sum -= grid_red[L(ix-1, iy)] / (hx*hx);
                    sum -= grid_red[L(ix+1, iy)] / (hx*hx);
                    sum -= grid_red[L(ix, iy-1)] / (hy*hy);
                    sum -= grid_red[L(ix, iy+1)] / (hy*hy);

                    grid_black[i] = (f[ix*(ny+1) + iy] - sum) /
                                (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                }
            }

            #pragma omp barrier

            // #pragma omp for
            // for (ix=1; ix<nx; ++ix) {
            //     dy = ((ix % 2) == 0) ? 2 : 1;

            //     for (iy=dy; iy<ny; iy += 2) {
            //         if (it == 0) {
            //             current_grid[ix*(ny+1) + iy] = 0;
            //             f[ix*(ny+1) + iy] = F(ix*hx, iy*hy);
            //             continue;
            //         }

            //         sum = 0;

            //         sum -= current_grid[(ix-1)*(ny+1) + iy] / (hx*hx);
            //         sum -= current_grid[(ix+1)*(ny+1) + iy] / (hx*hx);
            //         sum -= current_grid[ix*(ny+1) + (iy-1)] / (hy*hy);
            //         sum -= current_grid[ix*(ny+1) + (iy+1)] / (hy*hy);

            //         current_grid[ix*(ny+1) + iy] = (f[ix*(ny+1) + iy] - sum) /
            //                                             (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
            //     }
            // }

            // #pragma omp barrier

            // #pragma omp for
            // for (ix=1; ix<nx; ++ix) {
            //     dy = ((ix % 2) != 0) ? 2 : 1;

            //     for (iy=dy; iy<ny; iy += 2) {
            //         if (it == 0) {
            //             current_grid[ix*(ny+1) + iy] = 0;
            //             f[ix*(ny+1) + iy] = F(ix*hx, iy*hy);
            //             continue;
            //         }

            //         sum = 0;

            //         sum -= current_grid[(ix-1)*(ny+1) + iy] / (hx*hx);
            //         sum -= current_grid[(ix+1)*(ny+1) + iy] / (hx*hx);
            //         sum -= current_grid[ix*(ny+1) + (iy-1)] / (hy*hy);
            //         sum -= current_grid[ix*(ny+1) + (iy+1)] / (hy*hy);

            //         current_grid[ix*(ny+1) + iy] = (f[ix*(ny+1) + iy] - sum) /
            //                                             (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
            //     }
            // }

            // #pragma omp barrier
        }
    }

    int iix, iiy, ddy;
    for (iix=1; iix<nx; ++iix) {
        ddy = ((iix % 2) == 0) ? 2 : 1;

        for (iiy=ddy; iiy<ny; iiy += 2) {
            current_grid[iix*(ny+1) + iiy] = grid_red[L(iix, iiy)];
        }
    }

    for (iix=1; iix<nx; ++iix) {
        ddy = ((iix % 2) != 0) ? 2 : 1;

        for (iiy=ddy; iiy<ny; iiy += 2) {
            current_grid[iix*(ny+1) + iiy] = grid_black[L(iix, iiy)];
        }
    }
}

void cleanup() {
    free(current_grid);
    free(previous_grid);
}

int main(int argc, char **argv) {
    int ix, iy;
    double sum, residue, begin_time, end_time;
    FILE *solution;

    if (argc != 6) {
        fprintf(stderr, USAGE);
        exit(1);
    }

    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    num_threads = atoi(argv[3]);
    num_iterations = atoi(argv[4]);
    hx = 2 / (double) nx;
    hy = 1 / (double) ny;

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

    // allocate grids
    current_grid = (double*) malloc((nx+1) * (ny+1) * sizeof(double));
    previous_grid = (double*) malloc((nx+1) * (ny+1) * sizeof(double));
    f = (double*) malloc((nx+1) * (ny+1) * sizeof(double));
    if ((current_grid == NULL) || (previous_grid == NULL)) {
        fprintf(stderr, "error: memory allocation failed\n");
        exit(1);
    }

    init_grid();

    omp_set_num_threads(num_threads);

    begin_time = omp_get_wtime();

    // calculate using the appropriate method
    switch (argv[5][0]) {
    case 'j':
        jacobi();
        break;
    case 'g':
        gauss();
        break;
    default:
        fprintf(stderr, "error: '%s' is not a valid calculation method (valid options are 'j' for Jacobi or 'g' for Gauss-Seidel)\n", argv[5]);
        cleanup();
        exit(1);
    }

    end_time = omp_get_wtime();

    // write solution to file
    if ((solution = fopen("solution.txt", "w")) == NULL) {
        fprintf(stderr, "error: failed to open 'solution.txt' for writing\n");
        cleanup();
        exit(1);
    }
    for (ix=0; ix<=nx; ++ix) {
        for (iy=0; iy<=ny; ++iy) {
            fprintf(solution, "%lf %lf %lf\n", ix*hx, iy*hy, current_grid[ix*(ny+1) + iy]);
        }
        fprintf(solution, "\n");
    }
    fclose(solution);

    // calculate l2-norm of the residue
    residue = 0;
    for (ix=1; ix<nx; ++ix) {
        for (iy=1; iy<ny; ++iy) {
            sum = 0;

            sum -= current_grid[(ix-1)*(ny+1) + iy] / (hx*hx);
            sum -= current_grid[(ix+1)*(ny+1) + iy] / (hx*hx);
            sum += current_grid[ix*(ny+1) + iy] * (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
            sum -= current_grid[ix*(ny+1) + (iy-1)] / (hy*hy);
            sum -= current_grid[ix*(ny+1) + (iy+1)] / (hy*hy);

            residue += pow(f[ix*(ny+1) + iy] - sum, 2);
        }
    }

    printf("Tempo total de processamento: %lf segundos\n", end_time-begin_time);
    printf("Norma L2 do resíduo: %lf\n", sqrt(residue));

    cleanup();
    exit(0);
}
