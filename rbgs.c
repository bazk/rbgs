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

#define F(X, Y) 4 * M_PI * M_PI * sin(2 * M_PI * (X) * hx) * sinh(2 * M_PI * (Y) * hy)

#define L(X, Y) ((int) (X) / 2) * (ny-1) + (Y)

// #define GET_XY_RED(I, X, Y) X = (int) I / (ny-1);
//             if (X % 2 == 0) {
//                 X = X * 2;
//                 X += (I % 2 != 0) ? 1 : 0;
//             }
//             else {
//                 X = X * 2;
//                 X += (I % 2 == 0) ? 1 : 0;
//             }
//             Y = I % (ny-1);

// #define GET_XY_BLACK(I, X, Y) X = (int) I / (ny-1);
//             if (X % 2 == 0) {
//                 X = X * 2;
//                 X += (I % 2 == 0) ? 1 : 0;
//             }
//             else {
//                 X = X * 2;
//                 X += (I % 2 != 0) ? 1 : 0;
//             }
//             Y = I % (ny-1);

#define BORDER(X) sin(2*M_PI*((X)*hx)) * sinh(2*M_PI)

#define INCREMENT(A, B) if (++B == ny-1) { A += 2; B = 0; } \
                        else { A += (A % 2 == 0) ? 1 : -1; }

typedef double v2df __attribute__ ((vector_size (16)));

int nx, ny, num_threads, num_iterations;
double hx, hy;
FILE *solution;

void jacobi() {
    int ix, iy;
    double *grid_current, *grid_next, *f, *tmp;
    double sum, residue, begin_time, end_time;

    grid_current = (double*) malloc((nx+1) * (ny+1) * sizeof(double));
    grid_next = (double*) malloc((nx+1) * (ny+1) * sizeof(double));
    f = (double*) malloc((nx+1) * (ny+1) * sizeof(double));

    begin_time = omp_get_wtime();

    // initialize borders
    for (ix=0; ix<(nx+1); ++ix) {
        grid_current[ix*(ny+1) + 0] = grid_next[ix*(ny+1) + 0] = 0;
        grid_current[ix*(ny+1) + ny] = grid_next[ix*(ny+1) + ny] = sin(2*M_PI*(ix*hx)) * sinh(2*M_PI);
    }
    for (iy=0; iy<(ny+1); ++iy) {
        grid_current[0*(ny+1) + iy] = grid_next[0*(ny+1) + iy] = 0;
        grid_current[nx*(ny+1) + iy] = grid_next[nx*(ny+1) + iy] = 0;
    }

    #pragma omp parallel private(ix, iy, sum)
    {
        int it;

        #pragma omp for
        for (ix=1; ix<nx; ++ix) {
            for (iy=1; iy<ny; ++iy) {
                grid_current[ix*(ny+1) + iy] = 0;
                f[ix*(ny+1) + iy] = F(ix, iy);
            }
        }

        #pragma omp barrier

        for (it=0; it<num_iterations; ++it) {
            #pragma omp for
            for (ix=1; ix<nx; ++ix) {
                for (iy=1; iy<ny; ++iy) {
                    sum = 0;

                    sum -= grid_current[(ix-1)*(ny+1) + iy] / (hx*hx);
                    sum -= grid_current[(ix+1)*(ny+1) + iy] / (hx*hx);
                    sum -= grid_current[ix*(ny+1) + (iy-1)] / (hy*hy);
                    sum -= grid_current[ix*(ny+1) + (iy+1)] / (hy*hy);

                    grid_next[ix*(ny+1) + iy] = (f[ix*(ny+1) + iy] - sum) /
                                        (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                }
            }

            #pragma omp barrier

            #pragma omp master
            {
                tmp = grid_current;
                grid_current = grid_next;
                grid_next = tmp;
            }
        }
    }

    end_time = omp_get_wtime();

    residue = 0;
    for (ix=0; ix<(nx+1); ++ix) {
        for (iy=0; iy<(ny+1); ++iy) {
            if ((ix > 0) && (ix < nx) && (iy > 0) && (iy < ny)) {
                sum = 0;

                sum -= grid_current[(ix-1)*(ny+1) + iy] / (hx*hx);
                sum -= grid_current[(ix+1)*(ny+1) + iy] / (hx*hx);
                sum -= grid_current[ix*(ny+1) + (iy-1)] / (hy*hy);
                sum -= grid_current[ix*(ny+1) + (iy+1)] / (hy*hy);

                sum += grid_current[ix*(ny+1) + iy] * (2 / (hx*hx) + 2 / (hy*hy) + (K*K));

                residue += pow(F(ix, iy) - sum, 2);
            }

            fprintf(solution, "%lf %lf %lf\n", ix*hx, iy*hy, grid_current[ix*(ny+1) + iy]);
        }
        fprintf(solution, "\n");
    }

    printf("Tempo total de processamento: %lf segundos\n", end_time-begin_time);
    printf("Norma L2 do resíduo: %lf\n", sqrt(residue));

    free(grid_next);
    free(grid_current);
    free(f);
}

inline double sum_neighbours(double *grid, int i, int ix, int iy, double *border) {
    double sum = 0;

    if (ix % 2 == 0) {
        if (ix > 0)
            sum -= grid[i-(ny-1)] / (hx*hx);

        if (ix < nx-2)
            sum -= grid[i] / (hx*hx);
    }
    else {
        sum -= grid[i] / (hx*hx);

        if (ix < nx-2)
            sum -= grid[i+(ny-1)] / (hx*hx);
    }

    if (iy > 0)
        sum -= grid[i-1] / (hy*hy);

    if (iy < (ny-2))
        sum -= grid[i+1] / (hy*hy);
    else
        sum -= border[ix+1] / (hy*hy);

    return sum;
}

void gauss() {
    int ix, iy, bx, by;
    int i;
    double residue, residue_sum, begin_time, end_time, value;
    int size, size_x, size_y;
    int red;

    int *correction_x, *correction_y;

    v2df coef, hx_sq, hy_sq;
    coef[0] = (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
    coef[1] = (2 / (hx*hx) + 2 / (hy*hy) + (K*K));

    hx_sq[0] = hx * hx;
    hx_sq[1] = hx * hx;
    hy_sq[0] = hy * hy;
    hy_sq[1] = hy * hy;

    v2df *grid_red, *grid_black, *f_red, *f_black;
    double *grid_current;

    begin_time = omp_get_wtime();

    size_x = (int) ceil((nx-1) / 2.0) + 2;
    size_y = (int) ceil((ny-1) / 2.0) + 2;

    size = size_x * size_y;

    grid_red = (v2df*) malloc(size * sizeof(v2df));
    grid_black = (v2df*) malloc(size * sizeof(v2df));
    f_red = (v2df*) malloc(size * sizeof(v2df));
    f_black = (v2df*) malloc(size * sizeof(v2df));

    grid_current = (double*) malloc((nx+1) * (ny+1) * sizeof(double));

    correction_x = (int*) malloc(size_x * sizeof(int));
    correction_y = (int*) malloc(size_y * sizeof(int));

    #pragma omp parallel private(ix, iy, bx, by, i)
    {
        int it;
        v2df sumx, sumy;

        #pragma omp for
        for (bx = 1; bx < size_x-1; ++bx) {
            for (by = 1; by < size_y-1; ++by) {
                i = bx * size_y + by;

                if (by == 1) {
                    correction_x[bx] = 0;
                    if ((nx % 2 == 0) && (2*bx == nx))
                        correction_x[bx] = size_y;
                }

                correction_y[by] = 0;
                if ((ny % 2 == 0) && (2*by == ny))
                    correction_y[by] = 1;

                if (bx == 0) {
                    grid_red[i-size_y][0]   = 0;
                    grid_red[i-size_y][1]   = 0;
                    grid_black[i-size_y][0] = 0;
                    grid_black[i-size_y][1] = 0;
                }
                else if (bx == size_x-2) {
                    grid_red[i+size_y][0]   = 0;
                    grid_red[i+size_y][1]   = 0;
                    grid_black[i+size_y][0] = 0;
                    grid_black[i+size_y][1] = 0;
                }

                if (by == 0) {
                    grid_red[i-1][0]   = 0;
                    grid_red[i-1][1]   = 0;
                    grid_black[i-1][0] = 0;
                    grid_black[i-1][1] = 0;
                }
                else if (by == size_y-2) {
                    grid_red[i+1][0] = grid_black[i+1][1] = BORDER(2*bx-1);
                    grid_red[i+1][1] = grid_black[i+1][0] = BORDER(2*bx);
                }

                grid_red[i][0]   = 0;
                grid_red[i][1]   = 0;
                grid_black[i][0] = 0;
                grid_black[i][1] = 0;

                f_red[i][0]   = F(2*bx-1, 2*by-1);
                f_red[i][1]   = F(2*bx,   2*by);
                f_black[i][0] = F(2*bx,   2*by-1);
                f_black[i][1] = F(2*bx-1, 2*by);
            }
        }

        #pragma omp barrier

        for (it=0; it<num_iterations; ++it) {
            #pragma omp for
            for (bx = 1; bx < size_x-1; ++bx) {
                for (by = 1; by < size_y-1; ++by) {
                    i = bx * size_y + by;

                    sumx[0] = grid_black[i-size_y][0] + grid_black[i+correction_x[bx]][0];
                    sumx[1] = grid_black[i][1] + grid_black[i+size_y][1];
                    sumy[0] = grid_black[i-1][1] + grid_black[i+correction_y[by]][1];
                    sumy[1] = grid_black[i][0] + grid_black[i+1][0];

                    grid_red[i] = (f_red[i] + sumx / hx_sq + sumy / hy_sq) / coef;
                }
            }

            #pragma omp barrier

            #pragma omp for
            for (bx = 1; bx < size_x-1; ++bx) {
                for (by = 1; by < size_y-1; ++by) {
                    i = bx * size_y + by;

                    sumx[0] = grid_red[i][0] + grid_red[i+size_y][0];
                    sumx[1] = grid_red[i-size_y][1] + grid_red[i+correction_x[bx]][1];
                    sumy[0] = grid_red[i-1][1] + grid_red[i+correction_y[by]][1];
                    sumy[1] = grid_red[i][0] + grid_red[i+1][0];

                    grid_black[i] = (f_black[i] + sumx / hx_sq + sumy / hy_sq) / coef;
                }
            }

            #pragma omp barrier
        }
    }

    end_time = omp_get_wtime();

    for (bx = 1; bx < size_x-1; ++bx) {
        for (by = 1; by < size_y-1; ++by) {
            i = bx * size_y + by;

            grid_current[(2*bx-1)*(ny+1) + (2*by-1)] = grid_red[i][0];
            grid_current[(2*bx)*(ny+1)   + (2*by)  ] = grid_red[i][1];
            grid_current[(2*bx)*(ny+1)   + (2*by-1)] = grid_black[i][0];
            grid_current[(2*bx-1)*(ny+1) + (2*by)  ] = grid_black[i][1];
        }
    }

    for (ix=0; ix<(nx+1); ++ix) {
        grid_current[ix*(ny+1) + 0] = 0;
        grid_current[ix*(ny+1) + ny] = sin(2*M_PI*(ix*hx)) * sinh(2*M_PI);
    }
    for (iy=0; iy<(ny+1); ++iy) {
        grid_current[0*(ny+1) + iy] = 0;
        grid_current[nx*(ny+1) + iy] = 0;
    }

    residue = 0;
    for (ix=0; ix<(nx+1); ++ix) {
        for (iy=0; iy<(ny+1); ++iy) {
            if ((ix > 0) && (ix < nx) && (iy > 0) && (iy < ny)) {
                residue_sum = 0;

                residue_sum -= grid_current[(ix-1)*(ny+1) + iy] / (hx*hx);
                residue_sum -= grid_current[(ix+1)*(ny+1) + iy] / (hx*hx);
                residue_sum -= grid_current[ix*(ny+1) + (iy-1)] / (hy*hy);
                residue_sum -= grid_current[ix*(ny+1) + (iy+1)] / (hy*hy);

                residue_sum += grid_current[ix*(ny+1) + iy] * (2 / (hx*hx) + 2 / (hy*hy) + (K*K));

                residue += pow(F(ix, iy) - residue_sum, 2);
            }

            fprintf(solution, "%lf %lf %lf\n", ix*hx, iy*hy, grid_current[ix*(ny+1) + iy]);
        }
        fprintf(solution, "\n");
    }

    printf("Tempo total de processamento: %lf segundos\n", end_time-begin_time);
    printf("Norma L2 do resíduo: %lf\n", sqrt(residue));

    free(grid_red);
    free(grid_black);
    free(f_red);
    free(f_black);
    free(grid_current);
}

int main(int argc, char **argv) {
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

    if (num_iterations < 0) {
        fprintf(stderr, "error: %d is not a valid number of iterations\n", num_iterations);
        exit(1);
    }

    if ((solution = fopen("solution.txt", "w")) == NULL) {
        fprintf(stderr, "error: failed to open 'solution.txt' for writing\n");
        exit(1);
    }

    omp_set_num_threads(num_threads);

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
        exit(1);
    }

    fclose(solution);
    exit(0);
}
