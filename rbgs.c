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
#define BORDER(X) sin(2*M_PI*((X)*hx)) * sinh(2*M_PI)
#define IDX(X, Y) (X) * size_y + (Y)
#define IDX2(X, Y) ((int) ((X) / 2.0)) * size_y + (Y)

#ifndef PADDING
#define PADDING 16 // array padding to avoid cache thrashing
#endif

typedef double double2 __attribute__ ((vector_size (16)));

int nx, ny, num_threads, num_iterations;
double hx, hy;
FILE *solution;

void jacobi() {
    int it, x, y;
    int size_x, size_y;
    double *grid_current, *grid_next, *f, *tmp;
    double sum, residue, begin_time, end_time;

    size_x = nx+1;
    size_y = ny+1;

    grid_current = (double*) malloc(size_x * size_y * sizeof(double));
    grid_next = (double*) malloc(size_x * size_y * sizeof(double));
    f = (double*) malloc(size_x * size_y * sizeof(double));

    // initialize grid borders
    for (x=0; x<size_x; ++x) {
        grid_current[IDX(x, 0)] = grid_next[IDX(x, 0)] = 0;
        grid_current[IDX(x, ny)] = grid_next[IDX(x, ny)] = BORDER(x);
    }
    for (y=0; y<size_y; ++y) {
        grid_current[IDX(0, y)] = grid_next[IDX(0, y)] = 0;
        grid_current[IDX(nx, y)] = grid_next[IDX(nx, y)] = 0;
    }

    begin_time = omp_get_wtime();

    #pragma omp parallel private(it, x, y, sum)
    {
        // initialize grid and f
        #pragma omp for
        for (x=1; x<nx; ++x) {
            for (y=1; y<ny; ++y) {
                grid_current[IDX(x, y)] = 0;
                f[IDX(x, y)] = F(x, y);
            }
        }

        #pragma omp barrier

        for (it=0; it<num_iterations; ++it) {
            #pragma omp for
            for (x=1; x<nx; ++x) {
                for (y=1; y<ny; ++y) {
                    sum = 0;

                    sum -= grid_current[IDX(x-1, y)] / (hx*hx);
                    sum -= grid_current[IDX(x+1, y)] / (hx*hx);
                    sum -= grid_current[IDX(x, y-1)] / (hy*hy);
                    sum -= grid_current[IDX(x, y+1)] / (hy*hy);

                    grid_next[IDX(x, y)] = (f[IDX(x, y)] - sum) /
                                        (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                }
            }

            #pragma omp barrier

            // swap current grid with next grid
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
    for (x=0; x<size_x; ++x) {
        for (y=0; y<size_y; ++y) {
            if ((x > 0) && (x < nx) && (y > 0) && (y < ny)) {
                sum = 0;

                sum -= grid_current[IDX(x-1, y)] / (hx*hx);
                sum -= grid_current[IDX(x+1, y)] / (hx*hx);
                sum -= grid_current[IDX(x, y-1)] / (hy*hy);
                sum -= grid_current[IDX(x, y+1)] / (hy*hy);

                sum += grid_current[IDX(x, y)] * (2 / (hx*hx) + 2 / (hy*hy) + (K*K));

                residue += pow(F(x, y) - sum, 2);
            }

            fprintf(solution, "%lf %lf %lf\n", x*hx, y*hy, grid_current[IDX(x, y)]);
        }
        fprintf(solution, "\n");
    }

    printf("Tempo total de processamento: %lf segundos\n", end_time-begin_time);
    printf("Norma L2 do resíduo: %lf\n", sqrt(residue));

    free(grid_next);
    free(grid_current);
    free(f);
}

void gauss_v1() {
    int it, x, y, yy;
    int size_x, size_y;
    double residue, sum;
    double begin_time, end_time;
    double *f;
    double *grid_current;

    size_x = nx+1;
    size_y = ny+1;

    grid_current = (double*) malloc(size_x * size_y * sizeof(double));
    f = (double*) malloc(size_x * size_y * sizeof(double));

    for (x=0; x<(nx+1); ++x) {
        grid_current[IDX(x, 0)] = 0;
        grid_current[IDX(x, ny)] = BORDER(x);
    }
    for (y=0; y<(ny+1); ++y) {
        grid_current[IDX(0, y)] = 0;
        grid_current[IDX(nx, y)] = 0;
    }

    begin_time = omp_get_wtime();

    #pragma omp parallel private(it, x, y, yy, sum)
    {
        #pragma omp for
        for (x=1; x<nx; ++x) {
            for (yy=1; yy<ny; yy+=2) {
                if (x % 2 == 0)
                    y = yy;
                else
                    y = yy + 1;

                if (y >= ny)
                    continue;

                grid_current[IDX(x, y)] = 0;
                f[IDX(x, y)] = F(x, y);
            }
        }

        #pragma omp for
        for (x=1; x<nx; ++x) {
            for (yy=1; yy<ny; yy+=2) {
                if (x % 2 == 0)
                    y = yy + 1;
                else
                    y = yy;

                if (y >= ny)
                    continue;

                grid_current[IDX(x, y)] = 0;
                f[IDX(x, y)] = F(x, y);
            }
        }

        #pragma omp barrier

        for (it=0; it<num_iterations; ++it) {
            #pragma omp for
            for (x=1; x<nx; ++x) {
                for (yy=1; yy<ny; yy+=2) {
                    if (x % 2 == 0)
                        y = yy;
                    else
                        y = yy + 1;

                    if (y >= ny)
                        continue;

                    sum = 0;

                    sum -= grid_current[IDX(x-1, y)] / (hx*hx);
                    sum -= grid_current[IDX(x+1, y)] / (hx*hx);
                    sum -= grid_current[IDX(x, y-1)] / (hy*hy);
                    sum -= grid_current[IDX(x, y+1)] / (hy*hy);

                    grid_current[IDX(x, y)] = (f[IDX(x, y)] - sum) /
                                        (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                }
            }

            #pragma omp barrier

            #pragma omp for
            for (x=1; x<nx; ++x) {
                for (yy=1; yy<ny; yy+=2) {
                    if (x % 2 == 0)
                        y = yy + 1;
                    else
                        y = yy;

                    if (y >= ny)
                        continue;

                    sum = 0;

                    sum -= grid_current[IDX(x-1, y)] / (hx*hx);
                    sum -= grid_current[IDX(x+1, y)] / (hx*hx);
                    sum -= grid_current[IDX(x, y-1)] / (hy*hy);
                    sum -= grid_current[IDX(x, y+1)] / (hy*hy);

                    grid_current[IDX(x, y)] = (f[IDX(x, y)] - sum) /
                                        (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                }
            }

            #pragma omp barrier
        }
    }

    end_time = omp_get_wtime();

    residue = 0;
    for (x=0; x<(nx+1); ++x) {
        for (y=0; y<(ny+1); ++y) {
            if ((x > 0) && (x < nx) && (y > 0) && (y < ny)) {
                sum = 0;

                sum -= grid_current[(x-1)*(ny+1) + y] / (hx*hx);
                sum -= grid_current[(x+1)*(ny+1) + y] / (hx*hx);
                sum -= grid_current[x*(ny+1) + (y-1)] / (hy*hy);
                sum -= grid_current[x*(ny+1) + (y+1)] / (hy*hy);

                sum += grid_current[x*(ny+1) + y] * (2 / (hx*hx) + 2 / (hy*hy) + (K*K));

                residue += pow(F(x, y) - sum, 2);
            }

            fprintf(solution, "%lf %lf %lf\n", x*hx, y*hy, grid_current[x*(ny+1) + y]);
        }
        fprintf(solution, "\n");
    }

    printf("Tempo total de processamento: %lf segundos\n", end_time-begin_time);
    printf("Norma L2 do resíduo: %lf\n", sqrt(residue));

    free(grid_current);
    free(f);
}

void gauss_v2() {
    int it, x, y, xx, idx;
    int size_x, size_y;
    double residue, sum;
    double begin_time, end_time;
    double *grid_red, *grid_black, *f_red, *f_black;
    double *grid_current;

    size_x = (int) ceil((nx+1) / 2.0);
    size_y = ny+1;

    grid_red = (double*) malloc(size_x * size_y * sizeof(double));
    grid_black = (double*) malloc(size_x * size_y * sizeof(double));
    f_red = (double*) malloc(size_x * size_y * sizeof(double));
    f_black = (double*) malloc(size_x * size_y * sizeof(double));

    begin_time = omp_get_wtime();

    #pragma omp parallel private(it, x, y, xx, idx, sum)
    {
        #pragma omp for
        for (xx=1; xx<nx; xx+=2) {
            for (y=1; y<ny; ++y) {
                if (y % 2 == 0)
                    x = xx + 1;
                else
                    x = xx;

                if (x >= nx)
                    continue;

                if (x == 1) {
                    grid_black[IDX2(x-1, y)] = 0;
                }
                else if (x == nx-1) {
                    grid_black[IDX2(x+1, y)] = 0;
                }

                if (y == 1) {
                    grid_black[IDX2(x, y-1)] = 0;
                }
                else if (y == ny-1) {
                    grid_black[IDX2(x, y+1)] = BORDER(x);
                }

                idx = IDX2(x, y);

                grid_red[idx] = 0;
                f_red[idx] = F(x, y);
            }
        }

        #pragma omp for
        for (xx=1; xx<nx; xx+=2) {
            for (y=1; y<ny; ++y) {
                if (y % 2 == 0)
                    x = xx;
                else
                    x = xx + 1;

                if (x >= nx)
                    continue;

                if (x == 1) {
                    grid_red[IDX2(x-1, y)] = 0;
                }
                else if (x == nx-1) {
                    grid_red[IDX2(x+1, y)] = 0;
                }

                if (y == 1) {
                    grid_red[IDX2(x, y-1)] = 0;
                }
                else if (y == ny-1) {
                    grid_red[IDX2(x, y+1)] = BORDER(x);
                }

                idx = IDX2(x, y);

                grid_black[idx] = 0;
                f_black[idx] = F(x, y);
            }
        }

        #pragma omp barrier

        for (it=0; it<num_iterations; ++it) {
            #pragma omp for
            for (xx=1; xx<nx; xx+=2) {
                for (y=1; y<ny; ++y) {
                    if (y % 2 == 0)
                        x = xx + 1;
                    else
                        x = xx;

                    if (x >= nx)
                        continue;

                    idx = IDX2(x, y);

                    sum = 0;

                    if (x % 2 == 0) {
                        sum -= grid_black[idx-size_y] / (hx*hx);
                        sum -= grid_black[idx] / (hx*hx);
                    }
                    else {
                        sum -= grid_black[idx] / (hx*hx);
                        sum -= grid_black[idx+size_y] / (hx*hx);
                    }

                    sum -= grid_black[idx-1] / (hy*hy);
                    sum -= grid_black[idx+1] / (hy*hy);

                    grid_red[idx] = (f_red[idx] - sum) /
                        (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                }
            }

            #pragma omp barrier

            #pragma omp for
            for (xx=1; xx<nx; xx+=2) {
                for (y=1; y<ny; ++y) {
                    if (y % 2 == 0)
                        x = xx;
                    else
                        x = xx + 1;

                    if (x >= nx)
                        continue;

                    idx = IDX2(x, y);

                    sum = 0;

                    if (x % 2 == 0) {
                        sum -= grid_red[idx-size_y] / (hx*hx);
                        sum -= grid_red[idx] / (hx*hx);
                    }
                    else {
                        sum -= grid_red[idx] / (hx*hx);
                        sum -= grid_red[idx+size_y] / (hx*hx);
                    }

                    sum -= grid_red[idx-1] / (hy*hy);
                    sum -= grid_red[idx+1] / (hy*hy);

                    grid_black[idx] = (f_black[idx] - sum) /
                        (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                }
            }

            #pragma omp barrier
        }
    }

    end_time = omp_get_wtime();

    grid_current = (double*) malloc((nx+1) * (ny+1) * sizeof(double));

    for (xx=0; xx<nx+1; xx+=2) {
        for (y=0; y<ny+1; ++y) {
            if (y % 2 == 0)
                x = xx;
            else
                x = xx + 1;

            if (x <= nx)
                grid_current[IDX(x, y)] = grid_red[IDX2(x, y)];

            if (y % 2 == 0)
                x = xx + 1;
            else
                x = xx;

            if (x <= nx)
                grid_current[IDX(x, y)] = grid_black[IDX2(x, y)];
        }
    }

    residue = 0;
    for (x=0; x<(nx+1); ++x) {
        for (y=0; y<(ny+1); ++y) {
            if ((x > 0) && (x < nx) && (y > 0) && (y < ny)) {
                sum = 0;

                sum -= grid_current[(x-1)*(ny+1) + y] / (hx*hx);
                sum -= grid_current[(x+1)*(ny+1) + y] / (hx*hx);
                sum -= grid_current[x*(ny+1) + (y-1)] / (hy*hy);
                sum -= grid_current[x*(ny+1) + (y+1)] / (hy*hy);

                sum += grid_current[x*(ny+1) + y] * (2 / (hx*hx) + 2 / (hy*hy) + (K*K));

                residue += pow(F(x, y) - sum, 2);
            }

            fprintf(solution, "%lf %lf %lf\n", x*hx, y*hy, grid_current[x*(ny+1) + y]);
        }
        fprintf(solution, "\n");
    }

    printf("Tempo total de processamento: %lf segundos\n", end_time-begin_time);
    printf("Norma L2 do resíduo: %lf\n", sqrt(residue));

    free(grid_current);
    free(grid_red);
    free(grid_black);
    free(f_red);
    free(f_black);
}

void gauss_v3() {
    int it, x, y, bx, by, idx;
    int size_x, size_y;
    double residue, sum;
    double begin_time, end_time;
    double2 *grid_red, *grid_black, *f_red, *f_black;
    double2 coef, hx_sq, hy_sq;
    double2 sum_x, sum_y;
    int *correction_x, *correction_y;
    double *grid_current;

    coef[0] = (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
    coef[1] = (2 / (hx*hx) + 2 / (hy*hy) + (K*K));

    hx_sq[0] = hx * hx;
    hx_sq[1] = hx * hx;
    hy_sq[0] = hy * hy;
    hy_sq[1] = hy * hy;

    size_x = (int) ceil((nx-1) / 2.0) + 2;
    size_y = (int) ceil((ny-1) / 2.0) + 2 + PADDING;

    grid_red = (double2*) malloc(size_x * size_y * sizeof(double2));
    grid_black = (double2*) malloc(size_x * size_y * sizeof(double2));
    f_red = (double2*) malloc(size_x * size_y * sizeof(double2));
    f_black = (double2*) malloc(size_x * size_y * sizeof(double2));

    correction_x = (int*) malloc(size_x * sizeof(int));
    correction_y = (int*) malloc(size_y * sizeof(int));

    begin_time = omp_get_wtime();

    #pragma omp parallel private(it, bx, by, idx, sum_x, sum_y)
    {
        #pragma omp for
        for (bx = 1; bx < size_x-1; ++bx) {
            for (by = 1; by < size_y-1-PADDING; ++by) {
                idx = bx * size_y + by;

                if (by == 1) {
                    correction_x[bx] = 0;
                    if ((nx % 2 == 0) && (2*bx == nx))
                        correction_x[bx] = size_y;
                }

                correction_y[by] = 0;
                if ((ny % 2 == 0) && (2*by == ny))
                    correction_y[by] = 1;

                if (bx == 0) {
                    grid_red[idx-size_y][0]   = 0;
                    grid_red[idx-size_y][1]   = 0;
                    grid_black[idx-size_y][0] = 0;
                    grid_black[idx-size_y][1] = 0;
                }
                else if (bx == size_x-2) {
                    grid_red[idx+size_y][0]   = 0;
                    grid_red[idx+size_y][1]   = 0;
                    grid_black[idx+size_y][0] = 0;
                    grid_black[idx+size_y][1] = 0;
                }

                if (by == 0) {
                    grid_red[idx-1][0]   = 0;
                    grid_red[idx-1][1]   = 0;
                    grid_black[idx-1][0] = 0;
                    grid_black[idx-1][1] = 0;
                }
                else if (by == size_y-2-PADDING) {
                    grid_red[idx+1][0] = grid_black[idx+1][1] = BORDER(2*bx-1);
                    grid_red[idx+1][1] = grid_black[idx+1][0] = BORDER(2*bx);
                }

                grid_red[idx][0]   = 0;
                grid_red[idx][1]   = 0;
                grid_black[idx][0] = 0;
                grid_black[idx][1] = 0;

                f_red[idx][0]   = F(2*bx-1, 2*by-1);
                f_red[idx][1]   = F(2*bx,   2*by);
                f_black[idx][0] = F(2*bx,   2*by-1);
                f_black[idx][1] = F(2*bx-1, 2*by);
            }
        }

        #pragma omp barrier

        for (it=0; it<num_iterations; ++it) {
            #pragma omp for
            for (bx = 1; bx < size_x-1; ++bx) {
                for (by = 1; by < size_y-1-PADDING; ++by) {
                    idx = bx * size_y + by;

                    sum_x[0] = grid_black[idx-size_y][0] + grid_black[idx+correction_x[bx]][0];
                    sum_x[1] = grid_black[idx][1] + grid_black[idx+size_y][1];
                    sum_y[0] = grid_black[idx-1][1] + grid_black[idx+correction_y[by]][1];
                    sum_y[1] = grid_black[idx][0] + grid_black[idx+1][0];

                    grid_red[idx] = (f_red[idx] + sum_x / hx_sq + sum_y / hy_sq) / coef;
                }
            }

            #pragma omp barrier

            #pragma omp for
            for (bx = 1; bx < size_x-1; ++bx) {
                for (by = 1; by < size_y-1-PADDING; ++by) {
                    idx = bx * size_y + by;

                    sum_x[0] = grid_red[idx][0] + grid_red[idx+size_y][0];
                    sum_x[1] = grid_red[idx-size_y][1] + grid_red[idx+correction_x[bx]][1];
                    sum_y[0] = grid_red[idx-1][1] + grid_red[idx+correction_y[by]][1];
                    sum_y[1] = grid_red[idx][0] + grid_red[idx+1][0];

                    grid_black[idx] = (f_black[idx] + sum_x / hx_sq + sum_y / hy_sq) / coef;
                }
            }

            #pragma omp barrier
        }
    }

    end_time = omp_get_wtime();

    grid_current = (double*) malloc((nx+1) * (ny+1) * sizeof(double));

    for (bx = 1; bx < size_x-1; ++bx) {
        for (by = 1; by < size_y-1-PADDING; ++by) {
            idx = bx * size_y + by;

            grid_current[(2*bx-1)*(ny+1) + (2*by-1)] = grid_red[idx][0];
            grid_current[(2*bx)*(ny+1)   + (2*by)  ] = grid_red[idx][1];
            grid_current[(2*bx)*(ny+1)   + (2*by-1)] = grid_black[idx][0];
            grid_current[(2*bx-1)*(ny+1) + (2*by)  ] = grid_black[idx][1];
        }
    }

    for (x=0; x<(nx+1); ++x) {
        grid_current[x*(ny+1) + 0] = 0;
        grid_current[x*(ny+1) + ny] = BORDER(x);
    }
    for (y=0; y<(ny+1); ++y) {
        grid_current[0*(ny+1) + y] = 0;
        grid_current[nx*(ny+1) + y] = 0;
    }

    residue = 0;
    for (x=0; x<(nx+1); ++x) {
        for (y=0; y<(ny+1); ++y) {
            if ((x > 0) && (x < nx) && (y > 0) && (y < ny)) {
                sum = 0;

                sum -= grid_current[(x-1)*(ny+1) + y] / (hx*hx);
                sum -= grid_current[(x+1)*(ny+1) + y] / (hx*hx);
                sum -= grid_current[x*(ny+1) + (y-1)] / (hy*hy);
                sum -= grid_current[x*(ny+1) + (y+1)] / (hy*hy);

                sum += grid_current[x*(ny+1) + y] * (2 / (hx*hx) + 2 / (hy*hy) + (K*K));

                residue += pow(F(x, y) - sum, 2);
            }

            fprintf(solution, "%lf %lf %lf\n", x*hx, y*hy, grid_current[x*(ny+1) + y]);
        }
        fprintf(solution, "\n");
    }

    printf("Tempo total de processamento: %lf segundos\n", end_time-begin_time);
    printf("Norma L2 do resíduo: %lf\n", sqrt(residue));

    free(grid_current);
    free(grid_red);
    free(grid_black);
    free(f_red);
    free(f_black);
    free(correction_x);
    free(correction_y);
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
        // gauss_v1();
        // gauss_v2();
        gauss_v3();
        break;
    default:
        fprintf(stderr, "error: '%s' is not a valid calculation method (valid options are 'j' for Jacobi or 'g' for Gauss-Seidel)\n", argv[5]);
        exit(1);
    }

    fclose(solution);
    exit(0);
}
