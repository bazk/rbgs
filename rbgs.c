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

#define INCREMENT(A, B) if (++B == ny-1) { A += 2; B = 0; } \
                        else { A += (A % 2 == 0) ? 1 : -1; }

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
                f[ix*(ny+1) + iy] = F(ix*hx, iy*hy);
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

                residue += pow(F(ix*hx, iy*hy) - sum, 2);
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
    int ix, iy;
    double residue, residue_sum, begin_time, end_time, value;
    int size;
    int red;

    double *grid_red, *grid_black, *f_red, *f_black;
    double *border;

    begin_time = omp_get_wtime();

    size = ceil((nx-1) / 2.0) * (ny-1);

    grid_red = (double*) malloc(size * sizeof(double));
    grid_black = (double*) malloc(size * sizeof(double));
    f_red = (double*) malloc(size * sizeof(double));
    f_black = (double*) malloc(size * sizeof(double));

    border = (double*) malloc((nx+1) * sizeof(double));

    for (ix=0; ix<(nx+1); ix++)
        border[ix] = sin(2*M_PI*(ix*hx)) * sinh(2*M_PI);

    #pragma omp parallel private(ix, iy)
    {
        int it, i, j, from, to, chunk;
        int from_ix_red, from_ix_black, from_iy;
        double sum[4];
        // double sum;

        chunk = ceil(size / (float) omp_get_num_threads());
        from = chunk * omp_get_thread_num();
        to = from + chunk;
        if (to > size)
            to = size;

        ix = (int) from / (ny-1);
        from_ix_red = ix * 2;
        from_ix_black = ix * 2;

        if (ix % 2 == 0) {
            if (from % 2 == 0)
                from_ix_black += 1;
            else
                from_ix_red += 1;
        }
        else {
            if (from % 2 == 0)
                from_ix_red += 1;
            else
                from_ix_black += 1;
        }

        from_iy = from % (ny-1);

        ix = from_ix_red;
        iy = from_iy;
        for (i=from; i<to; ++i) {
            f_red[i] = F((ix+1)*hx, (iy+1)*hy);
            INCREMENT(ix, iy);
        }

        ix = from_ix_black;
        iy = from_iy;
        for (i=from; i<to; ++i) {
            f_black[i] = F((ix+1)*hx, (iy+1)*hy);
            INCREMENT(ix, iy);
        }

        #pragma omp barrier

        for (it=0; it<num_iterations; ++it) {
            ix = from_ix_red;
            iy = from_iy;
            i = from;
            while (i<to) {
                sum[0] = sum_neighbours(grid_black, i, ix, iy, border);
                INCREMENT(ix, iy);

                sum[1] = sum_neighbours(grid_black, i+1, ix, iy, border);
                INCREMENT(ix, iy);

                sum[2] = sum_neighbours(grid_black, i+2, ix, iy, border);
                INCREMENT(ix, iy);

                sum[3] = sum_neighbours(grid_black, i+3, ix, iy, border);
                INCREMENT(ix, iy);

                grid_red[i] = (f_red[i] - sum[0]) /
                        (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                grid_red[i+1] = (f_red[i+1] - sum[1]) /
                        (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                grid_red[i+2] = (f_red[i+2] - sum[2]) /
                        (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                grid_red[i+3] = (f_red[i+3] - sum[3]) /
                        (2 / (hx*hx) + 2 / (hy*hy) + (K*K));

                i += 4;
            }

            #pragma omp barrier

            ix = from_ix_black;
            iy = from_iy;
            i = from;
            while (i<to) {
                sum[0] = sum_neighbours(grid_red, i, ix, iy, border);
                INCREMENT(ix, iy);

                sum[1] = sum_neighbours(grid_red, i+1, ix, iy, border);
                INCREMENT(ix, iy);

                sum[2] = sum_neighbours(grid_red, i+2, ix, iy, border);
                INCREMENT(ix, iy);

                sum[3] = sum_neighbours(grid_red, i+3, ix, iy, border);
                INCREMENT(ix, iy);

                grid_black[i] = (f_black[i] - sum[0]) /
                        (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                grid_black[i+1] = (f_black[i+1] - sum[1]) /
                        (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                grid_black[i+2] = (f_black[i+2] - sum[2]) /
                        (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                grid_black[i+3] = (f_black[i+3] - sum[3]) /
                        (2 / (hx*hx) + 2 / (hy*hy) + (K*K));

                i += 4;
            }

            #pragma omp barrier
        }
    }

    end_time = omp_get_wtime();

    residue = 0;
    red = 1;
    for (ix=0; ix<=nx; ++ix) {
        for (iy=0; iy<=ny; ++iy) {
            if ((ix == 0) || (iy == 0) || (ix == nx))
                value = 0;
            else if (iy == ny)
                value = border[ix];
            else if (red)
                value = grid_red[L(ix-1, iy-1)];
            else
                value = grid_black[L(ix-1, iy-1)];

            if ((ix > 0) && (ix < nx) && (iy > 0) && (iy < ny)) {
                if (red) {
                    residue_sum = sum_neighbours(grid_black, L(ix-1, iy-1), ix-1, iy-1, border);
                    residue_sum += grid_red[L(ix-1, iy-1)] * (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                }
                else {
                    residue_sum = sum_neighbours(grid_red, L(ix-1, iy-1), ix-1, iy-1, border);
                    residue_sum += grid_black[L(ix-1, iy-1)] * (2 / (hx*hx) + 2 / (hy*hy) + (K*K));
                }

                residue += pow(F(ix*hx, iy*hy) - residue_sum, 2);
            }

            red = !red;

            fprintf(solution, "%lf %lf %lf\n", ix*hx, iy*hy, value);
        }
        fprintf(solution, "\n");
    }

    printf("Tempo total de processamento: %lf segundos\n", end_time-begin_time);
    printf("Norma L2 do resíduo: %lf\n", sqrt(residue));

    free(grid_red);
    free(grid_black);
    free(f_red);
    free(f_black);
    free(border);
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
