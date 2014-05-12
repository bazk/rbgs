#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define USAGE "Usage:\n\trbgs nx ny t c [j|g]\n\nWhere:\n \
 nx,ny\tnumber of discretization intervals in x and y axis, repectively\n \
 t\tnumber of threads\n \
 c\tnumber of iterations\n \
 j|q\tmethod to be used (j for Jacobi, g for Gaussian)\n\n"

typedef enum { JACOBI, GAUSS } method_t;

int main(int argc, char **argv) {
    int nx, ny, num_threads, num_iterations;
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

    // open solution.txt file to write the resulting solution
    if ((solution = fopen("solution.txt", "w")) == NULL) {
        fprintf(stderr, "error: failed to open 'solution.txt' for writing\n");
        exit(1);
    }

    // cleanup
    fclose(solution);
    exit(0);
}
