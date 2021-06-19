#include <mpi.h>
#include <iostream>
#include <stdexcept>
#include <cmath>

static const double T = 0.1;
static const double dt = 0.0002;
static const double h = 0.02;
static const size_t num_iterations = T / dt;

double calculate_temperature(double u_left, double u, double u_right) {
    return dt / h / h * (u_left - 2 * u + u_right) + u;
}

double term(size_t m, double x, double t) {
    return exp(-M_PI * M_PI * pow(2 * m + 1, 2) * t) / (2 * m + 1) * sin(M_PI * (2 * m + 1) * x);
}

double precise_solution(double x) {
    double res = 0;
    for (size_t i = 0; i < 10; ++i) {
        res += term(i, x, T);
    }
    return res * 4 / M_PI;
}

int main(int argc, char** argv) {
	MPI_Init(NULL, NULL);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (world_size < 11) {
	    throw std::invalid_argument("wrong amount of workers");
	}

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	double u_left, u = 1.0, u_right;
	if (world_rank == 0) {
	    u_left = 0.0;

        for (size_t i = 0; i < num_iterations; ++i) {
            MPI_Sendrecv(
                &u, 1, MPI_DOUBLE, world_rank + 1, 0,
                &u_right, 1, MPI_DOUBLE, world_rank + 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

            u = calculate_temperature(u_left, u, u_right);
        }

	} else if (0 < world_rank && world_rank < 10) {
        for (size_t i = 0; i < num_iterations; ++i) {
            MPI_Sendrecv(
                &u, 1, MPI_DOUBLE, world_rank - 1, 0,
                &u_left, 1, MPI_DOUBLE, world_rank - 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
            MPI_Sendrecv(
                &u, 1, MPI_DOUBLE, world_rank + 1, 0,
                &u_right, 1, MPI_DOUBLE, world_rank + 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

            u = calculate_temperature(u_left, u, u_right);
        }
	} else if (world_rank == 10) {
	    u_right = 0.0;

	    for (size_t i = 0; i < num_iterations; ++i) {
            MPI_Sendrecv(
                &u, 1, MPI_DOUBLE, world_rank - 1, 0,
                &u_left, 1, MPI_DOUBLE, world_rank - 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

            u = calculate_temperature(u_left, u, u_right);
        }
	}

	std::cout.precision(17);
	std::cout << std::fixed << "point " << world_rank << " temperature is: " << u << '\n';
	std::cout << std::fixed << "point " << world_rank << " precise temperature is: " << precise_solution(world_rank * 0.1) << '\n';

	MPI_Finalize();
	return 0;
}