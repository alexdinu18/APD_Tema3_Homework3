#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

#define NUM_COLORS 256

// structure for complex numbers
struct Complex{
	double re;
	double im;
};

//operations on complex numbers
double complex_modulus(struct Complex z);
struct Complex add_complex(struct Complex a, struct Complex b);
struct Complex multiply_complex(struct Complex a, struct Complex b);

int main(int argc, char** argv) {
	
	int rank, size, tag = 1;
	
	MPI_Status Stat;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	int type, MAX_STEPS;
	double x_min, x_max, y_min, y_max, resolution;
	struct Complex cj; // contant complex number used for Julia sets

	
	if (rank == 0) {
		// root process
		if (argc != 3) {
			printf("Wrong number of arguments!\n");
			exit(1);
		}
	
		FILE* fin = fopen(argv[1], "r");

	
		if (fin != NULL) {
			// reading from input file
			fscanf(fin, "%d", &type);
			fscanf(fin, "%lf", &x_min);
			fscanf(fin, "%lf", &x_max);
			fscanf(fin, "%lf", &y_min);
			fscanf(fin, "%lf", &y_max);
			fscanf(fin, "%lf", &resolution);
			fscanf(fin, "%d", &MAX_STEPS);
			if (type == 1) {
				fscanf(fin, "%lf %lf", &cj.re, &cj.im);
			}
		}
		else {
			printf("Error opening file!\n");
			exit(1);
		}
		fclose(fin);
	}
	
	// data from input file is being broadcast to all the other processes 
	MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&x_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&x_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&y_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&y_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&resolution, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&MAX_STEPS, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (type == 1) {
		MPI_Bcast (&cj.re, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast (&cj.im, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	
	if (!type) {
		// mandelbrot

		struct Complex z;
		int step, color;
		int i,j;
		
		// computing the size of the image
		int W = (int)(fabs((x_max-x_min) / resolution));
		int H = (int)(fabs((y_max-y_min) / resolution));
		
		// each process will compute a smaller part of the image
		int chunksize = H / size;
		int** partial_matrix = (int**)malloc(chunksize * sizeof(int*));
		for (i = 0; i < chunksize; i++)
			partial_matrix[i] = (int*)malloc(W * sizeof(int));
		
		// the algorithm for Mandelbrot Set
		for (i = rank * chunksize; i < (rank + 1) * chunksize; i ++) {
			for (j = 0; j < W; j++) {				
				z.re = 0;
				z.im = 0;
				step = 0;
				while(complex_modulus(z) < 2 && step < MAX_STEPS) {
					struct Complex c;
					c.re = x_min + j * resolution;
					c.im = y_min + i * resolution;
					z = add_complex(multiply_complex(z, z), c);
					step++;
				}
				color = step % NUM_COLORS;
				partial_matrix[i - rank * chunksize][j] = color;
			}
		}
		
		if (rank) {
			// if the current process is not the root,
			// then send the partial_matrix to the root
			for (i = 0; i < chunksize; i++) {
				MPI_Send(partial_matrix[i], W, MPI_INT, 0, tag, MPI_COMM_WORLD);
				free(partial_matrix[i]);
			}
			free(partial_matrix);
		}
		else {
			// root process
			
			// pgm = the matrix used to gather all the partial matrices
			int** pgm = (int**)malloc(H * sizeof(int*));
			for (i = 0; i < H; i++)
				pgm[i] = (int*)malloc(W * sizeof(int));
			
			// that's the part computed by the root process
			for (i = 0; i < chunksize; i++) {
				for (j = 0; j < W; j++)
					pgm[i][j] = partial_matrix[i][j];

				free(partial_matrix[i]);
			}
			free(partial_matrix);
			
			// the root receives the partial matrices
			for(i = 1; i < size; i++)
				for (j = 0; j < chunksize; j++)
					MPI_Recv(pgm[i * chunksize + j], W, MPI_INT, i, tag, MPI_COMM_WORLD, &Stat);
			
			FILE* fout = fopen(argv[2], "w");
			
			fprintf(fout, "P2\n");
			fprintf(fout, "%d %d\n", W, H);
			fprintf(fout, "%d\n", NUM_COLORS - 1);
			
			// write the final matrix to file
			for (i = H - 1; i >= 0; i--) {
				for (j = 0; j < W; j++) {
					fprintf(fout, "%d ", pgm[i][j]);
				}
				fprintf(fout, "\n");
				free(pgm[i]);
			}
			free(pgm);
			
			fclose(fout);
		}
	}
	
	else {
		// julia

		struct Complex z;
		int step, color;
		int i,j;
		
		// computing the size of the image
		int W = (int)(fabs((x_max-x_min) / resolution));
		int H = (int)(fabs((y_max-y_min) / resolution));
		
		// each process will compute a smaller part of the image
		int chunksize = H / size;
		int** partial_matrix = (int**)malloc(chunksize * sizeof(int*));
		for (i = 0; i < chunksize; i++)
			partial_matrix[i] = (int*)malloc(W * sizeof(int));
		
		// the algorithm for Julia Sets
		for (i = rank * chunksize; i < (rank + 1) * chunksize; i ++) {
			for (j = 0; j < W; j++) {				
				z.re = x_min + j*resolution;
				z.im = y_min + i*resolution;
				step = 0;
				while(complex_modulus(z) < 2 && step < MAX_STEPS) {
					z = add_complex(multiply_complex(z, z), cj);
					step++;
				}
				color = step % NUM_COLORS;
				partial_matrix[i - rank * chunksize][j] = color;
			}
		}
		
		if (rank) {
			// if the current process is not the root,
			// then send the partial_matrix to the root
			for (i = 0; i < chunksize; i++) {
				MPI_Send(partial_matrix[i], W, MPI_INT, 0, tag, MPI_COMM_WORLD);
				free(partial_matrix[i]);
			}
			free(partial_matrix);
		}
		else {
			// root process
			
			// pgm = the matrix used to gather all the partial matrices
			int** pgm = (int**)malloc(H * sizeof(int*));
			for (i = 0; i < H; i++)
				pgm[i] = (int*)malloc(W * sizeof(int));
				
			// that's the part computed by the root process
			for (i = 0; i < chunksize; i++) {				
				for (j = 0; j < W; j++)
					pgm[i][j] = partial_matrix[i][j];
					
				free(partial_matrix[i]);
			}
			free(partial_matrix);
			
			// the root receives the partial matrices
			for(i = 1; i < size; i++)
				for (j = 0; j < chunksize; j++)
					MPI_Recv(pgm[i * chunksize + j], W, MPI_INT, i, tag, MPI_COMM_WORLD, &Stat);
			
			FILE* fout = fopen(argv[2], "w");
			
			fprintf(fout, "P2\n");
			fprintf(fout, "%d %d\n", W, H);
			fprintf(fout, "%d\n", NUM_COLORS - 1);
			
			// write the final matrix to file
			for (i = H - 1; i >= 0; i--) {
				for (j = 0; j < W; j++) {
					fprintf(fout, "%d ", pgm[i][j]);
				}
				fprintf(fout, "\n");
				free(pgm[i]);
			}
			free(pgm);
			
			fclose(fout);
		}
	}
	
	MPI_Finalize();
	return 0;
}

double complex_modulus(struct Complex z) {
	return sqrt(z.re * z.re + z.im * z.im);
}

struct Complex add_complex(struct Complex a, struct Complex b) {
	struct Complex c;
	c.re = a.re + b.re;
	c.im = a.im + b.im;
	return c;
}

struct Complex multiply_complex(struct Complex a, struct Complex b) {
	struct Complex c;
	c.re = a.re * b.re - a.im * b.im;
	c.im = a.re * b.im + a.im * b.re;
	return c;
}
