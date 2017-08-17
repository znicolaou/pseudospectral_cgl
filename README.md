# pseudospectral_cgl
Pseudospectral code to integrate the complex Ginzburg-Landau equation in two dimensions
This is a C program that utilizes the GNU Scientific Library (GSL), the FFTW fast Fourier transform, and OpenMP libraries to 
implement a parallel, pseudospectral integration of the two dimensional complex Ginzburg Landau equation.  It has been used
with GSL 2.4 (https://www.gnu.org/software/gsl/) and FFTW 3.3.5 (http://www.fftw.org/) with the C compiler 
gcc 4.9.2 (https://gcc.gnu.org/).  It was compiled with

  gcc -fopenmp -O3 -o 2dcgleic 2dcgleic.c -lgsl -lgslcblas -lfftw3 -lfftw3_omp -lm
  
Once compiled, running the program will produce a usage message.
