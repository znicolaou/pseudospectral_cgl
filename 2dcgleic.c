//Zachary G. Nicolaou, Northwestern University 4/4/2016
//Pseudospectral method to integrate the complex ginzburg landau equation in two dimensions from given ic
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#include <fftw3.h>
#include <omp.h>

struct parameters {double c1i; double c1f; double c3i; double c3f; double t2; double L; int N; int fcount;};

fftw_complex *fftw_y, *fftw_F, *fftw_y2, *fftw_F2;
fftw_plan iplan, fplan, f2plan, i2plan;

int func (double t, const double y[], double f[], void *params) {
    (void)(t);
    double c1, c3;
    double c1i = ((struct parameters *)params)->c1i;
    double c3i = ((struct parameters *)params)->c3i;
    double c1f = ((struct parameters *)params)->c1f;
    double c3f = ((struct parameters *)params)->c3f;
    double t2 = ((struct parameters *)params)->t2;
    if(t < t2) {
        c1=c1i + (c1f-c1i)*t/t2;
        c3=c3i + (c3f-c3i)*t/t2;
    }
    else {
        c1 = c1f;
        c3 = c3f;
    }
    double L = ((struct parameters *)params)->L;
    int N = ((struct parameters *)params)->N;
    double omegasquared, amp;
    ((struct parameters *)params)->fcount++;
    int i, j;
    
#pragma omp parallel for private(i, j) shared(fftw_y)
    for(i=0; i<N; i++) {
        for(j=0; j<N; j++){
            fftw_y[N*i+j][0]= y[2*(N*i+j)];
            fftw_y[N*i+j][1]= y[2*(N*i+j)+1];
        }
    }
    fftw_execute(fplan);
    
#pragma omp parallel for private(i, j, omegasquared) shared(fftw_F2)
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            omegasquared = ((N/2-abs(j-N/2))*(N/2-abs(j-N/2))+(N/2-abs(i-N/2))*(N/2-abs(i-N/2)))/(L*L);
            fftw_F2[N*i+j][0] = -omegasquared*(fftw_F[N*i+j][0])/(N*N);
            fftw_F2[N*i+j][1] = -omegasquared*(fftw_F[N*i+j][1])/(N*N);
        }
    }
    fftw_execute(i2plan);
    
#pragma omp parallel for private(i, j, amp) shared(f)
    for(i=0; i<N; i++) {
        for(j=0; j<N; j++){
            amp=(y[2*(N*i+j)]*y[2*(N*i+j)]+y[2*(N*i+j)+1]*y[2*(N*i+j)+1]);
            f[2*(N*i+j)] = y[2*(N*i+j)]+fftw_y2[N*i+j][0]-c1*fftw_y2[N*i+j][1]-amp*y[2*(N*i+j)]-amp*c3*y[2*(N*i+j)+1];
            f[2*(N*i+j)+1] = y[2*(N*i+j)+1]+fftw_y2[N*i+j][1]+c1*fftw_y2[N*i+j][0]-amp*y[2*(N*i+j)+1]+amp*c3*y[2*(N*i+j)];
        }
    }
    return GSL_SUCCESS;
}

int main (int argc, char* argv[]) {
    struct timeval start,end,start2,end2;
    
    if(argc != 16) {
        printf("usage: ./2dcgleic N L c1i c1f c3i c3f t1 t2 t3 dt epsabs epsrel Nthreads method out \n\n");
        printf("N is number of oscillators \n");
        printf("2Pi L is linear system size \n");
        printf("c1i is initial CGLE Laplacian coefficient \n");
        printf("c1f is final CGLE Laplacian coefficient \n");
        printf("c3i is initial CGLE cubic coefficient \n");
        printf("c3f is final CGLE cubic coefficient \n");
        printf("t1 is total integration time \n");
        printf("t2 is time at which c1 and c3 reach their final values.  After this time, start averaging and outputting animation data. \n");
        printf("t3 is time stop outputting animation data \n");
        printf("dt is the time between outputs \n");
        printf("epsabs is absolute error tolerance \n");
        printf("epsrel is relative error tolerance \n");
        printf("Nthreads number of threads to parallelize with\n");
        printf("method is time stepping method - possible values are rk4 for explicit Runge-Kutta 4, rkf45 for Runge-Kutta-Fehlberg 4/5, and adams for Adams multistep\n");
        printf("out is base file name for output and input.  The initial condition is retrieved from outic.dat if it exists; otherwise random initial conditions are used. The output files are: out.out contains time step data, outlast.dat contains the last state, outanimation.dat contains the states between timesteps between t2 and t3, outfrequency.dat contains the average frequency between t2 and t3 in dt steps, outvarfrequency.dat contains the variance of the frequency between t2 and t3 in dt steps, outorder.dat contains the average order parameter between t2 and t3 in dt steps, outvarorder.dat contains the variance of the order parameter between t2 and t3 in dt steps, amp contains the average amplitude between t2 and t3 in dt steps, outvaramp.dat contains the variance of the amplitude between t2 and t3 in dt steps. \n\n");
        printf("-----------------------------------------------------------------------\n");
        printf("Example 1) ./2dcgleic 768 192 2.0 2.0 0.72 0.72 1e3 1e3 1e3 1 1e-3 1e-3 6 rkf45 random \n");
        printf("A spiral is likely to nucleate out of amplitude turbulence with these parameters. If no spiral nucleation occurs or multiple spirals nucleate, try again. It may take a few attempts to get an isolated spiral. Use the Mathematica notebook plot.nb to refine the grid and run the next example. \n");
        printf("-----------------------------------------------------------------------\n");
        printf("Example 2) ./2dcgleic 1536 192 2.0 2.0 0.72 0.85 1e3 1e3 1e3 1 1e-3 1e-3 6 rkf45 refine \n");
        printf("Quasistatically increase c_3 to a value where the spiral nucleation rate is low.  The Mathematica notebook plot.nb can be used to center the spiral and create a new initial condition spiralic.dat. \n ");
        printf("-----------------------------------------------------------------------\n");
        printf("Example 3) ./2dcgleic 1536 192 2.0 2.0 0.85 0.85 1e4 1e3 1e4 1 1e-10 1e-10 6 rkf45 spiralcore \n");
        printf("The grid spacing and error tolerances here are converged to the continuum limit, and after this run the spiral should have grown to its full size.\n");
        printf("-----------------------------------------------------------------------\n");
        exit(0);
    }
    int N=atoi(argv[1]);
    double L = atof(argv[2]);
    double c1i = atof(argv[3]);
    double c1f = atof(argv[4]);
    double c3i = atof(argv[5]);
    double c3f = atof(argv[6]);
    double t = 0.;
    double t1 = atof(argv[7]);
    double t2 = atof(argv[8]);
    double t3 = atof(argv[9]);
    double dt = atof(argv[10]);
    struct parameters params = {c1i,c1f,c3i,c3f,t2,L,N,0};
    double epsabs = atof(argv[11]);
    double epsrel = atof(argv[12]);
    int Nthreads = atoi(argv[13]);
    char* method = argv[14];
    char* filebase = argv[15];
    double *y, *dydt, *frequency, *varfrequency, *order, *varorder, *amp, *varamp;
    double phi, ti, omegasquared, delta, delta2;
    int i,j,count=0, acount=0;
    FILE *out, *outlast, *outfrequency, *outvarfrequency, *outorder, *outvarorder, *outamp, *outvaramp, *outanimation, *in;
    char file[256];
    strcpy(file,filebase);
    strcat(file, "frequency.dat");
    outfrequency = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file,"varfrequency.dat");
    outvarfrequency = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file,"order.dat");
    outorder = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file,"varorder.dat");
    outvarorder = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file,"amp.dat");
    outamp = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file,"varamp.dat");
    outvaramp = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file,"animation.dat");
    outanimation = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file,".out");
    out = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file,"last.dat");
    outlast=fopen(file,"w");
    fclose(outlast);
    
    
    y=calloc(2*N*N,sizeof(double));
    dydt=calloc(2*N*N,sizeof(double));
    frequency=calloc(N*N,sizeof(double));
    varfrequency=calloc(N*N,sizeof(double));
    order=calloc(N*N,sizeof(double));
    varorder=calloc(N*N,sizeof(double));
    amp=calloc(N*N,sizeof(double));
    varamp=calloc(N*N,sizeof(double));
    fftw_y = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N*N);
    fftw_F = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N*N);
    fftw_F2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N*N);
    fftw_y2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N*N);
    
    omp_set_num_threads(Nthreads);
    fftw_init_threads();
    fftw_plan_with_nthreads(Nthreads);
    fplan = fftw_plan_dft_2d(N, N, fftw_y, fftw_F, FFTW_FORWARD, FFTW_MEASURE);
    iplan = fftw_plan_dft_2d(N, N, fftw_F, fftw_y, FFTW_BACKWARD, FFTW_MEASURE);
    f2plan = fftw_plan_dft_2d(N, N, fftw_y2, fftw_F2, FFTW_FORWARD, FFTW_MEASURE);
    i2plan = fftw_plan_dft_2d(N, N, fftw_F2, fftw_y2, FFTW_BACKWARD, FFTW_MEASURE);

    printf("2dcgleic %i %f %f %f %f %f %f %f %f %f %e %e %i %s %s\n", N, L, c1i, c1f, c3i, c3f, t1, t2, t3, dt, epsabs, epsrel, Nthreads, method, filebase);
    fprintf(out,"2dcgleic %i %f %f %f %f %f %f %f %f %f %e %e %i %s %s\n", N, L, c1i, c1f, c3i, c3f, t1, t2, t3, dt, epsabs, epsrel, Nthreads, method, filebase);

    strcpy(file,filebase);
    strcat(file,"ic.dat");
    if((in = fopen(file,"r"))){
        fread(y,sizeof(double),2*N*N,in);
    }
    else {
        printf("No initial condition file - using random initial conditions\n");
        //random initial conditions
        srand(time(NULL));
        double scalefac=3.6; //damp high wavenumber modes so the perturbation is not too jagged
        double phi,r;
        for(j = 0; j<N; j++) {
            for(i = 0; i<N; i++) {
                phi = 2*3.14/RAND_MAX*rand();
                r =exp(-scalefac*sqrt(((N/2-abs(j-N/2))*(N/2-abs(j-N/2))+(N/2-abs(i-N/2))*(N/2-abs(i-N/2)))/(L*L)))/RAND_MAX*rand();
                fftw_F[N*i+j][0] = r*cos(phi)/N;
                fftw_F[N*i+j][1] = r*sin(phi)/N;
            }
        }
        fftw_execute(iplan);
        
        for(j=0; j<N*N; j++) {
            y[2*j] = fftw_y[j][0];
            y[2*j+1] = fftw_y[j][1];
        }
    }
    fclose(in);
    
    gsl_odeiv2_system sys = {func, NULL, 2*N*N, &params};
    gsl_odeiv2_driver *d;
    if (strcmp(method,"adams") == 0)
        d = gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_msadams, 1e-10, epsabs, epsrel);
    else if (strcmp(method,"rk4") == 0)
        d = gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rk4, 1e-10, epsabs, epsrel);
    else if (strcmp(method, "rkf45") == 0)
        d = gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rkf45, 1e-10, epsabs, epsrel);
    else if (strcmp(method, "rkck") == 0)
        d = gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rkck, 1e-10, epsabs, epsrel);
    else if (strcmp(method, "rk8pd") == 0)
        d = gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rk8pd, 1e-10, epsabs, epsrel);
    else
        return 0;

    gettimeofday(&start,NULL);
    while(t<t1) {
        strcpy(file,filebase);
        strcat(file,"last.dat");
        outlast=fopen(file,"w");
        fwrite(y,sizeof(double),2*N*N,outlast);
        fflush(outlast);
        fclose(outlast);
        
        
        if(t>=t2 && t<=t3){
            fwrite(y,sizeof(double),2*N*N,outanimation);
        }
        if(t>=t2) {
            count++;
            func(t, y, dydt, &params);

            for(i=0; i<N; i++) {
                for(j=0; j<N;j++) {
                    fftw_y[N*i+j][0] = y[2*(N*i+j)];
                    fftw_y[N*i+j][1] = y[2*(N*i+j)+1];
                }
            }
            fftw_execute(fplan);
            
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    omegasquared = ((N/2-abs(j-N/2))*(N/2-abs(j-N/2))+(N/2-abs(i-N/2))*(N/2-abs(i-N/2)))/(L*L);
                    fftw_F2[N*i+j][0] = -omegasquared*(fftw_F[N*i+j][0])/(N*N);
                    fftw_F2[N*i+j][1] = -omegasquared*(fftw_F[N*i+j][1])/(N*N);
                }
            }
            fftw_execute(i2plan);
            
            for(i=0; i<N; i++) {
                for(j=0; j<N; j++) {
                    frequency[N*i+j] += (dydt[2*(N*i+j)+1]*y[2*(N*i+j)]-y[2*(N*i+j)+1]*dydt[2*(N*i+j)])/(y[2*(N*i+j)]*y[2*(N*i+j)]+y[2*(N*i+j)+1]*y[2*(N*i+j)+1]);
                    varfrequency[N*i+j] += pow((dydt[2*(N*i+j)+1]*y[2*(N*i+j)]-y[2*(N*i+j)+1]*dydt[2*(N*i+j)])/(y[2*(N*i+j)]*y[2*(N*i+j)]+y[2*(N*i+j)+1]*y[2*(N*i+j)+1]),2.0);
                    
                    order[N*i+j] += sqrt((1.0+c1f*c1f)/(y[2*(N*i+j)]*y[2*(N*i+j)]+y[2*(N*i+j)+1]*y[2*(N*i+j)+1]))*sqrt(fftw_y2[N*i+j][0]*fftw_y2[N*i+j][0]+fftw_y2[N*i+j][1]*fftw_y2[N*i+j][1]) ;
                    varorder[N*i+j] += pow(sqrt((1.0+c1f*c1f)/(y[2*(N*i+j)]*y[2*(N*i+j)]+y[2*(N*i+j)+1]*y[2*(N*i+j)+1]))*sqrt(fftw_y2[N*i+j][0]*fftw_y2[N*i+j][0]+fftw_y2[N*i+j][1]*fftw_y2[N*i+j][1]), 2.0);
                    
                    amp[N*i+j] += sqrt(y[2*(N*i+j)]*y[2*(N*i+j)]+y[2*(N*i+j)+1]*y[2*(N*i+j)+1]);
                    varamp[N*i+j] += pow(sqrt(y[2*(N*i+j)]*y[2*(N*i+j)]+y[2*(N*i+j)+1]*y[2*(N*i+j)+1]), 2.0);
                }
            }
        }
        
        ti = t+dt;
        int status = gsl_odeiv2_driver_apply(d, &t, ti, y);
        if(status != GSL_SUCCESS) {
            printf("error %s\n", gsl_strerror(status));
            return 1;
        }
        
        
        gettimeofday(&end,NULL);
        printf("%f\t%f\t%f\t%i\t\n",t/t1,end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec), (end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec))/(t/t1)*(1-t/t1), params.fcount);
        fprintf(out,"%f\t%f\t%f\t%i\t\n",t/t1,end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec), (end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec))/(t/t1)*(1-t/t1), params.fcount);
        fflush(stdout);
        fflush(out);
    }
    for(j=0; j<N*N; j++) {
        frequency[j] /= count;
        order[j] /= count;
        amp[j] /= count;
        varfrequency[j] = varfrequency[j]/count - pow(frequency[j], 2.0);
        varorder[j] = varorder[j]/count - pow(order[j], 2.0);
        varamp[j] = varamp[j]/count - pow(amp[j], 2.0);
    }
    strcpy(file,filebase);
    strcat(file,"last.dat");
    outlast=fopen(file,"w");
    fwrite(y,sizeof(double),2*N*N,outlast);
    fflush(outlast);
    fclose(outlast);
    if(t>=t2 && t<=t3){
        fwrite(y,sizeof(double),2*N*N,outanimation);
    }
    
    fwrite(frequency,sizeof(double),N*N,outfrequency);
    fwrite(varfrequency,sizeof(double),N*N,outvarfrequency);
    fwrite(order,sizeof(double),N*N,outorder);
    fwrite(varorder,sizeof(double), N*N, outvarorder);
    fwrite(amp,sizeof(double),N*N,outamp);
    fwrite(varamp,sizeof(double),N*N,outvaramp);

    
    
    printf("\n");
    
    fftw_cleanup_threads();
    gsl_odeiv2_driver_free (d);
    fftw_destroy_plan(iplan);
    fftw_destroy_plan(fplan);
    fftw_destroy_plan(f2plan);
    fftw_destroy_plan(i2plan);
    fftw_free(fftw_y);
    fftw_free(fftw_y2);
    fftw_free(fftw_F);
    fftw_free(fftw_F2);
    fclose(outfrequency);
    fclose(outvarfrequency);
    fclose(outorder);
    fclose(outvarorder);
    fclose(outamp);
    fclose(outvaramp);
    fclose(out);
    fclose(outanimation);
    free(y);
    free(dydt);
    free(frequency);
    free(varfrequency);
    free(order);
    free(varorder);
    free(amp);
    free(varamp);
    return 0;
}
