	
 #include <omp.h>
 #include <stdio.h>

 #define N 10

 int main(int argc, char *argv[]) {

 int i, tid;
 float a[N], b[N], c[N], d[N];

 /* Some initializations */
 for (i=0; i < N; i++) {
   a[i] = i * 1.5;
   b[i] = i + 22.35;
   }

 omp_set_num_threads(5);
 #pragma omp parallel shared(a,b,c,d) private(i, tid)
   {
   tid = omp_get_thread_num();

   #pragma omp sections nowait
     {

     #pragma omp section
     for (i=0; i < N; i++)
     {
         c[i] = a[i] + b[i];
         printf("thread id = %d i = %d\n", tid, i);
     }

     #pragma omp section
     for (i=0; i < N; i++)
     {
        d[i] = a[i] * b[i];
        printf("thread id = %d i = %d\n", tid, i);
     }

     }  /* end of sections */

   }  /* end of parallel region */

 }