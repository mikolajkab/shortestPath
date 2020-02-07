 #include <omp.h>
 #include <stdio.h>

 #define N 100
 #define CHUNKSIZE 5

 int main(int argc, char *argv[]) {

 int i, chunk, tid;
 float a[N], b[N], c[N];

 /* Some initializations */
 for (i=0; i < N; i++)
   a[i] = b[i] = i * 1.0;
 chunk = CHUNKSIZE;

//  omp_set_num_threads(5);
 #pragma omp parallel shared(a,b,c,chunk) private(i, tid)
   {
    tid = omp_get_thread_num();

    // #pragma omp for schedule(static, chunk) nowait
    for (i=0; i < N; i++)
    {
      c[i] = a[i] + b[i];
      printf("thread id = %d, i = %d\n", tid, i);
    }
   }   /* end of parallel region */

  for (i=0; i<N; i++)
  {
    printf("c[%d]: %f\n", i, c[i]);
  }
 }