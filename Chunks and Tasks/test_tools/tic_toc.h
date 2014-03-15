#ifndef TIC_TOC_HEADER
#define TIC_TOC_HEADER

#include <time.h>

#define BILLION  1000000000L;
#define TIME struct timespec

struct timespec tic()
{
  TIME time;
  clock_gettime( CLOCK_MONOTONIC_RAW, &time);
  
  return time;
}


double toc(struct timespec start)
{
  TIME stop;
  clock_gettime( CLOCK_MONOTONIC_RAW, &stop); 

   return (double)( stop.tv_sec - start.tv_sec )
          + (double)( stop.tv_nsec - start.tv_nsec )
            / BILLION;

}


#endif

