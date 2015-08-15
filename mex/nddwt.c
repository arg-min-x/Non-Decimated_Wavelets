//
//  nddwt.c
//  wavelets_3D
//
//  Created by Adam V. Rich on 8/14/15.
//  Copyright (c) 2015 Adam. All rights reserved.
//

#include "nddwt.h"
#include <stdio.h>
#include <fftw3.h>

fftw_plan init_fftw_plan(double *inR, double *inI, double *outR, double *outI, int *dims, int numDims){
    /*
     inR:        pointer to Real portion of input array
     inI:        pointer to Imaginary portion of input array
     outR:       pointer to Real portion of output array
     outI:       pointer to Imaginary portion of output array
     dims:       pointer to array of data dimensions
     numDims:    integer number of dimensions in the array
     */
    
    /* Initialize fftw plan */
    fftw_plan fftwplan;
    fftw_iodim dims_guru[numDims];
    
    /* Set dims for guru interface */
    for (int ind = 0; ind<numDims; ind++) {
        dims_guru[ind].n = dims[ind];
        if (ind ==0) {
            dims_guru[ind].is = 1;
            dims_guru[ind].os = 1;
        }
        else{
            dims_guru[ind].is = dims_guru[ind-1].is*dims_guru[ind-1].n;
            dims_guru[ind].os = dims_guru[ind-1].is*dims_guru[ind-1].n;
//            printf("%d\n",dims_guru[ind].is);
        }
    }
    
    //    FILE *fp;
    //    fp=fopen("/Users/adam/Documents/c_code/wavelets/wavelets/wisdom", "w");
    //     fftw_import_wisdom_from_filename("/Users/adam/Documents/c_code/wavelets/wavelets/wisdom");
    //    fftw_import_wisdom_from_file(fp);
    
    /* Initialize Plan */
    fftwplan = fftw_plan_guru_split_dft(numDims, dims_guru,0,NULL,inR,inI,outR,outI,FFTW_ESTIMATE);
    //    fftw_export_wisdom_to_file(fp);
    //    fclose(fp);
    
    /* Return an fftwplan */
    return fftwplan;
}

void pointByPoint(double *aR,double *aI,double *bR,double *bI,double *cR,double *cI, int size){
    double ac,bd,a_b,c_d;
    #pragma omp parallel for private(ac,bd,a_b,c_d)
    for (int ind = 0; ind < size; ind++) {
        ac = aR[ind]*bR[ind];
        bd = aI[ind]*bI[ind];
        a_b = aR[ind] + aI[ind];
        c_d = bR[ind] + bI[ind];
        cR[ind] = ac-bd;
        cI[ind] = a_b*c_d-ac-bd;
    }
    //    R[(a+ib)(c+id)]	=	ac-bd
    //    (6)
    //    I[(a+ib)(c+id)]	=	(a+b)(c+d)-ac-bd.
}

void nd_dwt_dec_1level(double *outR, double *outI, double *imageR, double *imageI,
                       double *kernelR, double *kernelI, int num_dims,int *dims){
    /* fftw plan init */
   int int_threads = fftw_init_threads();
   printf("%d\n",int_threads);
   fftw_plan_with_nthreads(8);
    fftw_plan ifftw_plan_in_place,fftw_plan_in_place;
    double numel = 1;
    int start_ind;
    fftw_plan_in_place = init_fftw_plan(imageR, imageI, imageR, imageI, dims, num_dims);
    ifftw_plan_in_place = init_fftw_plan(outI, outR, outI, outR, dims, num_dims);
    
    /* find the number of elements in the image */
    for (int ind =0; ind<num_dims; ind++) {
        numel *=dims[ind];
    }

    /* Take the fft of the input image */
    fftw_execute_split_dft(fftw_plan_in_place, imageR, imageI, imageR, imageI);
 
    /* Loop over each subband. multiply be the kernele and take ifft */
    #pragma omp parallel for private(start_ind)
    for (int ind = 0; ind<8; ind++) {
        /* point by point multiply and ifft for each subband */
        if (ind==0) {
            start_ind =0;
        }
        else{
            start_ind = ind*numel;
        }
        pointByPoint(imageR, imageI, &kernelR[start_ind], &kernelI[start_ind], &outR[start_ind], &outI[start_ind], numel);
        fftw_execute_split_dft(ifftw_plan_in_place, &outI[start_ind], &outR[start_ind], &outI[start_ind], &outR[start_ind]);
    }
    
//     /* Normalize the DFT */
//     for (int ind = 0; ind<numel*8; ind++) {
//         outR[ind] = outR[ind]/numel;
//         outI[ind] = outI[ind]/numel;
//     }
    
    /* Destory fftw_plans*/
    fftw_destroy_plan(fftw_plan_in_place);
    fftw_destroy_plan(ifftw_plan_in_place);
}

void nd_dwt_rec_1level(double *outR, double *outI, double *imageR, double *imageI,
                       double *kernelR, double *kernelI, int num_dims,int *dims){
    /* fftw plan init */
   int int_threads = fftw_init_threads();
   printf("%d\n",int_threads);
   fftw_plan_with_nthreads(8);
    fftw_plan ifftw_plan_in_place,fftw_plan_in_place;
    double numel = 1;
    int start_ind;
    fftw_plan_in_place = init_fftw_plan(imageR, imageI, imageR, imageI, dims, num_dims);
    ifftw_plan_in_place = init_fftw_plan(outI, outR, outI, outR, dims, num_dims);
    
    /* find the number of elements in the image */
    for (int ind =0; ind<num_dims; ind++) {
        numel *=dims[ind];
    }

    /* Take the fft of the input image */
    fftw_execute_split_dft(fftw_plan_in_place, imageR, imageI, imageR, imageI);
 
    /* Loop over each subband. multiply be the kernele and take ifft */
    #pragma omp parallel for private(start_ind)
    for (int ind = 0; ind<8; ind++) {
        /* point by point multiply and ifft for each subband */
        if (ind==0) {
            start_ind =0;
        }
        else{
            start_ind = ind*numel;
        }
        pointByPoint(imageR, imageI, &kernelR[start_ind], &kernelI[start_ind], &outR[start_ind], &outI[start_ind], numel);
        fftw_execute_split_dft(ifftw_plan_in_place, &outI[start_ind], &outR[start_ind], &outI[start_ind], &outR[start_ind]);
    }
    
//     /* Normalize the DFT */
//     for (int ind = 0; ind<numel*8; ind++) {
//         outR[ind] = outR[ind]/numel;
//         outI[ind] = outI[ind]/numel;
//     }
    
    /* Destory fftw_plans*/
    fftw_destroy_plan(fftw_plan_in_place);
    fftw_destroy_plan(ifftw_plan_in_place);
}



