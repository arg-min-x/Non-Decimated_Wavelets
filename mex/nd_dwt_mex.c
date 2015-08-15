#include "mex.h"
#include "matrix.h"
#include <stdlib.h>
#include <fftw3.h>
#include "nddwt.h"

/*  the gateway routine.  */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{

    /* Input Checks*/
    if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
        mexErrMsgIdAndTxt( "MATLAB:FFT2mx:invalidNumInputs",
            "Arrays must be double");
    }
    if (!mxIsComplex(prhs[0]) || !mxIsComplex(prhs[1]) ) {
        mexErrMsgIdAndTxt( "MATLAB:FFT2mx:invalidNumInputs",
            "Arrays must be complex");
    }
    if (mxGetNumberOfElements(prhs[1]) != mxGetNumberOfElements(prhs[0])*8 ) {
        mexErrMsgIdAndTxt( "MATLAB:FFT2mx:invalidNumInputs",
            "FIlter size and image size not consistant");
    }
    
    /* */
    int num_dims,ind, numel,num_dims_kernel;
    const int *dims_mat,*dims_kernel;
    int *dims_c;
    double *imageR, *imageI, *kernelR, *kernelI,*outR,*outI;
    
    /* Get pointers to input array */
    imageR =  (double *) mxGetPr(prhs[0]);
    imageI =  (double *) mxGetPi(prhs[0]);
    kernelR = (double *) mxGetPr(prhs[1]);
    kernelI = (double *) mxGetPi(prhs[1]);
    
    /* Get input dimesions */
    num_dims = mxGetNumberOfDimensions(prhs[0]);
    dims_mat = mxGetDimensions(prhs[0]);
    numel = mxGetNumberOfElements(prhs[0]);
    dims_kernel = mxGetDimensions(prhs[1]);
    num_dims_kernel = mxGetNumberOfDimensions(prhs[1]);
    
    /* C uses row major array storage, reverse dimension order to fix*/
    dims_c = (int *) malloc(sizeof(int)*num_dims);
    for (ind =0;ind<num_dims;ind++){
//         dims_c[ind] = dims_mat[num_dims-1-ind];
        dims_c[ind] = dims_mat[ind];
    }
    
    /* Create an output array */
    plhs[0] = mxCreateNumericArray(num_dims_kernel, dims_kernel, mxDOUBLE_CLASS, mxCOMPLEX);
    outR = (double *) mxGetPr(plhs[0]);
    outI = (double *) mxGetPi(plhs[0]);
    
    /* Take the wavelet transform */
    nd_dwt_dec_1level(outR, outI, imageR, imageI, kernelR, kernelI,num_dims,
                       dims_c);
    
    /* free dims_c*/
    free(dims_c);
}

/*        mexPrintf("c=%d\t mat =%d\n",dims_c[ind],dims_mat[num_dims-1-ind]); 
     mexPrintf("%d\n",numel);
 */

