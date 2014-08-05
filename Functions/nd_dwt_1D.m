%Calculates the 1 Dimensional nondecimated Wavelet transform using fast 
%    convlolution and periodic boundry conditions.  The Filters are 
%    pre-computed and stored in f_dec and f_rec to reduce computation time 
%    when performing Multiple forward/backward transforms
%
%Methods:
%   ud_dwt3D:   Constructor
%               Inputs: wname - Wavelet Filters to Use i.e. db1,db2,etc.
%                        Either a string 'db1' or a cell {'db4','db1'} where
%                        the first element in the string is filter for the 
%                        spatial domain and the second is the filter for the 
%                        time domain
%                       sizes - size of the 3D object [n1,n2,n3]
%
%   dec:        Multilevel Decomposition
%               Inputs: x - Image domain signal for decomposition
%                       levels - Number of decomposition Levels
%               Outputs: y - Multilevel non-decimated DWT coefficients in a
%                        2D array where the data is arranged [n1,n2,n3,bands]
%                        The bands are orginized as follows.  Let "HVT"
%                        represent the Horizontal, Vertical, and Temporal
%                        Bands.  The coefficients are ordered as follows,
%                        "LLL", "HLL", "LHL", "HHL", "LLH", "HLH", "LHH",
%                        "HHH" where "H" denotes the high frequency filter
%                        and L represents the low frequency filter.
%                        Successive levels of decomposition are stacked
%                        such that the highest "LLL" is in [n1,n2,n3,1]
%
%   rec:        Multilevel Reconstruction
%               Inputs: x - Wavelet coefficients in a 4D array size 
%                       [n1,n2,n3,bands].
%               Outputs: y - Reconstructed 3D array.%   
%
%**************************************************************************
% The Ohio State University
% Written by:   Adam Rich 
% Email:        rich.178@osu.edu
% Last update:  8/4/2014
%**************************************************************************
classdef nd_dwt_1D
    %ND_DWT_1D Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
    end
    
end

