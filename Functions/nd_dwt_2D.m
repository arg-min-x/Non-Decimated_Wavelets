%Calculates the nondecimated Wavelet transform using fast convlolution and
%   periodic boundry conditions.  The Filters are pre-computed and stored
%   in f_dec and f_rec to reduce computation time when performing Multiple
%   forward/backward transforms
%
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
%                        4D array where the data is arranged [n1,n2,n3,bands]
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
%                       [bands,n1,n2,n3].
%               Outputs: y - Reconstructed 3D array.%   
%
%**************************************************************************
% The Ohio State University
% Written by:   Adam Rich 
% Email:        rich.178@osu.edu
% Last update:  8/4/2014
%**************************************************************************
classdef nd_dwt_2D
    %ND_DWT_2D Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        f_dec;          % Decomposition Filters
        f_rec;          % Reconstruction Filters
        sizes;          % Size of the 3D Image
        f_size;         % Length of the filters
        wname;          % Wavelet used
    end
    
    methods
        % Constructor Object
        function obj = nd_dwt_2D(wname,sizes)
            % Set Image size
            obj.sizes = sizes;
            
            if ischar(wname)
                obj.wname = {wname,wname};
            elseif iscell(wname)
                obj.wname = wname;
            end
            
            % Get the Filter Coefficients
            [obj.f_dec,obj.f_rec,obj.f_size] = obj.get_filters(obj.wname);
            
        end
        
        % Returns the Filters and
        function [f_dec,f_rec,f_size] = get_filters(obj,wname)
            
            % Get the Decomposition Filters Filters for Spatial Domain
            [LO_D,HI_D]   = wfilters(wname{1});
            [LO_D2,HI_D2] = wfilters(wname{2});
            
            % Find the filter size
            f_size.s = length(LO_D);
            
            % Get the 2D Filters by taking outer products
            f_dec.LL = LO_D.'*LO_D2;
            f_dec.HL = HI_D.'*LO_D2;
            f_dec.LH = LO_D.'*HI_D2;
            f_dec.HH = HI_D.'*HI_D2;
        

            % Get th Reconstruction FiltersFilters
            [~,~,LO_R,HI_R] = wfilters(wname{1});
            [~,~,LO_R2,HI_R2] = wfilters(wname{2});
            
            % Get the 3D Filters by taking outer products
            f_rec.LL = LO_R.'*LO_R2;
            f_rec.LH = LO_R.'*HI_R2;
            f_rec.HL = HI_R.'*LO_R2;
            f_rec.HH = HI_R.'*HI_R2;
        end
    end
    
end

