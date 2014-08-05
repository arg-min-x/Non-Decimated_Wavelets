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
classdef nd_dwt_4D
    %ND_DWT_4D Summary of this class goes here
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
        function obj = nd_dwt_4D(wname,sizes)
            % Set Image size
            obj.sizes = sizes;
            
            if ischar(wname)
                obj.wname = {wname,wname,wname,wname};
            elseif iscell(wname)
                obj.wname = wname;
            end
            
            % Get the Filter Coefficients
            [obj.f_dec,obj.f_rec,obj.f_size] = obj.get_filters(obj.wname);
        end
        
        % Returns the Filters and 
        function [f_dec,f_rec,f_size] = get_filters(obj,wname)
            % Get Filters for the first domain
            [LO_D,HI_D]   = wfilters(wname{1});
            [LO_D2,HI_D2] = wfilters(wname{2});
            [LO_D3,HI_D3] = wfilters(wname{3});
            [LO_D4,HI_D4] = wfilters(wname{4});
            
            % Find the filter size
            f_size.s1 = length(LO_D);
            f_size.s2 = length(LO_D2);
            f_size.s3 = length(LO_D3);
            f_size.s4 = length(LO_D4);
            
            % Get the 2D Filters by taking outer products
            dec_LL = LO_D.'*LO_D2;
            dec_HL = HI_D.'*LO_D2;
            dec_LH = LO_D.'*HI_D2;
            dec_HH = HI_D.'*HI_D2;
            
            % Take the Outerproducts for the third dimension
            for ind = 1:size(dec_LL,2)
                dec_LLL(:,ind,:) = dec_LL(:,ind)*LO_D3;
                dec_HLL(:,ind,:) = dec_HL(:,ind)*LO_D3;
                dec_LHL(:,ind,:) = dec_LH(:,ind)*LO_D3;
                dec_HHL(:,ind,:) = dec_HH(:,ind)*LO_D3;
                dec_LLH(:,ind,:) = dec_LL(:,ind)*HI_D3;
                dec_HLH(:,ind,:) = dec_HL(:,ind)*HI_D3;
                dec_LHH(:,ind,:) = dec_LH(:,ind)*HI_D3;
                dec_HHH(:,ind,:) = dec_HH(:,ind)*HI_D3;
            end
            
            % Take the Outerproducts for the fourth dimension
            for ii = 1:size(dec_LL,2)
                for kk = 1:size(dec_LL,3)
                    f_dec.LLLL(:,ii,kk,:) = dec_LLL(:,ind)*LO_D4;
                    f_dec.HLLL(:,ii,kk,:) = dec_HLL(:,ind)*LO_D4;
                    f_dec.LHLL(:,ii,kk,:) = dec_LHL(:,ind)*LO_D4;
                    f_dec.HHLL(:,ii,kk,:) = dec_HHL(:,ind)*LO_D4;
                    f_dec.LLHL(:,ii,kk,:) = dec_LLH(:,ind)*LO_D4;
                    f_dec.HLHL(:,ii,kk,:) = dec_HLH(:,ind)*LO_D4;
                    f_dec.LHHL(:,ii,kk,:) = dec_LHH(:,ind)*LO_D4;
                    f_dec.HHHL(:,ii,kk,:) = dec_HHH(:,ind)*LO_D4;
                    f_dec.LLLH(:,ii,kk,:) = dec_LLL(:,ind)*HI_D4;
                    f_dec.HLLH(:,ii,kk,:) = dec_HLL(:,ind)*HI_D4;
                    f_dec.LHLH(:,ii,kk,:) = dec_LHL(:,ind)*HI_D4;
                    f_dec.HHLH(:,ii,kk,:) = dec_HHL(:,ind)*HI_D4;
                    f_dec.LLHH(:,ii,kk,:) = dec_LLH(:,ind)*HI_D4;
                    f_dec.HLHH(:,ii,kk,:) = dec_HLH(:,ind)*HI_D4;
                    f_dec.LHHH(:,ii,kk,:) = dec_LHH(:,ind)*HI_D4;
                    f_dec.HHHH(:,ii,kk,:) = dec_HHH(:,ind)*HI_D4;
                end
            end
            
            f_rec = [];
        end
    end
    
end

