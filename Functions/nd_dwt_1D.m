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
%                       sizes - length of the 1D signal
%
%   dec:        Multilevel Decomposition
%               Inputs: x - Image domain signal for decomposition
%                       levels - Number of decomposition Levels
%               Outputs: y - Multilevel non-decimated DWT coefficients in a
%                        2D array where the data is arranged [n1,bands]
%                        The bands are orginized as follows.The coefficients 
%                        are ordered as follows,"L", "H" where"H" denotes 
%                        the high frequency filter and L represents the low 
%                        frequency filter. Successive levels of decomposition 
%                        are stacked such that the highest "L" is in [n1,1]
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
        f_dec;          % Decomposition Filters
        sizes;          % Size of the Signal
        f_size;         % Length of the filters
        wname;          % Wavelet used
    end
    
    methods
        % Constructor Object
        function obj = nd_dwt_1D(wname,sizes)
            % Check Inputs
            if ischar(wname)
                obj.wname = {wname,wname};
            elseif iscell(wname)
                error('Wavelet Name Must be a string');
            end
            
            if length(sizes) ~= 1
                error('1D array length must be a scalar');
            end
            
            % Set Image size
            obj.sizes = sizes;                  

            % Get the Filter Coefficients
            [obj.f_dec,obj.f_size] = obj.get_filters(obj.wname);
            
        end
        
        % Multilevel Undecimated Wavelet Decomposition
        function y = dec(obj,x,level)
            % Fourier Transform of Signal
            if size(x,1) ==1
                x = x.';
            end
            x = fft(x);
            
            % Preallocate
            y = zeros([obj.sizes, 2+(level-1)]);

            % Calculate Mutlilevel Wavelet decomposition
            for ind = 1:level
                % First Level
                if ind ==1
                    y = level_1_dec(obj,x);
                % Succssive Levels
                else
                    y = cat(2,level_1_dec(obj,fftn(squeeze(y(:,1)))), y(:,2:end));
                end
            end
                      
        end
        
        % Multilevel Undecimated Wavelet Reconstruction
        function y = rec(obj,x)
            
            % Find the decomposition level
            level = ceil(size(x,2)-1)
            
            % Fourier Transform of Signal
            x = fft(x,[],1);
            
            % Reconstruct from Multiple Levels
            for ind = 1:level
                % First Level
                if ind ==1
                    y = level_1_rec(obj,x);
                % Succssive Levels
                else
                    y = fft(y);
                    y = level_1_rec(obj,cat(2,y,x(:,3+(ind-2))));
                end
            end 
            
        end
    end
    
    %% Private Methods
    methods (Access = protected,Hidden = true)
        
        % Returns the Filters and 
        function [f_dec,f_size] = get_filters(obj,wname)
        % Decomposition Filters
        
            % Get Filters for Spatial Domain
            [LO_D,HI_D] = wave_filters(wname{1});
            
            % Find the filter size
            f_size.s1 = length(LO_D);
            
            % Dimension Check
            if f_size.s1 > obj.sizes(1)
                error(['First Dimension of Data is shorter than the wavelet'...
                    ,' filter being used']);
            end
            
            % Add a circularshift of half the filter length to the 
            % reconstruction filters by adding phase to them
            shift = exp(1j*2*pi*f_size.s1/2*linspace(0,1-1/obj.sizes,obj.sizes));
            
            % Take the Fourier Transform of the Kernels for Fast
            % Convolution
            f_dec.L = 1/sqrt(2)*(shift.*fft(LO_D,obj.sizes)).';
            f_dec.H = 1/sqrt(2)*(shift.*fft(HI_D,obj.sizes)).';
        end
        
        % Single Level Redundant Wavelet Decomposition
        function y = level_1_dec(obj,x_f)
            % Preallocate
            y = zeros([obj.sizes,2]);
            
            % Calculate Wavelet Coefficents Using Fast Convolution
            y(:,1) = ifft(x_f.*obj.f_dec.L);
            y(:,2) = ifft(x_f.*obj.f_dec.H);
        end
        
        % Single Level Redundant Wavelet Reconstruction
        function y = level_1_rec(obj,x_f)
            
            % Reconstruct the 3D array using Fast Convolution
            y = ifft(squeeze(x_f(:,1)).*conj(obj.f_dec.L));
            y = y + ifft(squeeze(x_f(:,2)).*conj(obj.f_dec.H));
        end
    end
    
end

