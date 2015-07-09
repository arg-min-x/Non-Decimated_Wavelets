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
%
%                       sizes - length of the 1D signal
%                           
%                       preserve_l2_norm - An optional third input.  If set
%                        TRUE, the l2 norm in the wavelet domain will be
%                        equal to the l2 norm in the signal domain
%
%   dec:        Multilevel Decomposition
%               Inputs: x - Image domain signal for decomposition
%
%                       levels - Number of decomposition Levels
%
%               Outputs: y - Multilevel non-decimated DWT coefficients in a
%                        2D array where the data is arranged [n1,bands]
%                        The bands are orginized as follows.The coefficients 
%                        are ordered as follows,"L", "H" where"H" denotes 
%                        the high frequency filter and L represents the low 
%                        frequency filter. Successive levels of decomposition 
%                        are stacked such that the highest "L" is in [n1,1]
%
%   rec:        Multilevel Reconstruction
%               Inputs: x - Wavelet coefficients in a 2D array size 
%                       [n1,bands].
%
%               Outputs: y - Reconstructed 1D array.%   
%
%**************************************************************************
% The Ohio State University
% Written by:   Adam Rich 
% Last update:  2/5/2015
%**************************************************************************

classdef nd_dwt_1D
    %ND_DWT_1D Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        f_dec;          % Decomposition Filters
        sizes;          % Size of the Signal
        f_size;         % Length of the filters
        wname;          % Wavelet used
        pres_l2_norm;   % Binary indicator to preserver l2 norm of coefficients
    end
    
    methods
        % Constructor Object
        function obj = nd_dwt_1D(wname,sizes,varargin)
            % Check Inputs
            if ischar(wname)
                obj.wname = {wname,wname};
            elseif iscell(wname)
                error('Wavelet Name Must be a string');
            end
            
            if length(sizes) ~= 1
                error('1D array length must be a scalar');
            end
            
            if isempty(varargin)
                obj.pres_l2_norm = 0;
            else
                obj.pres_l2_norm = varargin{1};
            end
            
            % Set Image size
            obj.sizes = sizes;                  

            % Get the Filter Coefficients
            [obj.f_dec,obj.f_size] = obj.get_filters(obj.wname);
            
        end
        
        % Multilevel Undecimated Wavelet Decomposition
        function y = dec(obj,x,level)
            
            % Check if input is real
            if isreal(x)
                x_real = 1;
            else
                x_real = 0;
            end
            
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
            
            % Take the real part if the input was real
            if x_real
                y = real(y);
            end     
        end
        
        % Multilevel Undecimated Wavelet Reconstruction
        function y = rec(obj,x)
            
            % Check if input is real
            if isreal(x)
                x_real = 1;
            else
                x_real = 0;
            end
            
            % Find the decomposition level
            level = ceil(size(x,2)-1);
            
            % Fourier Transform of Signal
            x = fft(x,[],1);
            
            % Reconstruct from Multiple Levels
            for ind = 1:level
                % First Level
                if ind ==1
                    y = level_1_rec(obj,x);
                    if ~obj.pres_l2_norm
                        y = y/2;
                    end
                % Succssive Levels
                else
                    y = fft(y);
                    y = level_1_rec(obj,cat(2,y,x(:,3+(ind-2))));
                    if ~obj.pres_l2_norm
                        y = y/2;
                    end
                end
            end
            
            % Take the real part if the input was real
            if x_real
                y = real(y);
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
            if obj.pres_l2_norm 
                scale = 1/sqrt(2);
            else
                scale = 1;
            end
            
            f_dec.L = scale*(shift.*fft(LO_D,obj.sizes)).';
            f_dec.H = scale*(shift.*fft(HI_D,obj.sizes)).';
        end
        
        % Single Level Redundant Wavelet Decomposition
        function y = level_1_dec(obj,x_f)
            
            % Preallocate
            y = zeros([obj.sizes,2]);
            
            % Calculate Wavelet Coefficents Using Fast Convolution
            y(:,1) = ifft(x_f.*obj.f_dec.L);
            y(:,2) = ifft(x_f.*obj.f_dec.H);
            
            if isreal(x_f)
                y = real(y);
            end
        end
        
        % Single Level Redundant Wavelet Reconstruction
        function y = level_1_rec(obj,x_f)
            
            % Reconstruct the 3D array using Fast Convolution
            y = ifft(squeeze(x_f(:,1)).*conj(obj.f_dec.L));
            y = y + ifft(squeeze(x_f(:,2)).*conj(obj.f_dec.H));
            
            if isreal(x_f)
                y = real(y);
            end
        end
    end
    
end

