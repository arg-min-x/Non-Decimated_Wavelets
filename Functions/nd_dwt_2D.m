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
%
%                       sizes - size of the 2D object [n1,n2]
%
%               Optional Inputs:
%                        pres_l2_norm -If set TRUE, the l2 norm in the 
%                        wavelet domain will be equal to the l2 norm in the
%                        signal domain.  Default is FALSE
% 
%                        compute - A string that sets the method of
%                        computation.  'mat' computes the wavelet tranform
%                        in Matlab.  'mex' computes the wavelet transfrom
%                        using a c mex file.  'gpu' computes the wavelets
%                        using matlab on a the GPU. When using 'gpu' the 
%                        input is expected to be a gpuArray and the output 
%                        also a gpuArray.  'gpu_off' computes the wavelets 
%                        using the gpu in matlab.  The calculation is offloaded
%                        to the GPU for calculation and brought back to 
%                        system memory afterwords. The input and output in 
%                        this case are both 'double' or 'single' depending
%                        on the precision used (see below)
%
%                        precision - a string that specifies double or
%                        single precision computation.  'double' or
%                        'single' are valid inputs
%
%   dec:        Multilevel Decomposition
%               Inputs: x - Image domain signal for decomposition
%                       levels - Number of decomposition Levels
%               Outputs: y - Multilevel non-decimated DWT coefficients in a
%                        4D array where the data is arranged [n1,n2,bands]
%                        The bands are orginized as follows.  Let "HVT"
%                        represent the Horizontal, and Vertical Bands.  The
%                        coefficients are ordered as follows, "LL", "HL", 
%                        "LH", "HH", where "H" denotes the high frequency 
%                        filter and L represents the low frequency filter. 
%                        Successive levels of decomposition are stacked such 
%                        that the highest "LL" is in [n1,n2,1]
%
%   rec:        Multilevel Reconstruction
%               Inputs: x - Wavelet coefficients in a 3D array size 
%                       [n1,n2,bands].
%
%               Outputs: y - Reconstructed 2D array.%   
%
%**************************************************************************
% The Ohio State University
% Written by:   Adam Rich 
% Last update:  2/5/2015
%**************************************************************************

classdef nd_dwt_2D
    properties
        f_dec;          % Decomposition Filters
        sizes;          % Size of the 3D Image
        f_size;         % Length of the filters
        wname;          % Wavelet used
        pres_l2_norm;   % Binary indicator to preserver l2 norm of coefficients
        compute;        % How to compute the wavelet transform
        precision;      % which precision to use
    end
    
    %% Public Methods
    methods
        % Constructor Object
        function obj = nd_dwt_2D(wname,sizes,varargin)
            % Set Image size
            if length(sizes) ~=2
                error('The sizes vector must be length 2');
            else
                obj.sizes = sizes;
            end
            
            % Check wname input
            if mod(length(varargin),2)
                error('Optional inputs must come in pairs')
            end
            if ischar(wname)
                obj.wname = {wname,wname};
            elseif iscell(wname)
                if length(wname) ==2
                    obj.wname = wname;
                else
                    error(['You must specify two filter names in a cell array'...
                            ,'of length 2, or a single string for the same'...
                            ,' filter to be used in all dimensions']);
                end
            end
            
            % Set Default Options
            obj.pres_l2_norm = 0;
            obj.precision = 'double';
            obj.compute = 'mat';
            
            % Copy any optional inputs 
            for ind = 1:2:length(varargin)
                switch lower(varargin{ind})
                    case 'pres_l2_norm'
                        obj.pres_l2_norm = varargin{ind+1};
                    case 'compute'
                        obj.compute = varargin{ind+1};
                    case 'precision'
                        obj.precision = varargin{ind+1};
                    otherwise
                        warning(sprintf('Unknown optional input #%d ingoring!',ind))
                end
            end
            
            if strcmpi(obj.compute,'mex') && strcmpi(obj.precision,'single')
                error('Single precsision is not currently supported for mex computation');
            end 
            
            % Get the Filter Coefficients
            [obj.f_dec,obj.f_size] = obj.get_filters(obj.wname);
            
            % Typecast the filters as single for single precision
            if strcmpi(obj.precision,'single')
                obj.f_dec = single(obj.f_dec);
            end
            
            % Put the filters on the GPU if using GPU computing
            if strcmpi(obj.compute,'gpu')
                obj.f_dec = gpuArray(obj.f_dec);
            end
            
        end
        
        % Multilevel Undecimated Wavelet Decomposition
        function y = dec(obj,x,level)
            
            % put the input array on the gpu
            if strcmpi(obj.compute,'gpu_off')
                x = gpuArray(x);
            end
            
            % Check if input is real
            if isreal(x)
                x_real = 1;
            else
                x_real = 0;
            end
            
            % Fourier Transform of Signal
            x = fft2(x);
            
            % Use Mex computation
            if strcmpi(obj.compute,'mex')
                y = nd_dwt_mex(x,obj.f_dec,0,level,obj.pres_l2_norm);
                
            % Use Matlab  
            else
                % Preallocate
                if ( strcmpi(obj.compute,'gpu_off') || strcmpi(obj.compute,'gpu') ) && strcmpi(obj.precision,'single')
                    y = zeros([obj.sizes, 4+3*(level-1)],'single');
                    y = gpuArray(y);
                elseif ( strcmpi(obj.compute,'gpu_off') || strcmpi(obj.compute,'gpu') ) && ~strcmpi(obj.precision,'single')
                    y = zeros([obj.sizes, 4+3*(level-1)],'gpuArray');
                elseif strcmpi(obj.precision,'single')
                    y = zeros([obj.sizes, 4+3*(level-1)],'single');
                else
                    y = zeros([obj.sizes, 4+3*(level-1)]);
                end

                % Calculate Mutlilevel Wavelet decomposition
                for ind = 1:level
                    % First Level
                    if ind ==1
                        y = level_1_dec(obj,x);
                    % Succssive Levels
                    else
                        y = cat(3,level_1_dec(obj,fftn(squeeze(y(:,:,1)))), y(:,:,2:end));
                    end
                end
            end
            
            % Take the real part if the input was real
            if x_real
                y = real(y);
            end
            
            % put arrays back in cpu memory
            if strcmpi(obj.compute,'gpu_off')
                y = gather(y);
            end
        end
        
        % Multilevel Undecimated Wavelet Reconstruction
        function y = rec(obj,x)
            
            % put the input array on the gpu
            if strcmpi(obj.compute,'gpu_off')
                x = gpuArray(x);
            end

            % Check if input is real
            if isreal(x)
                x_real = 1;
            else
                x_real = 0;
            end
            
            % Find the decomposition level
            level = 1+(size(x,3)-4)/3;
            
            % Fourier Transform of Signal
            x = fft2(x);
            
            % Use c mex version if chosen
            if strcmpi(obj.compute,'mex')
                y = nd_dwt_mex(x,obj.f_dec,1,level,obj.pres_l2_norm);
            else
                
                % Reconstruct from Multiple Levels
                for ind = 1:level
                    % First Level
                    if ind ==1
                        y = level_1_rec(obj,x);
                        if ~obj.pres_l2_norm
                            y = y/4;
                        end
                    % Succssive Levels
                    else
                        y = fftn(y);
                        y = level_1_rec(obj,cat(3,y,x(:,:,5+(ind-2)*3:7+(ind-2)*3)));
                        if ~obj.pres_l2_norm
                            y = y/4;
                        end
                    end
                end 
            end
            
            % Take the real part if the input was real
            if x_real
                y = real(y);
            end
            % put arrays back in cpu memory
            if strcmpi(obj.compute,'gpu_off')
                y = gather(y);
            end
        end
    end
    
    %% Private Methods
    methods (Access = protected,Hidden = true)
        
        % Returns the Filters and 
        function [f_dec,f_size] = get_filters(obj,wname)
        % Decomposition Filters
        
            % Get the filters
            [LO_D,HI_D] = wave_filters(wname{1});
            [LO_D2,HI_D2] = wave_filters(wname{2});
            
            % Find the filter size
            f_size.s1 = length(LO_D);
            f_size.s2 = length(LO_D2);
            
            % Dimension Check
            if f_size.s1 > obj.sizes(1)
                error(['First Dimension of Data is shorter than the wavelet'...
                    ,' filter being used']);
            elseif f_size.s2 > obj.sizes(2)
                error(['Second Dimension of Data is shorter than the wavelet'...
                    ,' filter being used']);
            end
            
            % Get the 2D Filters by taking outer products
            dec_LL = LO_D.'*LO_D2;
            dec_HL = HI_D.'*LO_D2;
            dec_LH = LO_D.'*HI_D2;
            dec_HH = HI_D.'*HI_D2;

            % Add a circularshift of half the filter length to the 
            % reconstruction filters by adding phase to them
            phase1 = exp(1j*2*pi*f_size.s1/2*linspace(0,1-1/obj.sizes(1),obj.sizes(1)));
            phase2 = exp(1j*2*pi*f_size.s2/2*linspace(0,1-1/obj.sizes(2),obj.sizes(2)));

            % 2D Phase
            shift = phase1.'*phase2;
            
            % Take the Fourier Transform of the Kernels for Fast
            % Convolution
            if obj.pres_l2_norm 
                scale = 1/2;
            else
                scale = 1;
            end
            if strcmpi(obj.compute,'mex')
                scale2 = 1/prod(obj.sizes);
            else
                scale2 = 1;
            end
            f_dec(:,:,1) = scale2*scale*shift.*fftn(dec_LL,[obj.sizes(1),obj.sizes(2)]);
            f_dec(:,:,2) = scale2*scale*shift.*fftn(dec_HL,[obj.sizes(1),obj.sizes(2)]);
            f_dec(:,:,3) = scale2*scale*shift.*fftn(dec_LH,[obj.sizes(1),obj.sizes(2)]);
            f_dec(:,:,4) = scale2*scale*shift.*fftn(dec_HH,[obj.sizes(1),obj.sizes(2)]);
        end
        
        % Single Level Redundant Wavelet Decomposition
        function y = level_1_dec(obj,x_f)
            % Preallocate
            if ( strcmpi(obj.compute,'gpu_off') || strcmpi(obj.compute,'gpu') ) && strcmpi(obj.precision,'single')
                y = zeros([obj.sizes,4],'single');
                y = gpuArray(y);
            elseif ( strcmpi(obj.compute,'gpu_off') || strcmpi(obj.compute,'gpu') ) && ~strcmpi(obj.precision,'single')
                y = zeros([obj.sizes,4],'gpuArray');
            elseif strcmpi(obj.precision,'single')
                y = zeros([obj.sizes,4],'single');
            else
                y = zeros([obj.sizes,4]);
            end
            
            % Calculate Wavelet Coefficents Using Fast Convolution
            y(:,:,1) = ifftn(x_f.*obj.f_dec(:,:,1));
            y(:,:,2) = ifftn(x_f.*obj.f_dec(:,:,2));
            y(:,:,3) = ifftn(x_f.*obj.f_dec(:,:,3));
            y(:,:,4) = ifftn(x_f.*obj.f_dec(:,:,4));
            
        end
        
        % Single Level Redundant Wavelet Reconstruction
        function y = level_1_rec(obj,x_f)
            % Reconstruct the 3D array using Fast Convolution
            y = ifftn( sum(x_f(:,:,1:4).*conj(obj.f_dec),3) );
        end
    end
end

