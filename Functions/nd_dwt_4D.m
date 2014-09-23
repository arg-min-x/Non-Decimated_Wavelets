%Calculates the nondecimated Wavelet transform using fast convlolution and
%   periodic boundry conditions.  The Filters are pre-computed and stored
%   in f_dec and f_rec to reduce computation time when performing Multiple
%   forward/backward transforms
%
%
%Methods:
%   ud_dwt_4D:  Constructor
%               Inputs: wname - Wavelet Filters to Use i.e. db1,db2,etc.
%                        Either a string 'db1' or a cell  of length 3 e.g. 
%                        {'db4','db1','db8} where the first element in 
%                        the string is filter for the first dimension, the 
%                        second element the seconde dimension, etc
%                       sizes - size of the 3D object [n1,n2,n3]
%
%   dec:        Multilevel Decomposition
%               Inputs: x - Image domain signal for decomposition
%                       levels - Number of decomposition Levels
%               Outputs: y - Multilevel non-decimated DWT coefficients in a
%                        5D array where the data is arranged [n1,n2,n3,n4,bands]
%                        The bands are orginized as follows.  Let "s1s2s3s4"
%                        represent the Horizontal, Vertical, and Temporal
%                        Bands.  The coefficients are ordered as follows,
%                        "LLLL", "HLLL", "LHLL", "HHLL", "LLHL",...,"HHHH
%                        where "H" denotes the high frequency filter and "L" 
%                        represents the low frequency filter. Successive 
%                        levels of decomposition are stacked such that the 
%                        highest "LLLL" is in [n1,n2,n3,n4,1]
%
%   rec:        Multilevel Reconstruction
%               Inputs: x - Wavelet coefficients in a 5D array size 
%                       [n1,n2,n3,n4,bands].
%               Outputs: y - Reconstructed 4D array.
%
%**************************************************************************
% The Ohio State University
% Written by:   Adam Rich 
% Email:        rich.178@osu.edu
% Last update:  8/8/2014
%**************************************************************************

classdef nd_dwt_4D
    properties
        f_dec;          % Decomposition Filters
        f_rec;          % Reconstruction Filters
        sizes;          % Size of the 3D Image
        f_size;         % Length of the filters
        wname;          % Wavelet used
        pres_l2_norm;   % Binary indicator to preserver l2 norm of coefficients
    end
    
    methods
        % Constructor Object
        function obj = nd_dwt_4D(wname,sizes,varargin)
            % Set Image size
            obj.sizes = sizes;
            
            if ischar(wname)
                obj.wname = {wname,wname,wname,wname};
            elseif iscell(wname)
                obj.wname = wname;
            end
            
            if isempty(varargin)
                obj.pres_l2_norm = 0;
            else
                obj.pres_l2_norm = varargin{1};
            end
            
            % Get the Filter Coefficients
            [obj.f_dec,obj.f_size] = obj.get_filters(obj.wname);
            
        end
        
        % Multilevel Undecimated Wavelet Decomposition
        function y = dec(obj,x,level)
            % Fourier Transform of Signal
            x = fftn(x);

            % Preallocate
            y = zeros([obj.sizes, 16+15*(level-1)]);
            
            % Calculate Mutlilevel Wavelet decomposition
            for ind = 1:level
                % First Level
                if ind ==1
                    y = level_1_dec(obj,x);
                % Succssive Levels
                else
                    y = cat(5,level_1_dec(obj,fftn(squeeze(y(:,:,:,:,1)))), y(:,:,:,:,2:end));
                end
            end
                      
        end
        
        % Multilevel Undecimated Wavelet Reconstruction
        function y = rec(obj,x)
            
            % Find the decomposition level
            level = 1+(size(x,5)-16)/15;
            
            % Fourier Transform of Signal
            x = fft(fft(fft(fft(x,[],1),[],2),[],3),[],4);
            
            % Reconstruct from Multiple Levels
            for ind = 1:level
                % First Level
                if ind ==1
                    y = level_1_rec(obj,x);
                    if ~obj.pres_l2_norm
                        y = y/16;
                    end
                % Succssive Levels
                else
                    y = fftn(y);
                    y = level_1_rec(obj,cat(5,y,x(:,:,:,:,17+(ind-2)*15:31+(ind-2)*15)));
                    if ~obj.pres_l2_norm
                        y = y/16;
                    end
                end
            end 
            
        end
    end
    
     %% Private Methods
    methods (Access = protected,Hidden = true)
        
        % Returns the Filters and 
        function [f_dec,f_size] = get_filters(obj,wname)
            % Get Filters for the first domain
            [LO_D,HI_D]   = wave_filters(wname{1});
            [LO_D2,HI_D2] = wave_filters(wname{2});
            [LO_D3,HI_D3] = wave_filters(wname{3});
            [LO_D4,HI_D4] = wave_filters(wname{4});
            
            % Find the filter size
            f_size.s1 = length(LO_D);
            f_size.s2 = length(LO_D2);
            f_size.s3 = length(LO_D3);
            f_size.s4 = length(LO_D4);
            
            % Dimension Check
            if f_size.s1 > obj.sizes(1)
                error(['First Dimension of Data is shorter than the wavelet'...
                    ,' filter being used']);
            elseif f_size.s2 > obj.sizes(2)
                error(['Second Dimension of Data is shorter than the wavelet'...
                    ,' filter being used']);
            elseif f_size.s3 > obj.sizes(3)
                error(['Third Dimension of Data is shorter than the wavelet'...
                    ,' filter being used']);
            elseif f_size.s4 > obj.sizes(4)
                error(['Fourth Dimension of Data is shorter than the wavelet'...
                    ,' filter being used']);
            end
            
            % Get the 2D Filters by taking outer products
            dec_LL = LO_D.'*LO_D2;
            dec_HL = HI_D.'*LO_D2;
            dec_LH = LO_D.'*HI_D2;
            dec_HH = HI_D.'*HI_D2;
            
            % Take the Outerproducts for the third dimension
            dec_LLL = zeros(length(LO_D),length(LO_D2),length(LO_D3));
            dec_HLL = zeros(length(LO_D),length(LO_D2),length(LO_D3));
            dec_LHL = zeros(length(LO_D),length(LO_D2),length(LO_D3));
            dec_HHL = zeros(length(LO_D),length(LO_D2),length(LO_D3));
            dec_LLH = zeros(length(LO_D),length(LO_D2),length(LO_D3));
            dec_HLH = zeros(length(LO_D),length(LO_D2),length(LO_D3));
            dec_LHH = zeros(length(LO_D),length(LO_D2),length(LO_D3));
            dec_HHH = zeros(length(LO_D),length(LO_D2),length(LO_D3));
            
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
            for kk = 1:size(dec_LLL,3)
                for ii = 1:size(dec_LLL,2)
                    f_dec.LLLL(:,ii,kk,:) = dec_LLL(:,ii,kk)*LO_D4;
                    f_dec.HLLL(:,ii,kk,:) = dec_HLL(:,ii,kk)*LO_D4;
                    f_dec.LHLL(:,ii,kk,:) = dec_LHL(:,ii,kk)*LO_D4;
                    f_dec.HHLL(:,ii,kk,:) = dec_HHL(:,ii,kk)*LO_D4;
                    f_dec.LLHL(:,ii,kk,:) = dec_LLH(:,ii,kk)*LO_D4;
                    f_dec.HLHL(:,ii,kk,:) = dec_HLH(:,ii,kk)*LO_D4;
                    f_dec.LHHL(:,ii,kk,:) = dec_LHH(:,ii,kk)*LO_D4;
                    f_dec.HHHL(:,ii,kk,:) = dec_HHH(:,ii,kk)*LO_D4;
                    f_dec.LLLH(:,ii,kk,:) = dec_LLL(:,ii,kk)*HI_D4;
                    f_dec.HLLH(:,ii,kk,:) = dec_HLL(:,ii,kk)*HI_D4;
                    f_dec.LHLH(:,ii,kk,:) = dec_LHL(:,ii,kk)*HI_D4;
                    f_dec.HHLH(:,ii,kk,:) = dec_HHL(:,ii,kk)*HI_D4;
                    f_dec.LLHH(:,ii,kk,:) = dec_LLH(:,ii,kk)*HI_D4;
                    f_dec.HLHH(:,ii,kk,:) = dec_HLH(:,ii,kk)*HI_D4;
                    f_dec.LHHH(:,ii,kk,:) = dec_LHH(:,ii,kk)*HI_D4;
                    f_dec.HHHH(:,ii,kk,:) = dec_HHH(:,ii,kk)*HI_D4;
                end
            end

            % Add a circularshift of half the filter length to the 
            % reconstruction filters by adding phase to them
            phase1 = exp(1j*2*pi*f_size.s1/2*linspace(0,1-1/obj.sizes(1),obj.sizes(1)));
            phase2 = exp(1j*2*pi*f_size.s2/2*linspace(0,1-1/obj.sizes(2),obj.sizes(2)));
            phase3 = exp(1j*2*pi*f_size.s3/2*linspace(0,1-1/obj.sizes(3),obj.sizes(3)));
            phase4 = exp(1j*2*pi*f_size.s4/2*linspace(0,1-1/obj.sizes(4),obj.sizes(4)));

            % 2D Phase
            shift_2 = phase1.'*phase2;
            
            % 3D Phase
            shift_3 = zeros([obj.sizes(1),obj.sizes(2),obj.sizes(3)]);
            for ind = 1:size(shift_2,2)
               shift_3(:,ind,:) = shift_2(:,ind)*phase3; 
            end
            
            % 4D Phase
            shift = zeros([obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            for ii = 1:size(shift_3,2)
                for kk = 1:size(shift_3,3)
                    shift(:,ii,kk,:) = shift_3(:,ii,kk)*phase4;
                end
            end
            
            % Take the Fourier Transform of the Kernels for Fast
            % Convolution
            if obj.pres_l2_norm 
                scale = 1/sqrt(16);
            else
                scale = 1;
            end
            f_dec.LLLL = scale*shift.*fftn(f_dec.LLLL,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.HLLL = scale*shift.*fftn(f_dec.HLLL,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.LHLL = scale*shift.*fftn(f_dec.LHLL,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.HHLL = scale*shift.*fftn(f_dec.HHLL,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.LLHL = scale*shift.*fftn(f_dec.LLHL,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.HLHL = scale*shift.*fftn(f_dec.HLHL,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.LHHL = scale*shift.*fftn(f_dec.LHHL,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.HHHL = scale*shift.*fftn(f_dec.HHHL,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.LLLH = scale*shift.*fftn(f_dec.LLLH,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.HLLH = scale*shift.*fftn(f_dec.HLLH,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.LHLH = scale*shift.*fftn(f_dec.LHLH,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.HHLH = scale*shift.*fftn(f_dec.HHLH,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.LLHH = scale*shift.*fftn(f_dec.LLHH,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.HLHH = scale*shift.*fftn(f_dec.HLHH,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.LHHH = scale*shift.*fftn(f_dec.LHHH,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
            f_dec.HHHH = scale*shift.*fftn(f_dec.HHHH,[obj.sizes(1),obj.sizes(2),obj.sizes(3),obj.sizes(4)]);
        end
        
        % Single Level Redundant Wavelet Decomposition
        function y = level_1_dec(obj,x_f)
            % Preallocate
            y = zeros([obj.sizes,16]);
            
            % Calculate Wavelet Coefficents Using Fast Convolution
            y(:,:,:,:,1) =  ifftn(x_f.*obj.f_dec.LLLL);
            y(:,:,:,:,2) =  ifftn(x_f.*obj.f_dec.HLLL);
            y(:,:,:,:,3) =  ifftn(x_f.*obj.f_dec.LHLL);
            y(:,:,:,:,4) =  ifftn(x_f.*obj.f_dec.HHLL);
            y(:,:,:,:,5) =  ifftn(x_f.*obj.f_dec.LLHL);
            y(:,:,:,:,6) =  ifftn(x_f.*obj.f_dec.HLHL);
            y(:,:,:,:,7) =  ifftn(x_f.*obj.f_dec.LHHL);
            y(:,:,:,:,8) =  ifftn(x_f.*obj.f_dec.HHHL);
            y(:,:,:,:,9) =  ifftn(x_f.*obj.f_dec.LLLH);
            y(:,:,:,:,10) = ifftn(x_f.*obj.f_dec.HLLH);
            y(:,:,:,:,11) = ifftn(x_f.*obj.f_dec.LHLH);
            y(:,:,:,:,12) = ifftn(x_f.*obj.f_dec.HHLH);
            y(:,:,:,:,13) = ifftn(x_f.*obj.f_dec.LLHH);
            y(:,:,:,:,14) = ifftn(x_f.*obj.f_dec.HLHH);
            y(:,:,:,:,15) = ifftn(x_f.*obj.f_dec.LHHH);
            y(:,:,:,:,16) = ifftn(x_f.*obj.f_dec.HHHH);
            
        end
        
        % Single Level Redundant Wavelet Reconstruction
        function y = level_1_rec(obj,x_f)
            
            % Reconstruct the 3D array using Fast Convolution
            y = ifftn(squeeze(x_f(:,:,:,:,1)).*conj(obj.f_dec.LLLL));
            y = y + ifftn(squeeze(x_f(:,:,:,:,2)).*conj(obj.f_dec.HLLL));
            y = y + ifftn(squeeze(x_f(:,:,:,:,3)).*conj(obj.f_dec.LHLL));
            y = y + ifftn(squeeze(x_f(:,:,:,:,4)).*conj(obj.f_dec.HHLL));
            y = y + ifftn(squeeze(x_f(:,:,:,:,5)).*conj(obj.f_dec.LLHL));
            y = y + ifftn(squeeze(x_f(:,:,:,:,6)).*conj(obj.f_dec.HLHL));
            y = y + ifftn(squeeze(x_f(:,:,:,:,7)).*conj(obj.f_dec.LHHL));
            y = y + ifftn(squeeze(x_f(:,:,:,:,8)).*conj(obj.f_dec.HHHL));
            y = y + ifftn(squeeze(x_f(:,:,:,:,9)).*conj(obj.f_dec.LLLH));
            y = y + ifftn(squeeze(x_f(:,:,:,:,10)).*conj(obj.f_dec.HLLH));
            y = y + ifftn(squeeze(x_f(:,:,:,:,11)).*conj(obj.f_dec.LHLH));
            y = y + ifftn(squeeze(x_f(:,:,:,:,12)).*conj(obj.f_dec.HHLH));
            y = y + ifftn(squeeze(x_f(:,:,:,:,13)).*conj(obj.f_dec.LLHH));
            y = y + ifftn(squeeze(x_f(:,:,:,:,14)).*conj(obj.f_dec.HLHH));
            y = y + ifftn(squeeze(x_f(:,:,:,:,15)).*conj(obj.f_dec.LHHH));
            y = y + ifftn(squeeze(x_f(:,:,:,:,16)).*conj(obj.f_dec.HHHH));

        end
    end
end

