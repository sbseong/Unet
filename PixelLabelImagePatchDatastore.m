% PixelLabelImagePatchDatastore  Image patch data source for training
% Semantic segmentation networks.
%
%   A PixelLabelImagePatchDatastore object encapsulates a data source that
%   creates batches of image patches along with corresponding ground truth
%   label patches to feed into a semantic segmentation network.
%
%   PixelLabelImagePatchDatastore properties:
%       MiniBatchSize           - Number of patches in a minibatch
%       BatchesPerImage         - Number of batches of patches to be
%                                 extracted per image 
%       PatchSize               - Size of the image patches
%
%   PixelLabelImagePatchDatastore methods:
%       PixelLabelImagePatchDatastore - Create a PixelLabelImagePatchDatastore
%

%   Copyright 2018 The MathWorks, Inc.

classdef PixelLabelImagePatchDatastore < matlab.io.Datastore &...
        matlab.io.datastore.MiniBatchable &...
        matlab.io.datastore.Shuffleable
      
    properties
        
        MiniBatchSize
        
    end
    
    properties (SetAccess = protected)
        
        NumObservations
        
    end
    
    properties (Access = private)
        
        InputImageDatastore
        
        InputPixelLabelDatastore
        
        BatchesPerImage
        
        PatchSize
        
        CurrentFullImage
        
        CurrentLabelImage
        
        CurrentImageIndex
        
        CurrentMiniBatchIndex
        
        NumBatchesReadFromCurrentImage        
        
    end
    
    methods
        
        function ds = PixelLabelImagePatchDatastore(imds, pxds, varargin)
            %
            %   source = PixelLabelImagePatchDatastore(imds, pxds)
            %            creates a PixelLabelImagePatchDatastore, which
            %            creates batches of image patches along with
            %            corresponding ground truth label patches as a data
            %            source for a semantic segmentation network.
            %
            %   source = PixelLabelImagePatchDatastore(__, Name, Value,__)
            %            creates a PixelLabelImagePatchDatastore with additional
            %            parameters controlling the data generation
            %            process.
            %
            %   Parameters are:
            %
            %   MiniBatchSize             : Integer specifying the size of
            %                               the minibatch.
            %                               Default is 64.
            %
            %   BatchesPerImage           : Integer specifying the number
            %                               of batches generated from an
            %                               image.
            %                               Default is 1.
            %
            %   PatchSize                 : Size of the random crops. It
            %                               can be an integer scalar
            %                               specifying same row and column
            %                               sizes or a two element integer
            %                               vector specifying different row
            %                               and column sizes.
            %                               Default is 50.
            %
            %   NOTE: This function requires the Deep Learning Toolbox (TM).
                       
            narginchk(2,8);
            
            validateImagedatastore(imds);
            validatePixelLabeldatastore(pxds);
            
            options = parseInputs(varargin{:});
            
            ds.BatchesPerImage = options.BatchesPerImage;
            
            ds.MiniBatchSize = options.MiniBatchSize;
            
            ds.NumObservations = length(imds.Files) * ds.MiniBatchSize * ds.BatchesPerImage;
            
            if isscalar(options.PatchSize)
                ds.PatchSize = [options.PatchSize options.PatchSize];
            else
                ds.PatchSize = options.PatchSize;
            end
            
            ds.InputImageDatastore = imds.copy();
            ds.InputPixelLabelDatastore = pxds.copy();
                        
            ds.reset();
            
        end
        
    end
        
    methods
        
        function [miniBatchTable,info] = read(ds)
            
            if (ds.NumBatchesReadFromCurrentImage == ds.BatchesPerImage)
                
                ds.NumBatchesReadFromCurrentImage = 0;
                
                ds.CurrentImageIndex = ds.CurrentImageIndex + 1;
                
                ds.CurrentFullImage = ds.InputImageDatastore.readimage(ds.CurrentImageIndex);
                ds.CurrentLabelImage = ds.InputPixelLabelDatastore.readNumeric(ds.CurrentImageIndex);
                                
            end

            I = ds.CurrentFullImage;
            labelI = ds.CurrentLabelImage{1};
            
            imPatches    = cell(ds.MiniBatchSize,1);
            labelPatches = cell(ds.MiniBatchSize,1);
            
            for i = 1:ds.MiniBatchSize
                
                [imPatches{i}, labelPatches{i}] = randcrop(I,labelI,ds.PatchSize);
                
                
                %% Apply augmentation randomly
                switch randi([1 3],1)
                    case 1
                        % No augmentation
                        
                    case 2
                        % Rotate randomly by [30, 45, 60, 90] degrees
                        angles = [30, 45, 60, 90];
                        ang_idx = randi([1 4],1);
                        imPatches{i} = imrotate(imPatches{i},...
                                                    angles(ang_idx),...
                                                    'nearest',...
                                                    'crop');
                        labelPatches{i} = imrotate(labelPatches{i},...
                                                    angles(ang_idx),...
                                                    'nearest',...
                                                    'crop');
                        
                    case 3
                        % flip in X-Y direction
                        imPatches{i} = fliplr(imPatches{i});
                        labelPatches{i} = fliplr(labelPatches{i});
                        
                    otherwise
                        error('Unexpected error')
                end

                
            end
            
            ds.NumBatchesReadFromCurrentImage = ds.NumBatchesReadFromCurrentImage + 1;
            
            ds.CurrentMiniBatchIndex = ds.CurrentMiniBatchIndex + 1;
            
            % convert numeric to categorical
            classNames = ds.InputPixelLabelDatastore.ClassNames;
            label_idxs = 1:length(classNames);
            for i = 1:length(labelPatches)
                labelPatches{i} = categorical(labelPatches{i}, label_idxs, classNames);
            end
            
            miniBatchTable = [table(imPatches) table(labelPatches)];
            
            info.CurrentImageIndexFromDatastore = ds.CurrentImageIndex;
            
            info.BatchIndexFromCurrentImage = ds.NumBatchesReadFromCurrentImage;
            
        end
        
        function TF = hasdata(ds)
            
            outOfData = (ds.CurrentImageIndex >= length(ds.InputImageDatastore.Files)) &&...
                (ds.NumBatchesReadFromCurrentImage >= ds.BatchesPerImage);
            
            TF = ~outOfData;
            
        end
        
        function reset(ds)
            
            ds.CurrentFullImage = ds.InputImageDatastore.readimage(1);
            ds.CurrentLabelImage = ds.InputPixelLabelDatastore.readNumeric(1);
            
            ds.CurrentImageIndex = 1;
            
            ds.NumBatchesReadFromCurrentImage = 0;
            
        end
        
    end
    
    
    
    methods (Hidden)
        function frac = progress(ds)
            frac = ds.CurrentMiniBatchIndex / ds.TotalNumberOfMiniBatches;
        end
    end
    
    % Required method definitions from matlab.io.datastore.Shuffleable
    % mixin.
    
    methods
        
        function dsrand = shuffle(ds)
            
            % To shuffle, shuffle underlying ImageDatastores
            dsrand = copy(ds);
            ord = randperm( numel(ds.InputImageDatastore.Files) );
            
            dsrand.InputImageDatastore.Files = ...
                                dsrand.InputImageDatastore.Files(ord);
            dsrand.InputPixelLabelDatastore.shuffle(ord);
            
        end
        
    end
    
    
    methods(Static, Hidden = true)
        function self = loadobj(S)
            self = PixelLabelImagePatchDatastore(S.InputImageDatastore, ...
                S.InputPixelLabelDatastore, ...
                'MiniBatchSize', S.MiniBatchSize,...
                'BatchesPerImage', S.BatchesPerImage,...
                'PatchSize', [S.PatchSize(1) S.PatchSize(2)]);
        end
    end
    
    methods (Hidden)
        function S = saveobj(self)
            
            % Serialize PixelLabelImagePatchDatastore object
            S = struct('InputImageDatastore',self.InputImageDatastore,...
                'InputPixelLabelDatastore', self.InputPixelLabelDatastore,... 
                'MiniBatchSize',self.MiniBatchSize,...
                'BatchesPerImage',self.BatchesPerImage,...
                'PatchSize',self.PatchSize);
        end
        
    end
    
end


function [cropRawImage, cropLabelImage]  = randcrop(img,imgLabel,cropSize)

patchStartRow = randi([1,size(img,1)-cropSize(1)+1],1);

patchStartCol = randi([1,size(img,2)-cropSize(2)+1],1);

patchEndRow = patchStartRow + cropSize(1) - 1;

patchEndCol = patchStartCol + cropSize(2) - 1;

cropRawImage = img(patchStartRow:patchEndRow,patchStartCol:patchEndCol,:);

cropLabelImage = imgLabel(patchStartRow:patchEndRow,patchStartCol:patchEndCol,:);

end


function B = validateImagedatastore(ds)

validateattributes(ds, {'matlab.io.datastore.ImageDatastore'}, ...
    {'nonempty','vector'}, mfilename, 'IMDS');
validateattributes(ds.Files, {'cell'}, {'nonempty'}, mfilename, 'IMDS');

B = true;

end

function B = validatePixelLabeldatastore(ds)

validateattributes(ds, {'matlab.io.datastore.PixelLabelDatastore'}, ...
    {'nonempty','vector'}, mfilename, 'IMDS');
validateattributes(ds.Files, {'cell'}, {'nonempty'}, mfilename, 'IMDS');

B = true;

end


function options = parseInputs(varargin)

parser = inputParser();
parser.addParameter('BatchesPerImage',1,@validateBatchesPerImage);
parser.addParameter('PatchSize',50,@validatePatchSize);
parser.addParameter('MiniBatchSize',64,@validateMiniBatchSize);

parser.parse(varargin{:});
options = parser.Results;

end


function B = validateBatchesPerImage(BatchesPerImage)

attributes = {'nonempty','real','scalar', ...
    'positive','integer','finite','nonsparse','nonnan','nonzero'};

validateattributes(BatchesPerImage,images.internal.iptnumerictypes, attributes,...
    mfilename,'BatchesPerImage');

B = true;

end


function B = validateMiniBatchSize(miniBatchSize)

attributes = {'nonempty','real','scalar', ...
    'positive','integer','finite','nonsparse','nonnan','nonzero'};

validateattributes(miniBatchSize,images.internal.iptnumerictypes, attributes,...
    mfilename,'MiniBatchSize');

B = true;

end

function B = validatePatchSize(PatchSize)

attributes = {'nonempty','real','vector', ...
    'positive','integer','finite','nonsparse','nonnan','nonzero'};

validateattributes(PatchSize,images.internal.iptnumerictypes, attributes,...
    mfilename,'PatchSize');

if numel(PatchSize) > 2
    error('Invalid PatchSize');
end

B = true;

end