function downloadTrainedUnet(url, destination)
% DOWNLOADTRAINEDUNET Helper function to download the pre-trained U-net
% network for the multispectral data-set downloaded from the link below.
%
%  'http://www.cis.rit.edu/~rmk6217/rit18_data.mat'
%

%   Copyright 2017 The MathWorks, Inc.

filename = 'multispectralUnet.mat';
imageDirFullPath = fullfile(destination,'trainedUnet');
imageFileFullPath = fullfile(imageDirFullPath,filename);

if ~exist(imageFileFullPath,'file')
    fprintf('Downloading Pre-trained U-net for Hamlin Beach dataset...\n');
    fprintf('This will take several minutes to download...\n');
    mkdir(imageDirFullPath);
    websave(imageFileFullPath,url);
    fprintf('done.\n\n');
end
end