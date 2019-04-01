function downloadHamlinBeachMSIData(url, destination)
% DOWNLOADHAMLINBEACHMSIDATA Helper function to download the labeled Hamlin 
% beach multispectral dataset
%
%  'http://www.cis.rit.edu/~rmk6217/rit18_data.mat'
%
% References 
% ---------- 
%
% Ronald Kemker, Carl Salvaggio & Christopher Kanan (2017). High-Resolution 
% Multispectral Dataset for Semantic Segmentation. CoRR, abs/1703.01918.

%   Copyright 2017 The MathWorks, Inc.

filename = 'rit18_data.mat';
imageDirFullPath = fullfile(destination,'rit18_data');
imageFileFullPath = fullfile(imageDirFullPath,filename);

if ~exist(imageFileFullPath,'file')
    fprintf('Downloading Hamlin Beach dataset...\n');
    fprintf('This will take several minutes to download...\n');
    mkdir(imageDirFullPath);
    websave(imageFileFullPath,url);
    fprintf('done.\n\n');
end
end