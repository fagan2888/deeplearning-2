function patches = sampleIMAGES(images, patchDim, patchNum)

patches = zeros(patchDim * patchDim, patchNum);
patches(:) = images(1:patchDim * patchDim * patchNum);
% patches = normalizeData(patches);
imageDim = size(images,1);
imageNum = size(images,3);

n = 1;
while n <= patchNum
    for x=1:patchDim:imageDim
        for y=1:patchDim:imageDim
            for i=1:imageNum
                pat = images(x:x+patchDim - 1,y:y+patchDim-1,i);
                patches(:,n) = pat(:);
                n = n + 1;
                if n > patchNum
                    return;
                end
                
                
            end
        end
    end
end



end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
