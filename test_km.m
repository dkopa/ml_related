clear all; clc;

numDim = 2
numClass = 2
numObs = 1000

numIt = 100
eps = 0.0001

X_1 = (0.05 * randn(numObs/2, numDim)) + 0.1;
X_2 = (0.05 * randn(numObs/2, numDim)) + 0.3;

X = [X_1; X_2];

gtLabels = randi(numClass, [1, numObs]);
plot(X(:,1), X(:,2), '*');
means = rand(numClass, numDim);
tDist = zeros(numObs, numClass);
tMean = zeros(numClass, numDim);

% diffMean = list()
for itc = 1:numIt
    itc
    % Compute distances
    for j = 1:numClass
        tDist(:,j) = sqrt(sum(bsxfun(@minus, X, means(j, :)).^2, 2));
    end
    
    [~, tLabels] = min(tDist, [], 2);
        
    % Update mean and compute objective score
    objDist = 0;
    for j = 1:numClass
        clSamples = X(tLabels==j,:);
        tMean(j,:) = mean(clSamples);
        objDist = objDist + sum(sqrt(sum(bsxfun(@minus, clSamples, tMean(j,:)).^2, 2)))
    end
    
    diffMean(itc) = sqrt(sum(sum((means - tMean).^2)));

    if(diffMean(itc) < eps)
        break
    end
    diffMean(itc)
    %print(objDist)
    %print(tMean)
    %print(np.sqrt(np.sum(np.abs(means - tMean))))
    display('===')
    means
    tMean
    means = tMean;
end

plot(diffMean, '-*')