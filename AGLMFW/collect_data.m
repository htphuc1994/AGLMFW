% -------------------------------------------------------------------------
% Collect best-ACC rows from every .mat file in the folder
% -------------------------------------------------------------------------
% files = dir(fullfile(pwd,'*.mat'));          
% Entropy_soft_min;Hedge;Power;Dirichlet;Sparsemax;EntropyAdam;UCB;LCurve;Entmax15;EntropyAdaptive
files = dir(fullfile(pwd, 'Res/YALE-*-weight=EntropyAdaptive.mat'));

bestPerFile = table( ...
    'Size',   [numel(files) 7], ...
    'VariableTypes', ["string" repmat("double",1,6)], ...
    'VariableNames', {'File' 'ACC' 'NMI' 'Purity' 'Fscore' 'Precision' 'AR'});

for k = 1:numel(files)
    data  = load(fullfile(files(k).folder, files(k).name));

    %--- grab the first numeric array inside the .mat ---------------------
    fldNames = fieldnames(data);
    res      = data.(fldNames{1});           % matrix with the 8 metrics
    % (If your files store the matrix under a fixed name, replace this line
    %  with:  res = data.resultsMatrix;  etc.)

    %--- find the row with the best ACC -----------------------------------
    [~, idx] = max(res(:,7));                % 7-th column is ACC
    bestRow  = res(idx,:);

    %--- pack the six requested numbers in the specified order ------------
    bestPerFile{k,"File"}      = string(files(k).name);
    bestPerFile{k,{'ACC','NMI','Purity','Fscore','Precision','AR'}} = ...
        bestRow([7 4 8 1 2 5]);
end

disp(bestPerFile)

fprintf('%50s  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f\n', 'File', 'ACC', 'NMI', 'Purity', 'Fscore', 'Precision', 'AR');
for i = 1:height(bestPerFile)
    fprintf('%50s  & %6.4f & %6.4f & %6.4f & %6.4f & %6.4f & %6.4f\n', ...
        bestPerFile.File(i), ...
        bestPerFile.ACC(i), ...
        bestPerFile.NMI(i), ...
        bestPerFile.Purity(i), ...
        bestPerFile.Fscore(i), ...
        bestPerFile.Precision(i), ...
        bestPerFile.AR(i));
end