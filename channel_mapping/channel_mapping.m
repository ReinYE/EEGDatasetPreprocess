EEG = pop_chanedit(EEG);  % ensure chanlocs loaded

% =========================
% 1. Get your electrode coordinates (currently CTF, mm)
% =========================
labels = {EEG.chanlocs.labels};
X = [EEG.chanlocs.X]';
Y = [EEG.chanlocs.Y]';
Z = [EEG.chanlocs.Z]';

user_xyz_ctf = [X Y Z];

% =========================
% 2. Convert CTF -> EEGLAB-like head coordinates
% -------------------------
% Use your metadata:
% EEGCoordinateSystem = CTF
% EEGCoordinateUnits  = mm
%
% Here we convert axis directions before comparing
% with EEGLAB standard template.
%
% Common practical conversion:
%   x_eeglab =  x_ctf
%   y_eeglab =  y_ctf
%   z_eeglab =  z_ctf
%
% BUT because coordinate conventions may differ in left-right /
% anterior direction handling across systems, we explicitly test
% a small set of axis transforms and keep the best one globally.
% =========================

candidate_transforms = {
    @(p) [ p(:,1),  p(:,2),  p(:,3)], 'identity';
    @(p) [ p(:,1), -p(:,2),  p(:,3)], 'flipY';
    @(p) [-p(:,1),  p(:,2),  p(:,3)], 'flipX';
    @(p) [-p(:,1), -p(:,2),  p(:,3)], 'flipXY';
};

% =========================
% 3. Load standard template points
% =========================
standard_locs = readlocs('standard-10-5-cap385.elp');

target_names = {'AF3','AF4','F3','F4','FC5','FC6','F7','F8'};

target_idx = find(ismember({standard_locs.labels}, target_names));
target_locs = standard_locs(target_idx);

% keep same order as target_names
[~, order] = ismember(target_names, {target_locs.labels});
target_locs = target_locs(order);

target_xyz = [[target_locs.X]' [target_locs.Y]' [target_locs.Z]'];

% =========================
% 4. Normalize template coordinates
% =========================
target_xyz_norm = target_xyz ./ vecnorm(target_xyz,2,2);

% =========================
% 5. Test transforms and choose the best global match
% =========================
best_total = inf;
best_name = '';
best_dist = [];
best_user_xyz_norm = [];

for t = 1:size(candidate_transforms,1)
    tfun = candidate_transforms{t,1};
    tname = candidate_transforms{t,2};

    user_xyz = tfun(user_xyz_ctf);

    % normalize after coordinate transform
    user_xyz_norm = user_xyz ./ vecnorm(user_xyz,2,2);

    % distance matrix
    dist = pdist2(target_xyz_norm, user_xyz_norm);

    % unique greedy matching
    temp_dist = dist;
    total = 0;
    for i = 1:length(target_names)
        [dmin, idx_linear] = min(temp_dist(:));
        [r, c] = ind2sub(size(temp_dist), idx_linear);
        total = total + dmin;
        temp_dist(r,:) = inf;
        temp_dist(:,c) = inf;
    end

    if total < best_total
        best_total = total;
        best_name = tname;
        best_dist = dist;
        best_user_xyz_norm = user_xyz_norm;
    end
end

fprintf('Best transform = %s\n', best_name);

% =========================
% 6. Final nearest-neighbor mapping
% =========================
fprintf('\n===== Mapping Results =====\n')

used_channels = false(1, length(labels));
mapping = cell(length(target_names), 4); % target, matched, distance, meaning

for i = 1:length(target_names)
    drow = best_dist(i,:);
    drow(used_channels) = inf;
    [d, idx] = min(drow);

    if d < 0.15
        meaning = 'very close match';
    elseif d < 0.30
        meaning = 'good match';
    elseif d < 0.50
        meaning = 'possible match';
    else
        meaning = 'weak match';
    end

    used_channels(idx) = true;
    mapping{i,1} = target_names{i};
    mapping{i,2} = labels{idx};
    mapping{i,3} = d;
    mapping{i,4} = meaning;

    fprintf('%s --> %s (distance=%.4f, %s)\n', ...
        target_names{i}, labels{idx}, d, meaning);
end

% =========================
% 7. Save mapping results
% =========================
T = cell2table(mapping, ...
    'VariableNames', {'Target','MatchedChannel','Distance','Meaning'});

writetable(T, 'mapping_results.csv');
disp('Saved to mapping_results.csv');