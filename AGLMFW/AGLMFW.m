function [y_idx, Tobj] = AGLMFW(Ss, Fs, opts, r2Temp)
% ---------------------------------------------------------------------
%  Adaptive-Graph Laplacian Multi-view Fusion with
%  automatic view weights
% ---------------------------------------------------------------------
k        = opts.k;                 % #clusters / embedding dim
v        = length(Ss);             % #views
[n,  ~]  = size(Ss{1});
max_iter = 60;

% ---------- warm-start from inputs (unchanged) -----------------------
for idx = 1:v
    Rs{idx} = Fs{idx};
    M{idx}  = Ss{idx};
    A0{idx} = M{idx} - diag(diag(M{idx}));
    u{idx}  = ones(1,n);           % NEW – initial weights for inner loop
end

% ---------- NEW: automatic-weight book-keeping -----------------------
alpha      = ones(1,v)/v;          % start uniform
alphaPrev  = alpha;
tau        = [];   tau0 = [];

% ===================== outer optimisation ============================
for iter = 1:max_iter
    % ----- (a) update common representation Y ------------------------
    T = zeros(n,k);
    for idx = 1:v
        T = T + alpha(idx) * (M{idx} * Rs{idx});   % Σ α_v S_v R_v
    end
    Y = max(T, 0);                                 % non-neg constraint

    % ----- (b) update per-view factors R -----------------------------
    for idx = 1:v
        [Ur,~,Vr] = svds(M{idx}' * Y, k);
        Rs{idx}   = Ur * Vr';
    end

    % ----- (c) update graphs M (unchanged except  u initialised) -----
    for idx = 1:v
        temp = Rs{idx} * Y';
        for i = 1:n
            ai = A0{idx}(i,:);
            di = temp(i,:);
            ei = zeros(1,n);  ei(i) = 1;
            lambda = r2Temp;

            for ii = 1    % same single Newton step you had
                ad = u{idx} .* ai + lambda * di;
                si = EProjSimplexdiag(ad, u{idx} + (lambda/2) * ei);
                u{idx} = 1 ./ (2 * sqrt((si - ai).^2 + eps));
            end
            M{idx}(i,:) = si;
        end
    end

    % ----- (d) per-view loss and overall objective -------------------
    e   = zeros(1,v);
    obj = 0;
    for idx = 1:v
        loss     = norm(M{idx} - Y * Rs{idx}', 'fro')^2 ...
            + norm(Ss{idx} - M{idx}, 1);
        e(idx) = loss;
        obj     = obj + loss;
    end
    Tobj(iter) = obj;

    % ----- (e) automatic weights -----------------------------
    % if iter == 1
    %   tau0 = 0.05 * max(std(e), 1e-12);   % avoid τ = 0
    %   tau  = tau0;
    % elseif norm(alpha - alphaPrev) >= 1e-3  % still moving ⇒ cool
    %   tau  = 0.9 * tau;
    % end
    % alphaPrev = alpha;
    % alpha     = UpdateViewWeights_Entropy(e, tau);
    % alpha = UpdateViewWeights_Hedge(e, alphaPrev, 0.6);

    %%%% ---
    % if iter==1
    %   alpha = ones(1,v)/v;          % start uniform
    % else
    %   alpha = UpdateViewWeights_Hedge(e, alpha, 0.3);
    % end
    % alphaPrev = alpha;                % still useful for convergence test
    %%%% ---

    % alphaPrev = alpha;
    % alpha     = UpdateViewWeights_Power(e, 1);   % p=1 is a good start
    %%%% ---

    alphaPrev = alpha;
    mTau = 0.1 * (max(e) - min(e));
    alpha     = UpdateViewWeights_Dirichlet(e, mTau, 0.02);
    %%%% ---

    % if iter==1
    %   tau = .05*std(e);                       % rough seed
    % end
    % [alpha, tau] = UpdateViewWeights_EntropyAdaptive(e, tau, 0.5*log(v));
    % alphaPrev    = alpha;
    %%%% ---

    % alphaPrev = alpha;
    % alpha     = UpdateViewWeights_Sparsemax(e);
    %%%% ---

    % alphaPrev = alpha;
    % alpha     = UpdateViewWeights_Entmax15(e);
    %%%% ---

    %% GOOODDDD
    % alphaPrev = alpha;
    % alpha     = UpdateViewWeights_Tempered(e, 2);   % T = 2 is a mild start

    % --- inside the outer iteration, after you computed the loss vector e ---
    % if iter==1
    %     Tschedule = 2;                      % start mildly sharp
    % else
    %     % optional: adapt T (e.g. cool it) if you like
    %     Tschedule = 0.95*Tschedule;
    % end
    % alphaPrev = alpha;
    % alpha     = UpdateViewWeights_Tempered(e, Tschedule);
    %%%% ---


    % if iter==1, m = ones(1,v)/v; end
    % [alpha, m] = UpdateViewWeights_EntropyAdam(e, 0.1, m, 0.9);
    % alphaPrev  = alpha;
    %%%% ---


    % if iter==1, pulls = zeros(1,v); end
    %   [alpha, pulls] = UpdateViewWeights_UCB(e, pulls, iter, 1);
    % alphaPrev      = alpha;
    % %%%% ---

    % alphaPrev = alpha;
    % alpha     = UpdateViewWeights_Lcurve(e, 0.05);   % γ = 0.05
    %%%% ---

    % alphaPrev = alpha;
    % alpha     = UpdateViewWeights_InverseVar(e);
    %%%% ---



    % ----- (f) convergence check -------------------------------------
    if iter > 1 && abs(obj - Tobj(iter-1)) / Tobj(iter-1) < 1e-8
        Tobj = Tobj(1:iter);
        break;
    end
end

% ---------------------- final cluster labels -------------------------
[~, y_idx] = max(Y, [], 2);
end
% =====================================================================

% ---------------------------------------------------------------------
function alpha = UpdateViewWeights_Entropy(e, tau)
% soft-min on the simplex: α_v ∝ exp(−(e_v−min e)/τ)
z = exp(-(e - min(e)) / tau);      % shift ⇒ numerical stability
alpha = z / sum(z);
end

function alpha = UpdateViewWeights_Hedge(e, alphaPrev, eta)
% eta : learning-rate (e.g. 0.1 ~ 1)
g        = (e - min(e));          % losses shifted ≥ 0
alphaTmp = alphaPrev .* exp(-eta * g);
alpha    = alphaTmp / sum(alphaTmp);
end

function alpha = UpdateViewWeights_Power(e, p)
% p in (0,∞).  p→∞  → almost hard-min;  p→0 → almost uniform
invLoss = (max(e) - e) .^ p;      % larger loss  → smaller weight
alpha   = invLoss / sum(invLoss);
end

function alpha = UpdateViewWeights_Dirichlet(e, tau, beta)
z     = exp(-(e - min(e))./tau);
alpha = (z + beta) ./ sum(z + beta);   % beta ≈ 0.01 – 0.05
end

function [alpha, tau] = UpdateViewWeights_EntropyAdaptive(e, tauInit, Htarget)
tau = tauInit;
for t = 1:20
    z      = exp(-(e - min(e))/tau);
    alpha  = z / sum(z);
    H      = -sum(alpha .* log(alpha + eps));          % entropy
    g      = H - Htarget;   if abs(g) < 1e-3, return; end
    dH_dt  = dot(alpha, (e - sum(alpha.*e))) / tau^2;  % derivative
    tau    = max(tau - g / (dH_dt+eps), 1e-12);        % Newton step
end
end

function alpha = UpdateViewWeights_Sparsemax(e)
%  α = sparsemax(−e)   (Martins & Astudillo, 2016)
%
%  Input : e  – 1×V loss vector  (row or column OK)
%  Output: α  – 1×V weights on the simplex, some can be exactly 0

z          = -e(:)';                     % row vector of scores
[zsort,~]  = sort(z,'descend');          % sorted scores
K          = numel(zsort);               % scalar count
cssv       = cumsum(zsort);
rho        = find(zsort - (cssv-1)./(1:K) > 0, 1, 'last');
theta      = (cssv(rho) - 1) / rho;
alpha      = max(z - theta, 0);
alpha      = alpha / sum(alpha + eps);   % normalise & keep row shape
end

function alpha = UpdateViewWeights_Entmax15(e)
q   = 1.5;
z   = -e;                        % scores
m   = length(z);
% Algorithm: *Damped* bisection to find τ s.t. Σ(max( (z-τ)/(2-q), 0)^(1/(q-1)))=1
lo = min(z) - 1;  hi = max(z);
for it = 1:30
    tau = (lo+hi)/2;
    phi = max((z - tau), 0).^(1/(q-1));
    s   = sum(phi);
    if s > 1, lo = tau; else, hi = tau; end
end
alpha = phi / max(s, eps);
end

% function alpha = UpdateViewWeights_Tempered(e, T)
% % ---------------------------------------------------------------
% %  Tempered soft-max:  α_i  ∝  (e_i + eps)^(-1/T)
% %  Requires T to be a scalar.  e can be row or column.
% % ---------------------------------------------------------------
% if numel(T) ~= 1
%     error('UpdateViewWeights_Tempered:TemperatureNotScalar', ...
%           'T must be a scalar, but received a %s.', mat2str(size(T)));
% end
% 
% e = double(e(:)');            % force row and type
% z = (e + eps).^(-1 / T);      % element-wise power, no ./ needed
% alpha = z / sum(z);           % row / scalar  ⇒ row (1×V)
% end

function alpha = UpdateViewWeights_Tempered(e, T)
% T > 0 : "temperature" on the *log* scale
z     = exp(-log(e + eps) / T);     % smaller loss ⇒ larger z
% z     = (e + eps).^(-1./T);      % = exp(-log e / T)
alpha = z / sum(z);
end

function alpha = UpdateViewWeights_InverseVar(residualCell)
% residualCell{v}  is an n×1 vector of per-sample errs for view v
V = numel(residualCell);
invVar = zeros(1,V);
for v = 1:V
    invVar(v) = 1 / var(residualCell{v});
end
alpha = invVar / sum(invVar);
end

function [alpha, m] = UpdateViewWeights_EntropyAdam(e, tau, m, beta)
if ~isscalar(tau)
    error('tau must be a scalar');
end
z       = exp(-(e - min(e))/tau);
alpha_t = z / sum(z);
m       = beta*m + (1-beta)*alpha_t;    % EMA
alpha   = m / sum(m);
end

function [alpha, pulls] = UpdateViewWeights_UCB(e, pulls, step, c)
% pulls(v) = how many times view v has been chosen up to now
% step     = current outer iteration
% c        = exploration constant (≈0.5–2)
score = -e + c*sqrt(log(step) ./ (pulls + eps));
score = score - max(score);       % stabilise
z     = exp(score);
alpha = z / sum(z);
pulls = pulls + alpha;            % fractional pulls
end

function alpha = UpdateViewWeights_Lcurve(e, gamma)
% closed-form for quadratic regularised simplex projection
V   = numel(e);
b   = e / (2*gamma);
lambda = (sum(b) - 1) / V;
alpha  = max(b - lambda, 0);
alpha  = alpha / sum(alpha);
end