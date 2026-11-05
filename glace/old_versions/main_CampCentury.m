%% 1D EULERIAN MODEL — ARGON TRANSPORT IN ICE (advection + lattice diffusion)
%%
%% State variables:
%%   x       : Ar mixing ratio in clathrate gas    [mol/mol]
%%   C_ice   : Ar dissolved in ice lattice          [mol/m³ ice]
%%   C_bulk  : total Ar = clathrate + lattice       [mol/m³ ice]
%%
%% Thermodynamic equilibrium (instantaneous, at each timestep):
%%   C_ice  = x * P_diss * S  =  x * pS
%%   C_bulk = C_ice + x * n_air  =  x * (pS + n_air)
%%   =>  x  = C_bulk / (pS + n_air)
%%
%% Grid convention:
%%   z = 0   -> bed / fond du glacier   (MATLAB index 1, deepest)
%%   z = Z   -> surface of the ice      (MATLAB index end, shallowest)
%%   depth   -> depth from ice surface  (standard core convention, = depth_max - z)
%%
%% Reconstruction de C_bulk après diffusion (méthode Code 1) :
%%   Seul l'incrément de la phase lattice est ajouté au réservoir total.
%%   C_bulk(:,k+1) = C_bulk_adv + (Cice_new - Cice_adv)
%%   Le ré-équilibre clathrate/lattice n'intervient qu'en étape E
%%   via X = C_bulk / (pS + n_air).
%%
%% =========================================================
clear; clc; close all;
%% =========================================================
%% *** TOUS LES PARAMÈTRES À MODIFIER SONT ICI ***
%% =========================================================
%  Ne modifier que ce bloc. Le reste du code s'adapte automatiquement.
% --- Constantes physiques ---
rho_ice    = 917;         % densité de la glace [kg/m³]
Vm_STP     = 22.414e-3;  % volume molaire gaz idéal STP [m³/mol]
P_diss_val = 5e6;         % pression dissociation clathrate Ar [Pa]
S_Ar_val   = 10e-9;      % solubilité Ar dans glace [mol/m³/Pa]
TAC_val    = 0.091e-3;   % teneur totale en air [m³ STP/kg glace] (Sailer)
% --- Paramètres de transport (valeurs de référence littérature) ---
D_m2s       = 1e-11;   % diffusivité lattice Ar [m²/s]
flux_bottom = 1.8e-9;  % flux basal Ar [mol/m²/yr]
% --- Profil de vitesse verticale — Lliboutry (1979) pour lit gelé ---
% Profil analytique de vitesse verticale pour une calotte à lit gelé :
%   w(ζ) = −ḃ · [ (p+2)·ζ  −  (p+2)·ζ^(p+1)  +  ζ^(p+2) ]   avec ζ = z/H_ice
% Normalisé : vel_profile(Z_ref, Z_ref, Um) = Um  →  compatible Um=0 (leave-one-out)
%   b_acc  = taux d'accumulation annuel [m/yr eq. glace]   — Dansgaard et al. 1969
%   H_ice  = épaisseur totale de la calotte à Camp Century [m]
%   p_llib = paramètre de forme Lliboutry (p=3 typique pour calotte polaire)
b_acc   = 0.35;    % [m/yr ice equiv]
H_ice   = 1387;    % [m]   — Camp Century (Dansgaard et al. 1969)
p_llib  = 3;       % [-]   — paramètre de forme Lliboutry (1979)
% Umax = vitesse à z = Z (sommet du domaine) — calculé en §4 après que Z est connu
Umax = NaN;   % [m/yr] — placeholder, remplacé en §4
% Profil Lliboutry normalisé : f(ζ) = (p+2)·ζ − (p+2)·ζ^(p+1) + ζ^(p+2)
vel_profile = @(z_col, Z_ref, Um) Um .* ...
    ( (p_llib+2).*(z_col./H_ice) - (p_llib+2).*(z_col./H_ice).^(p_llib+1) + (z_col./H_ice).^(p_llib+2) ) ./ ...
    ( (p_llib+2).*(Z_ref./H_ice) - (p_llib+2).*(Z_ref./H_ice).^(p_llib+1) + (Z_ref./H_ice).^(p_llib+2) );
% gas_loss : taux de perte gazeuse λ [yr⁻¹], appliqué à la PHASE CLATHRATE uniquement
%   Physique : des clathrates se vident entièrement → X et C_ice inchangés,
%              seul C_bulk diminue via C_bulk = C_ice + (X*n_air)*exp(-λ*dt)
% Relation avec le gradient observé dG/dz [ml/kg/m] :
%   gas_loss = |dG/dz| * |Umax| / G_ref
% Exemple : dG/dz = 0.9 ml/kg/m, |Umax|=1e-4 m/yr, G_ref=50 ml/kg
%   => gas_loss ≈ 0.9 * 1e-4 / 50 = 1.8e-6 yr⁻¹
gas_loss  = 1.8e-6; %  = 0;        % taux de perte gazeuse  [yr⁻¹]  (0 = inactif)
% --- Condition aux limites au fond (z = 0) ---
% 'flux'      : flux Neumann imposé  = flux_bottom [mol/m²/yr]  (physique par défaut)
% 'zero_flux' : flux nul (Neumann homogène)  — test : profil doit rester constant si CI uniforme
% 'dirichlet' : valeur fixée C_ice = Cice_bottom_val            — test : BC symétrique
BC_bottom_type  = 'flux';
Cice_bottom_val = 0;    % [mol/m³] — utilisé uniquement si BC_bottom_type = 'dirichlet'
% --- Numérique ---
% NOTE : le schéma upwind 1er ordre introduit une diffusion numérique ∝ Umax*dz/2.
%        Un schéma TVD (ex. van Leer) serait plus précis aux grandes échelles de temps.
%        Pour l'instant l'upwind est conservé (Sandra, sept. 2026).
nz = 101;     % nombre de nœuds sur la verticale
T  = 1.25e5;  % durée totale de simulation [yr]  — âge glace Camp Century (~125 000 yr)
% ATTENTION CFL : avec Umax_DJ ≈ −0.05 m/yr et dz ≈ 0.3 m,
%   dt_max = dz/|Umax| ≈ 6 yr  →  dt = 3 yr donne CFL ≈ 0.5 (stable)
dt = 3;       % pas de temps [yr]
% --- Bornes d'optimisation (utilisées uniquement en Figure 8) ---
%   ordre : [Umax   D_m2s   gas_loss   flux_bottom]
LB_opt = [-2e-4,  2e-12, 0,      1e-10];
UB_opt = [-2e-5,  5e-11, 1e-4,  2e-9 ];
N_STARTS_opt = 10;   % nombre de points de départ aléatoires
%% =========================================================
%% — FIN DU BLOC PARAMÈTRES — NE PAS MODIFIER EN DESSOUS —
%% =========================================================

%% =========================================================
%% 1. Chargement des données Camp Century
%% =========================================================
data_file  = fullfile(fileparts(mfilename('fullpath')), ...
    'Table_S1_122025_for manuscript.xlsx');
sheet_name = 'CampCentury';
raw = readmatrix(data_file, 'Sheet', sheet_name, 'Range', 'C5:AD59');
depth     = raw(:,1);    X_lab     = raw(:,10);   X_u_lab   = raw(:,15);
G_lab     = raw(:,2);    G_u_lab   = raw(:,3);    debris    = raw(:,4);
bulk_CC   = raw(:,21);   bulk_CC_u = raw(:,26);
valid   = ~isnan(depth) & ~isnan(X_lab) & ~isnan(bulk_CC);
depth   = depth(valid);   G_lab   = G_lab(valid);   G_u_lab   = G_u_lab(valid);
debris  = debris(valid);  X_lab   = X_lab(valid);   X_u_lab   = X_u_lab(valid);
bulk_CC = bulk_CC(valid); bulk_CC_u = bulk_CC_u(valid);
[depth, idx] = sort(depth);
G_lab = G_lab(idx); G_u_lab = G_u_lab(idx); debris = debris(idx);
X_lab = X_lab(idx); X_u_lab = X_u_lab(idx);
bulk_CC = bulk_CC(idx); bulk_CC_u = bulk_CC_u(idx);
depth_min = min(depth);  depth_max = max(depth);
Z = depth_max - depth_min;
fprintf('\nDataset: %s\n', data_file);
fprintf('Depth range: %.2f – %.2f m  (column Z = %.2f m)\n', depth_min, depth_max, Z);

%% =========================================================
%% 2. Grille verticale
%% =========================================================
z  = linspace(0, Z, nz)';
dz = z(2) - z(1);
depth_model = depth_max - z;

%% =========================================================
%% 3. Temps
%% =========================================================
nt = round(T/dt);
t  = (0:nt)*dt;

%% =========================================================
%% 4. Grandeurs dérivées (calculées automatiquement depuis PARAMÈTRES)
%% =========================================================
D = D_m2s * 3.1536e7;           % [m²/yr]
% --- Calcul de Umax Lliboutry à z = Z (sommet du domaine de 30 m) ---
% w(Z) = −b_acc · [ (p+2)·ζ_Z − (p+2)·ζ_Z^(p+1) + ζ_Z^(p+2) ]   avec ζ_Z = Z/H_ice
zeta_Z = Z / H_ice;
Umax   = -b_acc * ( (p_llib+2)*zeta_Z - (p_llib+2)*zeta_Z^(p_llib+1) + zeta_Z^(p_llib+2) );
fprintf('Umax Lliboutry (z = %.1f m) = %.4e m/yr\n', Z, Umax);
u = vel_profile(z, Z, Umax);    % profil défini dans PARAMÈTRES — ne pas modifier ici
CFL_adv = max(abs(u)) * dt / dz;
SC_diff  = D * dt / dz^2;
fprintf('CFL (advection) = %.3g   |   Diffusion number = %.3g\n', CFL_adv, SC_diff);
if CFL_adv > 1, warning('Advection CFL > 1 — reduce dt or increase dz.'); end
P_diss = P_diss_val * ones(nz,1);
S_Ar   = S_Ar_val   * ones(nz,1);
pS     = P_diss .* S_Ar;

%% =========================================================
%% 5. Teneur en air
%% =========================================================
n_air = TAC_val * rho_ice / Vm_STP;   % [mol air/m³ ice] — scalaire

%% =========================================================
%% 6. Conversion données labo → unités modèle
%% =========================================================
Cbulk_lab         = bulk_CC * rho_ice;
Cbulk_lab_profile = interp1(depth, Cbulk_lab, depth_model, 'linear','extrap');

%% =========================================================
%% 7. Condition aux limites en surface (z = Z)
%% =========================================================
X_atm         = X_lab(1) / 100;
Cice_surface  = X_atm * pS(end);
Cbulk_surface = X_atm * (pS(end) + n_air);

%% =========================================================
%% 8. Condition initiale (t = 0)
%% =========================================================
% CI cohérente avec BC surface : les deux utilisent X_lab(1) (échantillon le plus haut)
% => à t=0, colonne uniforme au niveau atmosphérique (même valeur que la BC Dirichlet)
Cbulk0 = Cbulk_surface * ones(nz,1);
x0     = Cbulk0 ./ (pS + n_air);
Cice0  = x0 .* pS;
fprintf('\nIC = BC surface = %.4e mol/m³  (X_atm = %.2f ppm)\n', Cbulk_surface, X_atm*1e6);
fprintf('IC partition check: max |C_bulk - (C_ice + x*n_air)| = %.3e mol/m³\n', ...
    max(abs(Cbulk0 - (Cice0 + x0.*n_air))));

%% =========================================================
%% 9. Flux basal (z = 0)
%% =========================================================
% flux_bottom défini dans la section PARAMÈTRES

%% =========================================================
%% 10. Pré-allocation
%% =========================================================
C_bulk = zeros(nz, nt+1);  C_ice = zeros(nz, nt+1);
X      = zeros(nz, nt+1);
ADV    = zeros(nz, nt);    DIFF  = zeros(nz, nt);
C_bulk(:,1) = Cbulk0;  C_ice(:,1) = Cice0;  X(:,1) = x0;

%% =========================================================
%% 11. Matrice de diffusion implicite (Backward Euler)
%% =========================================================
% Physique (Sandra) : seul C_ice diffuse (grain boundary diffusion).
% C_bulk est le "grand réservoir" qui ne diffuse pas directement.
% Après chaque pas de diffusion, l'équilibre thermodynamique C_ice = X*pS
% est restauré via X = C_bulk / (pS + n_air)  (étape E).
r      = D * dt / dz^2;
A_diff = sparse(nz, nz);
for i = 2:nz-1
    A_diff(i,i-1) = -r; A_diff(i,i) = 1+2*r; A_diff(i,i+1) = -r;
end
% BC au fond (z = 0) — choix via BC_bottom_type
switch BC_bottom_type
    case {'flux','zero_flux'}
        % Neumann (point fantôme) : ∂C_ice/∂z|_0 = -flux_bottom/D
        A_diff(1,1) = 1+2*r; A_diff(1,2) = -2*r;
    case 'dirichlet'
        % Dirichlet : C_ice(z=0) = Cice_bottom_val
        A_diff(1,1) = 1;
    otherwise
        error('BC_bottom_type inconnu : %s', BC_bottom_type);
end
% BC en surface (z = Z) — Dirichlet (valeur atmosphérique imposée)
A_diff(end,end) = 1;
dA_diff = decomposition(A_diff);

%% =========================================================
%% 12. Boucle temporelle
%% =========================================================
u_face    = 0.5*(u(1:end-1) + u(2:end));
u_face(1) = 0;
% gas_loss est directement λ [yr⁻¹] — decay par pas de temps dt [yr]
% Équation : dC/dt = -λ*C  =>  C(t+dt) = C(t)*exp(-λ*dt)
decay  = exp(-gas_loss * dt);

for k = 1:nt

    %% -- A. ADVECT C_bulk (forme conservative, upwind) ---------------- %%
    C_old    = C_bulk(:,k);
    adv_flux = zeros(nz-1,1);
    for j = 2:nz-1
        if u_face(j) >= 0, C_up = C_old(j); else, C_up = C_old(j+1); end
        adv_flux(j) = u_face(j) * C_up;
    end
    adv_flux(nz-1) = u_face(nz-1) * Cbulk_surface;  % fix Sandra : BC surface imposée au flux upwind
    C_bulk_adv = C_old;
    for i = 2:nz-1
        C_bulk_adv(i) = C_old(i) + dt*(adv_flux(i-1) - adv_flux(i))/dz;
    end
    C_bulk_adv(end) = Cbulk_surface;
    ADV(:,k) = (C_bulk_adv - C_old)/dt;

    %% -- B. PARTITION ------------------------------------------------- %%
    x_adv    = C_bulk_adv ./ (pS + n_air);
    Cice_adv = x_adv .* pS;

    %% -- C. DIFFUSE C_ice (Backward Euler) ---------------------------- %%
    B = Cice_adv;
    switch BC_bottom_type
        case 'flux'
            B(1) = Cice_adv(1) + 2*flux_bottom*dt/dz;  % flux Neumann imposé
        case 'zero_flux'
            B(1) = Cice_adv(1);                          % flux nul (homogène)
        case 'dirichlet'
            B(1) = Cice_bottom_val;                       % valeur fixée
    end
    B(end) = Cice_surface;
    Cice_new = dA_diff \ B;

    %% -- D. REBUILD C_bulk -------------------------------------------- %%
    % Seul l'incrément de la phase lattice est ajouté au réservoir total.
    % Les clathrates ne se re-remplissent PAS pendant l'étape de diffusion :
    % c'est le re-équilibre thermodynamique (étape E) qui redistribue ensuite.
    C_bulk(:,k+1) = C_bulk_adv + (Cice_new - Cice_adv);
    DIFF(:,k)     = (C_bulk(:,k+1) - C_bulk_adv) / dt;

    %% -- E. RE-EQUILIBRATE -------------------------------------------- %%
    X(:,k+1)     = C_bulk(:,k+1) ./ (pS + n_air);
    C_ice(:,k+1) = X(:,k+1) .* pS;

    %% -- F. GAS LOSS (phase clathrate uniquement) --------------------- %%
    if gas_loss > 0
        % Des clathrates se vident : X et C_ice restent inchangés,
        % seul le compartiment gazeux piégé diminue.
        C_clath       = C_bulk(:,k+1) - C_ice(:,k+1);
        C_bulk(:,k+1) = C_ice(:,k+1) + C_clath * decay;
        % X(:,k+1) et C_ice(:,k+1) ne sont PAS modifiés
    end

end
fprintf('Simulation complete.\n');
verify_BC_conditions(z, dz, Z, pS, n_air*ones(nz,1), X_atm, dt, vel_profile);

%% =========================================================
%% 13. Retour aux unités labo
%% =========================================================
bulk_CC_model = C_bulk / rho_ice;

%% =========================================================
%% 14. Diagnostics
%% =========================================================
partition_err = C_bulk - (C_ice + X.*n_air);
fprintf('Partition consistency: max error = %.3e mol/m³\n', max(abs(partition_err(:))));
M_bulk = trapz(z, C_bulk);
fprintf('Final total Ar inventory: %.6e mol/m²\n', M_bulk(end));

%% =========================================================
%% FIGURES 1–6  (diagnostics existants)
%% =========================================================
X_lab_grid = interp1(depth, X_lab/100, depth_model, 'linear','extrap');
figure(1); clf;
subplot(1,3,1);
    plot(X(:,end)*1e6, z,'LineWidth',2); hold on;
    plot(X_lab_grid*1e6, z,'o','MarkerSize',4);
    xlabel('x (ppm)'); ylabel('Height above bed (m)');
    title('Ar mixing ratio'); legend('Model','Lab','Location','best'); grid on; box on;
subplot(1,3,2);
    plot(C_ice(:,end), z,'LineWidth',2);
    xlabel('C_{ice} (mol m^{-3})'); ylabel('Height above bed (m)');
    title('Lattice Ar'); grid on; box on;
subplot(1,3,3);
    plot(C_bulk(:,end), z,'LineWidth',2); hold on;
    plot(Cbulk_lab_profile, z,'o','MarkerSize',4);
    xlabel('C_{bulk} (mol m^{-3})'); ylabel('Height above bed (m)');
    title('Total Ar'); legend('Model','Lab','Location','best'); grid on; box on;
sgtitle('Final state — thermodynamic partition');

n_snap = 6; idx_k = round(linspace(1,nt+1,n_snap));
labels = arrayfun(@(ii) sprintf('t = %.2g yr',t(idx_k(ii))),1:n_snap,'UniformOutput',false);
figure(2); clf;
subplot(1,3,1); hold on;
    for ii=1:n_snap; plot(X(:,idx_k(ii))*1e6, z,'LineWidth',1.8); end
    xlabel('x (ppm)'); ylabel('Height above bed (m)'); title('x — mixing ratio'); grid on; box on;
subplot(1,3,2); hold on;
    for ii=1:n_snap; plot(C_ice(:,idx_k(ii)), z,'LineWidth',1.8); end
    xlabel('C_{ice} (mol m^{-3})'); ylabel('Height above bed (m)'); title('C_{ice} — lattice'); grid on; box on;
subplot(1,3,3); hold on;
    for ii=1:n_snap; plot(C_bulk(:,idx_k(ii)), z,'LineWidth',1.8); end
    xlabel('C_{bulk} (mol m^{-3})'); ylabel('Height above bed (m)'); title('C_{bulk} — total');
    legend(labels,'Location','best','FontSize',8); grid on; box on;
sgtitle('Temporal evolution');

skip_t = max(1,floor(nt/200)); idx_plot = 1:skip_t:nt;
t_plot = t(idx_plot); ADV_plot = ADV(:,idx_plot); DIFF_plot = DIFF(:,idx_plot);
figure(3); clf;
subplot(1,2,1); pcolor(t_plot,z,ADV_plot); shading flat; colorbar;
    xlabel('Time (yr)'); ylabel('Height above bed (m)');
    title('Advective tendency [mol m^{-3} yr^{-1}]'); grid on; box on;
subplot(1,2,2); pcolor(t_plot,z,DIFF_plot); shading flat; colorbar;
    xlabel('Time (yr)'); ylabel('Height above bed (m)');
    title('Diffusive tendency [mol m^{-3} yr^{-1}]'); grid on; box on;
sgtitle('Transport tendencies');

figure(4); clf;
subplot(1,3,1);
    plot(G_lab,depth,'o-','LineWidth',1.5); set(gca,'YDir','reverse');
    xlabel('Total gas content (mL/kg)'); ylabel('Depth from surface (m)');
    title('Total gas content'); grid on; box on;
subplot(1,3,2);
    plot([TAC_val TAC_val],[depth_min depth_max],'r-','LineWidth',2);
    set(gca,'YDir','reverse'); xlim([0 2*TAC_val]);
    xlabel('TAC (m³ STP kg^{-1})'); ylabel('Depth from surface (m)');
    title('TAC (constant — Sailer)'); grid on; box on;
subplot(1,3,3);
    plot(X_lab,depth,'o-','LineWidth',1.5); set(gca,'YDir','reverse');
    xlabel('Ar mixing ratio (%)'); ylabel('Depth from surface (m)');
    title('Ar mixing ratio'); grid on; box on;
sgtitle('Camp Century — lab measurements');

figure(5); clf;
subplot(1,3,1);
    plot(X_lab/100,depth,'ko','MarkerSize',5); hold on;
    plot(X(:,end),depth_model,'LineWidth',2); set(gca,'YDir','reverse');
    xlabel('x (mol/mol)'); ylabel('Depth from surface (m)');
    title('Ar mixing ratio'); legend('Lab','Model','Location','best'); grid on; box on;
subplot(1,3,2);
    plot(bulk_CC,depth,'ko','MarkerSize',5); hold on;
    plot(bulk_CC_model(:,end),depth_model,'LineWidth',2); set(gca,'YDir','reverse');
    xlabel('C_{bulk} (mol kg^{-1})'); ylabel('Depth from surface (m)');
    title('Total Ar'); legend('Lab','Model','Location','best'); grid on; box on;
subplot(1,3,3);
    plot(C_ice(:,end),depth_model,'LineWidth',2); set(gca,'YDir','reverse');
    xlabel('C_{ice} (mol m^{-3})'); ylabel('Depth from surface (m)');
    title('Lattice Ar (model only)'); grid on; box on;
sgtitle('Camp Century — model vs lab');

frac_lattice = C_ice(:,end) ./ C_bulk(:,end);
frac_clath   = 1 - frac_lattice;
figure(6); clf;
    plot(frac_lattice*100, z,'LineWidth',2); hold on;
    plot(frac_clath*100,   z,'LineWidth',2);
    xlabel('Fraction of total Ar (%)'); ylabel('Height above bed (m)');
    title('Ar partition — lattice vs clathrates (final state)');
    legend('Lattice','Clathrates','Location','best'); grid on; box on;
fprintf('\nFinal Ar partition:\n');
fprintf('  Lattice:    %.4f – %.4f %%\n', 100*min(frac_lattice),100*max(frac_lattice));
fprintf('  Clathrates: %.4f – %.4f %%\n', 100*min(frac_clath),  100*max(frac_clath));

%% =========================================================
%% FIGURE 9 — Construction progressive des processus (état final t = T)
%%   Courbe 1 : Diffusion + Source basale              (Umax=0, gas_loss=0)
%%   Courbe 2 : + Advection                            (Umax≠0, gas_loss=0)
%%   Courbe 3 : + Gas loss = modèle complet            (Umax≠0, gas_loss≠0)
%% NOTE : X et C_ice ne changent PAS avec gas_loss (clathrate phase uniquement)
%%        → courbes 2 et 3 identiques pour X et C_ice ; C_bulk diverge
%% =========================================================
fprintf('\n=== FIGURE 9 : Construction progressive des processus ===\n');
n_air_9 = n_air * ones(nz,1);
sim9a = run_model_attr(0,     D_m2s, 0,        flux_bottom, ...
            z, dz, Z, pS, n_air_9, X_atm, Cbulk0, T, dt, vel_profile);   % Diff + Source
sim9b = run_model_attr(Umax,  D_m2s, 0,        flux_bottom, ...
            z, dz, Z, pS, n_air_9, X_atm, Cbulk0, T, dt, vel_profile);   % + Advection
sim9c = run_model_attr(Umax,  D_m2s, gas_loss, flux_bottom, ...
            z, dz, Z, pS, n_air_9, X_atm, Cbulk0, T, dt, vel_profile);   % + Gas loss

X9    = {sim9a.X,          sim9b.X,          sim9c.X};
Cice9 = {sim9a.X .* pS,   sim9b.X .* pS,   sim9c.X .* pS};
Cb9   = {sim9a.C*rho_ice,  sim9b.C*rho_ice,  sim9c.C*rho_ice};

col9 = [0.000 0.447 0.698;   % bleu      — Diff + Source
        0.902 0.624 0.000;   % orange    — + Advection
        0.835 0.369 0.000];  % vermillon — + Gas loss
lbl9 = {'Diff + Source', '+ Advection', '+ Gas loss (complet)'};
lw9  = 2.2;
figure(9); clf; set(gcf,'Position',[100 100 1200 500]);
subplot(1,3,1); hold on;
    plot(X_atm*1e6*ones(2,1), [0 Z],'k:','LineWidth',1,'DisplayName','CI (t=0)');
    for ii = 1:3
        plot(X9{ii}*1e6, z, 'LineWidth',lw9, 'Color',col9(ii,:), 'DisplayName',lbl9{ii});
    end
    xlabel('x (ppm)'); ylabel('Height above bed (m)');
    title('x — mixing ratio');
    legend('Location','best','FontSize',8); grid on; box on;
subplot(1,3,2); hold on;
    plot(X_atm*pS(1)*1e3*ones(2,1), [0 Z],'k:','LineWidth',1,'HandleVisibility','off');
    for ii = 1:3
        plot(Cice9{ii}*1e3, z, 'LineWidth',lw9, 'Color',col9(ii,:), 'DisplayName',lbl9{ii});
    end
    xlabel('C_{ice} (mol m^{- 3}) \times 10^{-3}'); ylabel('Height above bed (m)');
    title('C_{ice} — lattice (inchangée par gas loss)');
    grid on; box on;
subplot(1,3,3); hold on;
    plot(Cbulk_surface*ones(2,1), [0 Z],'k:','LineWidth',1,'HandleVisibility','off');
    for ii = 1:3
        plot(Cb9{ii}, z, 'LineWidth',lw9, 'Color',col9(ii,:), 'DisplayName',lbl9{ii});
    end
    xlabel('C_{bulk} (mol m^{-3})'); ylabel('Height above bed (m)');
    title('C_{bulk} — total (affectée par gas loss)');
    grid on; box on;
sgtitle(sprintf('Construction progressive des processus  (t = %.2g yr)', T), 'FontSize',13);
fprintf('Figure 9 : OK\n');

%% =========================================================
%% FIGURE 7 — Importance des processus (leave-one-out)
%% Valeurs de référence imposées — AUCUNE optimisation
%% =========================================================
fprintf('\n=== FIGURE 7 : Attribution (valeurs littérature) ===\n');
p_ref = [Umax, D_m2s, gas_loss, flux_bottom];
NULL_VAL    = [0,     1e-15, 0,  0];
PROC_NAMES  = {'U_{max}','D','gas_{loss}','flux_{bot}'};
n_air_vec = n_air * ones(nz,1);
sim_all  = run_model_attr(p_ref(1),p_ref(2),p_ref(3),p_ref(4), ...
               z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt,vel_profile);
RMSE_all = calc_rmse_attr(sim_all.X, z, depth, X_lab);
fprintf('RMSE (tous processus) : %.4e\n', RMSE_all);
RMSE_loo = nan(4,1);
for ip = 1:4
    p_tmp     = p_ref;  p_tmp(ip) = NULL_VAL(ip);
    sim_tmp   = run_model_attr(p_tmp(1),p_tmp(2),p_tmp(3),p_tmp(4), ...
                    z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt,vel_profile);
    RMSE_loo(ip) = calc_rmse_attr(sim_tmp.X, z, depth, X_lab);
    fprintf('Sans %-12s : RMSE = %.4e  (Δ = +%.4e)\n', ...
        PROC_NAMES{ip}, RMSE_loo(ip), RMSE_loo(ip)-RMSE_all);
end
delta = RMSE_loo - RMSE_all;
[delta_sort, isort] = sort(delta,'descend');
names_sort = PROC_NAMES(isort);
CLR4 = [0.00 0.45 0.70; 0.90 0.62 0.00; 0.00 0.62 0.45; 0.80 0.40 0.00];
clr_sort = CLR4(isort,:);
figure(7); clf;
ax7 = axes; hold(ax7,'on');
for k = 1:4
    bar(ax7,k,delta_sort(k),0.6,'FaceColor',clr_sort(k,:),'EdgeColor','none');
    text(ax7,k,delta_sort(k)+max(delta_sort)*0.04, ...
        sprintf('+%.2e',delta_sort(k)), ...
        'HorizontalAlignment','center','FontSize',10,'FontWeight','bold', ...
        'Color',clr_sort(k,:)*0.75);
end
set(ax7,'XTick',1:4,'XTickLabel',names_sort,'FontSize',12,'Box','off');
ylabel(ax7,'\DeltaRMSE_X  (mol mol^{-1})','FontSize',11);
title(ax7,{'Importance relative des processus de transport', ...
    '(\DeltaRMSE = augmentation quand le processus est éteint)'}, ...
    'FontSize',11,'FontWeight','bold');
y_top = max(delta_sort) * 1.20;
if y_top <= 0, y_top = 1e-6; end
ylim(ax7,[0, y_top]);
grid(ax7,'on'); ax7.GridAlpha = 0.2;
saveas(figure(7),'figure7_importance.png');
fprintf('Figure 7 sauvegardée : figure7_importance.png\n');

% %% =========================================================
% %% FIGURE 8 — Profils X : best fit optimisé (leave-one-out)
% %% =========================================================
% fprintf('\n=== FIGURE 8 : Profils X avec optimisation ===\n');
% opts_opt = optimset('MaxIter',5000,'TolFun',1e-10,'Display','off');
% idx_all  = [1 2 3 4];
% objfun_all = @(t_vec) obj_attr(t_vec, idx_all, p_ref, LB_opt, UB_opt, ...
%     z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt,depth,X_lab,vel_profile);
% best_r = inf; best_t = [];
% for ns = 1:N_STARTS_opt
%     t0 = rand_start_attr(idx_all, LB_opt, UB_opt);
%     try
%         [t_s,r_s] = fminsearch(objfun_all,t0,opts_opt);
%         if r_s < best_r; best_r = r_s; best_t = t_s; end
%     catch; end
% end
% p_opt = p_ref;
% for k = 1:4; p_opt(k) = t2p_attr(best_t(k),k,LB_opt,UB_opt); end
% RMSE_opt = best_r;
% fprintf('Best fit : RMSE=%.3e  Umax=%.2e  D=%.2e  gas_loss=%.3f  flux_bot=%.2e\n', ...
%     RMSE_opt, p_opt(1),p_opt(2),p_opt(3),p_opt(4));
% sim_opt  = run_model_attr(p_opt(1),p_opt(2),p_opt(3),p_opt(4), ...
%                z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt,vel_profile);
% X_opt    = sim_opt.X * 100;
% X_loo_opt = cell(4,1);
% RMSE_loo_opt = nan(4,1);
% for ip = 1:4
%     p_tmp     = p_opt; p_tmp(ip) = NULL_VAL(ip);
%     sim_tmp   = run_model_attr(p_tmp(1),p_tmp(2),p_tmp(3),p_tmp(4), ...
%                     z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt,vel_profile);
%     X_loo_opt{ip}    = sim_tmp.X * 100;
%     RMSE_loo_opt(ip) = calc_rmse_attr(sim_tmp.X,z,depth,X_lab);
% end
% [~,imax_lab] = max(X_lab);
% hab_sed      = depth_max - depth(imax_lab);
% BROWN        = [0.55 0.27 0.07];
% LW = 2; MS = 6;
% all_X_fig8 = [X_lab; X_opt; cell2mat(cellfun(@(v)v,X_loo_opt,'UniformOutput',false))];
% xlim8 = [min(all_X_fig8)*0.97, max(all_X_fig8)*1.03];
% figure(8); clf;
% ax8 = axes; hold(ax8,'on');
% if hab_sed > 0
%     patch(ax8,xlim8([1 2 2 1]),[0 0 hab_sed hab_sed],BROWN, ...
%         'FaceAlpha',0.25,'EdgeColor','none');
% end
% plot(ax8,X_lab,depth_max-depth,'ko','MarkerSize',MS,'MarkerFaceColor','w', ...
%     'LineWidth',1.2,'DisplayName','Données labo');
% for ip = 1:4
%     plot(ax8,X_loo_opt{ip},z,'--','Color',CLR4(ip,:),'LineWidth',1.4, ...
%         'DisplayName',sprintf('sans %s',PROC_NAMES{ip}));
% end
% plot(ax8,X_opt,z,'-','Color',[0 0 0],'LineWidth',2.5, ...
%     'DisplayName',sprintf('Best fit (RMSE=%.2e)',RMSE_opt));
% xlabel(ax8,'Ar mixing ratio (%)','FontSize',11);
% ylabel(ax8,'Hauteur au-dessus du fond (m)','FontSize',11);
% title(ax8,{'Profils X — attribution par processus', ...
%     'Pointillés = best fit sans ce paramètre'},'FontSize',11,'FontWeight','bold');
% xlim(ax8,xlim8); ylim(ax8,[0 Z*1.05]);
% legend(ax8,'Location','northwest','FontSize',8,'Box','off');
% box(ax8,'off'); grid(ax8,'on'); ax8.GridAlpha = 0.2;
% saveas(figure(8),'figure8_profils.png');
% fprintf('Figure 8 sauvegardée : figure8_profils.png\n');

%% =========================================================
%% FONCTIONS LOCALES
%% =========================================================

function sim = run_model_attr(Umax, D_m2s, gas_loss, flux_bottom, ...
        z, dz, Z, pS, n_air, X_atm, Cbulk0, T, dt, vel_profile)
% Version vectorisée pour l'attribution (Figures 7, 8, 9)
% sim.X = mixing ratio après re-équilibre (inchangé par gas_loss)
% sim.C = C_bulk final / rho_ice (affecté par gas_loss)
    rho_ice = 917;
    D  = D_m2s * 3.1536e7;
    nz = numel(z);
    nt = round(T/dt);
    u      = vel_profile(z, Z, Umax);
    u_face = 0.5*(u(1:end-1)+u(2:end)); u_face(1) = 0;
    u_pos  = max(0,u_face);  u_neg = min(0,u_face);
    Cice_surf = X_atm * pS(end);
    Cb_surf   = X_atm * (pS(end) + n_air(end));
    C_bulk    = Cbulk0;
    r = D*dt/dz^2; e = ones(nz,1);
    A = spdiags([-r*e,(1+2*r)*e,-r*e],[-1,0,1],nz,nz);
    A(1,1) = 1+2*r; A(1,2) = -2*r;
    A(end,:) = 0;   A(end,end) = 1;
    dA = decomposition(A);
    decay_a = exp(-gas_loss * dt);
    for k = 1:nt
        F  = u_pos.*C_bulk(1:end-1) + u_neg.*C_bulk(2:end);
        F(end) = u_face(end) * Cb_surf;
        Ca = C_bulk;
        Ca(2:end-1) = C_bulk(2:end-1) + dt/dz*(F(1:end-1)-F(2:end));
        Ca(end) = Cb_surf;
        xa  = Ca./(pS+n_air);  Cia = xa.*pS;
        B = Cia; B(1) = Cia(1)+2*flux_bottom*dt/dz; B(end) = Cice_surf;
        Cice_new_a = dA\B;
        C_bulk = Ca + (Cice_new_a - Cia);
        x_new_a = C_bulk ./ (pS + n_air);
        if gas_loss > 0
            C_ice_a   = x_new_a .* pS;
            C_clath_a = C_bulk - C_ice_a;
            C_bulk    = C_ice_a + C_clath_a * decay_a;
        end
    end
    sim.X = x_new_a;
    sim.C = C_bulk / rho_ice;
end

function rmse = calc_rmse_attr(X_model, z, depth, X_lab)
    depth_max = max(depth);
    X_interp  = interp1(depth_max-z, X_model, depth,'linear','extrap');
    rmse      = sqrt(mean((X_interp - X_lab/100).^2));
end

function rmse = obj_attr(t_vec, idx_a, p_off, LB, UB, ...
        z,dz,Z,pS,n_air,X_atm,Cbulk0,T,dt,depth,X_lab,vel_profile)
    p = p_off;
    for k = 1:numel(idx_a)
        p(idx_a(k)) = t2p_attr(t_vec(k),idx_a(k),LB,UB);
    end
    try
        sim  = run_model_attr(p(1),p(2),p(3),p(4),z,dz,Z,pS,n_air,X_atm,Cbulk0,T,dt,vel_profile);
        rmse = calc_rmse_attr(sim.X,z,depth,X_lab);
    catch
        rmse = 1e6;
    end
end

function t0 = rand_start_attr(idx_a, LB, UB)
    t0 = zeros(1,numel(idx_a));
    for k = 1:numel(idx_a)
        ip = idx_a(k);
        switch ip
            case 1
                lo = min(log(abs(LB(1))),log(abs(UB(1))));
                hi = max(log(abs(LB(1))),log(abs(UB(1))));
                t0(k) = lo+(hi-lo)*rand();
            case {2,4}
                t0(k) = log(LB(ip))+(log(UB(ip))-log(LB(ip)))*rand();
            case 3
                t0(k) = LB(3)+(UB(3)-LB(3))*rand();
        end
    end
end

function p = t2p_attr(t, ip, LB, UB)
    switch ip
        case 1,     p = -exp(t); p = max(LB(1),min(UB(1),p));
        case {2,4}, p =  exp(t); p = max(LB(ip),min(UB(ip),p));
        case 3,     p = max(LB(3),min(UB(3),t));
    end
end

%% =========================================================
%% VÉRIFICATION DES 3 CONDITIONS BASALES
%% =========================================================
function verify_BC_conditions(z, dz, Z, pS, n_air_vec, X_atm, dt, vel_profile)
    fprintf('\n========================================\n');
    fprintf('  VÉRIFICATION DES 3 CONDITIONS BASALES\n');
    fprintf('========================================\n');
    nz       = numel(z);
    D_test   = 1e-11 * 3.1536e7;
    T_test   = 1e4;
    nt_test  = round(T_test / dt);
    Cice_surf = X_atm * pS(end);
    Cb_surf   = X_atm * (pS(end) + n_air_vec(end));
    Cb0       = Cb_surf * ones(nz, 1);
    all_pass = true;
    fprintf('\n[TEST 1]  BC_bottom = ''zero_flux''\n');
    fprintf('  Config  : Umax=0, flux=0, CI=BC_surf (état d''équilibre)\n');
    fprintf('  Attendu : max|ΔC_bulk| < 1e-10 mol/m³\n');
    Cb1    = mini_sim(Cb0, 0, D_test, 0, 'zero_flux', Cice_surf, ...
                      z, dz, Z, pS, n_air_vec, Cb_surf, Cice_surf, dt, nt_test, vel_profile);
    drift1 = max(abs(Cb1 - Cb0));
    pass1  = drift1 < 1e-10;
    all_pass = all_pass & pass1;
    fprintf('  Résultat: max|ΔC_bulk| = %.2e mol/m³  →  %s\n', drift1, ok(pass1));
    fprintf('\n[TEST 2]  BC_bottom = ''dirichlet''\n');
    fprintf('  Config  : Umax=0, BC_bot=BC_surf=CI (équilibre symétrique)\n');
    fprintf('  Attendu : max|ΔC_bulk| < 1e-10 mol/m³\n');
    Cb2    = mini_sim(Cb0, 0, D_test, 0, 'dirichlet', Cice_surf, ...
                      z, dz, Z, pS, n_air_vec, Cb_surf, Cice_surf, dt, nt_test, vel_profile);
    drift2 = max(abs(Cb2 - Cb0));
    pass2  = drift2 < 1e-10;
    all_pass = all_pass & pass2;
    fprintf('  Résultat: max|ΔC_bulk| = %.2e mol/m³  →  %s\n', drift2, ok(pass2));
    F_test  = 1e-6;
    nt_mass = 50;
    T_mass  = nt_mass * dt;
    fprintf('\n[TEST 3]  BC_bottom = ''flux''\n');
    fprintf('  Config  : Umax=0, D=0, flux=%.1e mol/m²/yr, T=%.0f yr\n', F_test, T_mass);
    fprintf('  Attendu : ΔM_ice = %.6e mol/m²\n', F_test * T_mass);
    Cb3 = mini_sim(Cb0, 0, 0, F_test, 'flux', 0, ...
                   z, dz, Z, pS, n_air_vec, Cb_surf, Cice_surf, dt, nt_mass, vel_profile);
    Cice0_v = Cb0  .* pS ./ (pS + n_air_vec);
    Cice3   = Cb3  .* pS ./ (pS + n_air_vec);
    dM_ice      = trapz(z, Cice3 - Cice0_v);
    dM_expected = F_test * T_mass;
    err3   = abs(dM_ice - dM_expected) / dM_expected;
    pass3  = err3 < 1e-2;
    all_pass = all_pass & pass3;
    fprintf('  ΔM_ice  = %.6e mol/m²  (attendu %.6e)\n', dM_ice, dM_expected);
    fprintf('  Erreur relative = %.2e  →  %s\n', err3, ok(pass3));
    fprintf('\n----------------------------------------\n');
    if all_pass
        fprintf('  BILAN : TOUS LES TESTS RÉUSSIS  ✓\n');
    else
        fprintf('  BILAN : CERTAINS TESTS ONT ÉCHOUÉ  ✗\n');
        fprintf('          → Vérifier la matrice A_diff (Section 11)\n');
    end
    fprintf('========================================\n\n');
end

function Cb_out = mini_sim(Cb0, Umax, D, flux_bot, bc_type, Cice_bot_val, ...
                            z, dz, Z, pS, n_air, Cb_surf, Cice_surf, dt, nt, vel_profile)
    nz        = numel(z);
    u         = vel_profile(z, Z, Umax);
    u_face    = 0.5*(u(1:end-1)+u(2:end));  u_face(1) = 0;
    u_pos     = max(0, u_face);  u_neg = min(0, u_face);
    r = D * dt / dz^2;
    e = ones(nz,1);
    A = spdiags([-r*e, (1+2*r)*e, -r*e], [-1,0,1], nz, nz);
    switch bc_type
        case {'flux','zero_flux'}
            A(1,1) = 1+2*r;  A(1,2) = -2*r;
        case 'dirichlet'
            A(1,:) = 0;  A(1,1) = 1;
    end
    A(end,:) = 0;  A(end,end) = 1;
    dA = decomposition(A);
    Cb = Cb0;
    for k = 1:nt
        F = u_pos.*Cb(1:end-1) + u_neg.*Cb(2:end);
        F(end) = u_face(end) * Cb_surf;
        Ca = Cb;
        Ca(2:end-1) = Cb(2:end-1) + dt/dz*(F(1:end-1)-F(2:end));
        Ca(end) = Cb_surf;
        Cia = Ca ./ (pS + n_air) .* pS;
        B = Cia;
        switch bc_type
            case 'flux',      B(1) = Cia(1) + 2*flux_bot*dt/dz;
            case 'zero_flux', B(1) = Cia(1);
            case 'dirichlet', B(1) = Cice_bot_val;
        end
        B(end) = Cice_surf;
        Ci_new = dA \ B;
        Cb = Ca + (Ci_new - Cia);
    end
    Cb_out = Cb;
end

function s = ok(b)
    if b, s = 'PASS  ✓'; else, s = 'FAIL  ✗'; end
end


%% =========================================================
%% Figure — Profil de vitesse d'advection verticale u(z)
%% =========================================================
figure(10); clf;
plot(u * 1e3, z, 'b-', 'LineWidth', 2);
hold on;
yline(Z, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8);
xline(0,  '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.6);

% Annotation Umax
text(Umax*1e3 * 0.9, Z * 0.93, ...
    sprintf('U_{max} = %.3f \\times10^{-3} m/yr', Umax*1e3), ...
    'FontSize', 9, 'Color', [0.3 0.3 0.3], 'HorizontalAlignment', 'right');

xlabel('Vitesse verticale u  (\times10^{-3} m/yr)', 'FontSize', 11);
ylabel('Hauteur au-dessus du fond z  (m)',           'FontSize', 11);
title(sprintf('Profil Lliboutry (1979)  —  p = %d,  H_{ice} = %.0f m', p_llib, H_ice), ...
    'FontSize', 12);
grid on;
xlim([min(u)*1e3 * 1.15,  0.05]);
ylim([0, Z * 1.12]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% selon formulation DJ 
% %% 1D EULERIAN MODEL — ARGON TRANSPORT IN ICE (advection + lattice diffusion)
% %%
% %% State variables:
% %%   x       : Ar mixing ratio in clathrate gas    [mol/mol]
% %%   C_ice   : Ar dissolved in ice lattice          [mol/m³ ice]
% %%   C_bulk  : total Ar = clathrate + lattice       [mol/m³ ice]
% %%
% %% Thermodynamic equilibrium (instantaneous, at each timestep):
% %%   C_ice  = x * P_diss * S  =  x * pS
% %%   C_bulk = C_ice + x * n_air  =  x * (pS + n_air)
% %%   =>  x  = C_bulk / (pS + n_air)
% %%
% %% Grid convention:
% %%   z = 0   -> bed / fond du glacier   (MATLAB index 1, deepest)
% %%   z = Z   -> surface of the ice      (MATLAB index end, shallowest)
% %%   depth   -> depth from ice surface  (standard core convention, = depth_max - z)
% %%
% %% Reconstruction de C_bulk après diffusion (méthode Code 1) :
% %%   Seul l'incrément de la phase lattice est ajouté au réservoir total.
% %%   C_bulk(:,k+1) = C_bulk_adv + (Cice_new - Cice_adv)
% %%   Le ré-équilibre clathrate/lattice n'intervient qu'en étape E
% %%   via X = C_bulk / (pS + n_air).
% %%
% %% =========================================================
% clear; clc; close all;
% %% =========================================================
% %% *** TOUS LES PARAMÈTRES À MODIFIER SONT ICI ***
% %% =========================================================
% %  Ne modifier que ce bloc. Le reste du code s'adapte automatiquement.
% % --- Constantes physiques ---
% rho_ice    = 917;         % densité de la glace [kg/m³]
% Vm_STP     = 22.414e-3;  % volume molaire gaz idéal STP [m³/mol]
% P_diss_val = 5e6;         % pression dissociation clathrate Ar [Pa]
% S_Ar_val   = 10e-9;      % solubilité Ar dans glace [mol/m³/Pa]
% TAC_val    = 0.091e-3;   % teneur totale en air [m³ STP/kg glace] (Sailer)
% % --- Paramètres de transport (valeurs de référence littérature) ---
% D_m2s       = 1e-11;   % diffusivité lattice Ar [m²/s]
% flux_bottom = 1.8e-9;  % flux basal Ar [mol/m²/yr]
% % --- Profil de vitesse verticale — Dansgaard-Johnsen (1969) pour Camp Century ---
% % Selon le modèle DJ, la vitesse verticale en dessous de la hauteur de kink h
% % est approximativement constante : w(z < h) ≈ −b·h / (2H)
% % Dans notre domaine des ~30 m basaux, u passe de 0 (fond) à Umax (sommet).
% %   b_acc  = taux d'accumulation annuel [m/yr eq. glace]   — Dansgaard et al. 1969
% %   H_ice  = épaisseur totale de la calotte à Camp Century [m]
% %   h_kink = hauteur du coude DJ au-dessus du fond         [m]  (Hammer et al. 1978)
% b_acc   = 0.35;    % [m/yr ice equiv]
% H_ice   = 1387;   % [m]   — Camp Century (Dansgaard et al. 1969)
% h_kink  = 400;    % [m]   — Camp Century (Hammer et al. 1978)
% Umax = -b_acc * h_kink / (2 * H_ice);   % ≈ −0.050 m/yr (vitesse basale DJ)
% % Dans les 30 m, le profil passe de u=0 au fond à u=Umax au sommet du domaine.
% % Profil quadratique (défaut) — physiquement raisonnable dans la zone basale :
% vel_profile = @(z_col, Z_ref, Um)  Um * (z_col/Z_ref).^2;
% %   Linéaire   : vel_profile = @(z_col, Z_ref, Um)  Um * (z_col/Z_ref);
% %   Constant   : vel_profile = @(z_col, Z_ref, Um)  Um * ones(size(z_col));
% % gas_loss : taux de perte gazeuse λ [yr⁻¹], appliqué à la PHASE CLATHRATE uniquement
% %   Physique : des clathrates se vident entièrement → X et C_ice inchangés,
% %              seul C_bulk diminue via C_bulk = C_ice + (X*n_air)*exp(-λ*dt)
% % Relation avec le gradient observé dG/dz [ml/kg/m] :
% %   gas_loss = |dG/dz| * |Umax| / G_ref
% % Exemple : dG/dz = 0.9 ml/kg/m, |Umax|=1e-4 m/yr, G_ref=50 ml/kg
% %   => gas_loss ≈ 0.9 * 1e-4 / 50 = 1.8e-6 yr⁻¹
% gas_loss    = 0;        % taux de perte gazeuse  [yr⁻¹]  (0 = inactif)
% % --- Condition aux limites au fond (z = 0) ---
% % 'flux'      : flux Neumann imposé  = flux_bottom [mol/m²/yr]  (physique par défaut)
% % 'zero_flux' : flux nul (Neumann homogène)  — test : profil doit rester constant si CI uniforme
% % 'dirichlet' : valeur fixée C_ice = Cice_bottom_val            — test : BC symétrique
% BC_bottom_type  = 'flux';
% Cice_bottom_val = 0;    % [mol/m³] — utilisé uniquement si BC_bottom_type = 'dirichlet'
% % --- Numérique ---
% % NOTE : le schéma upwind 1er ordre introduit une diffusion numérique ∝ Umax*dz/2.
% %        Un schéma TVD (ex. van Leer) serait plus précis aux grandes échelles de temps.
% %        Pour l'instant l'upwind est conservé (Sandra, sept. 2026).
% nz = 101;     % nombre de nœuds sur la verticale
% T  = 1.25e5;  % durée totale de simulation [yr]  — âge glace Camp Century (~125 000 yr)
% % ATTENTION CFL : avec Umax_DJ ≈ −0.05 m/yr et dz ≈ 0.3 m,
% %   dt_max = dz/|Umax| ≈ 6 yr  →  dt = 3 yr donne CFL ≈ 0.5 (stable)
% dt = 3;       % pas de temps [yr]
% % --- Bornes d'optimisation (utilisées uniquement en Figure 8) ---
% %   ordre : [Umax   D_m2s   gas_loss   flux_bottom]
% LB_opt = [-2e-4,  2e-12, 0,      1e-10];
% UB_opt = [-2e-5,  5e-11, 1e-4,  2e-9 ];
% N_STARTS_opt = 10;   % nombre de points de départ aléatoires
% %% =========================================================
% %% — FIN DU BLOC PARAMÈTRES — NE PAS MODIFIER EN DESSOUS —
% %% =========================================================
% 
% %% =========================================================
% %% 1. Chargement des données Camp Century
% %% =========================================================
% data_file  = fullfile(fileparts(mfilename('fullpath')), ...
%     'Table_S1_122025_for manuscript.xlsx');
% sheet_name = 'CampCentury';
% raw = readmatrix(data_file, 'Sheet', sheet_name, 'Range', 'C5:AD59');
% depth     = raw(:,1);    X_lab     = raw(:,10);   X_u_lab   = raw(:,15);
% G_lab     = raw(:,2);    G_u_lab   = raw(:,3);    debris    = raw(:,4);
% bulk_CC   = raw(:,21);   bulk_CC_u = raw(:,26);
% valid   = ~isnan(depth) & ~isnan(X_lab) & ~isnan(bulk_CC);
% depth   = depth(valid);   G_lab   = G_lab(valid);   G_u_lab   = G_u_lab(valid);
% debris  = debris(valid);  X_lab   = X_lab(valid);   X_u_lab   = X_u_lab(valid);
% bulk_CC = bulk_CC(valid); bulk_CC_u = bulk_CC_u(valid);
% [depth, idx] = sort(depth);
% G_lab = G_lab(idx); G_u_lab = G_u_lab(idx); debris = debris(idx);
% X_lab = X_lab(idx); X_u_lab = X_u_lab(idx);
% bulk_CC = bulk_CC(idx); bulk_CC_u = bulk_CC_u(idx);
% depth_min = min(depth);  depth_max = max(depth);
% Z = depth_max - depth_min;
% fprintf('\nDataset: %s\n', data_file);
% fprintf('Depth range: %.2f – %.2f m  (column Z = %.2f m)\n', depth_min, depth_max, Z);
% 
% %% =========================================================
% %% 2. Grille verticale
% %% =========================================================
% z  = linspace(0, Z, nz)';
% dz = z(2) - z(1);
% depth_model = depth_max - z;
% 
% %% =========================================================
% %% 3. Temps
% %% =========================================================
% nt = round(T/dt);
% t  = (0:nt)*dt;
% 
% %% =========================================================
% %% 4. Grandeurs dérivées (calculées automatiquement depuis PARAMÈTRES)
% %% =========================================================
% D = D_m2s * 3.1536e7;           % [m²/yr]
% u = vel_profile(z, Z, Umax);    % profil défini dans PARAMÈTRES — ne pas modifier ici
% CFL_adv = max(abs(u)) * dt / dz;
% SC_diff  = D * dt / dz^2;
% fprintf('CFL (advection) = %.3g   |   Diffusion number = %.3g\n', CFL_adv, SC_diff);
% if CFL_adv > 1, warning('Advection CFL > 1 — reduce dt or increase dz.'); end
% P_diss = P_diss_val * ones(nz,1);
% S_Ar   = S_Ar_val   * ones(nz,1);
% pS     = P_diss .* S_Ar;
% 
% %% =========================================================
% %% 5. Teneur en air
% %% =========================================================
% n_air = TAC_val * rho_ice / Vm_STP;   % [mol air/m³ ice] — scalaire
% 
% %% =========================================================
% %% 6. Conversion données labo → unités modèle
% %% =========================================================
% Cbulk_lab         = bulk_CC * rho_ice;
% Cbulk_lab_profile = interp1(depth, Cbulk_lab, depth_model, 'linear','extrap');
% 
% %% =========================================================
% %% 7. Condition aux limites en surface (z = Z)
% %% =========================================================
% X_atm         = X_lab(1) / 100;
% Cice_surface  = X_atm * pS(end);
% Cbulk_surface = X_atm * (pS(end) + n_air);
% 
% %% =========================================================
% %% 8. Condition initiale (t = 0)
% %% =========================================================
% % CI cohérente avec BC surface : les deux utilisent X_lab(1) (échantillon le plus haut)
% % => à t=0, colonne uniforme au niveau atmosphérique (même valeur que la BC Dirichlet)
% Cbulk0 = Cbulk_surface * ones(nz,1);
% x0     = Cbulk0 ./ (pS + n_air);
% Cice0  = x0 .* pS;
% fprintf('\nIC = BC surface = %.4e mol/m³  (X_atm = %.2f ppm)\n', Cbulk_surface, X_atm*1e6);
% fprintf('IC partition check: max |C_bulk - (C_ice + x*n_air)| = %.3e mol/m³\n', ...
%     max(abs(Cbulk0 - (Cice0 + x0.*n_air))));
% 
% %% =========================================================
% %% 9. Flux basal (z = 0)
% %% =========================================================
% % flux_bottom défini dans la section PARAMÈTRES
% 
% %% =========================================================
% %% 10. Pré-allocation
% %% =========================================================
% C_bulk = zeros(nz, nt+1);  C_ice = zeros(nz, nt+1);
% X      = zeros(nz, nt+1);
% ADV    = zeros(nz, nt);    DIFF  = zeros(nz, nt);
% C_bulk(:,1) = Cbulk0;  C_ice(:,1) = Cice0;  X(:,1) = x0;
% 
% %% =========================================================
% %% 11. Matrice de diffusion implicite (Backward Euler)
% %% =========================================================
% % Physique (Sandra) : seul C_ice diffuse (grain boundary diffusion).
% % C_bulk est le "grand réservoir" qui ne diffuse pas directement.
% % Après chaque pas de diffusion, l'équilibre thermodynamique C_ice = X*pS
% % est restauré via X = C_bulk / (pS + n_air)  (étape E).
% r      = D * dt / dz^2;
% A_diff = sparse(nz, nz);
% for i = 2:nz-1
%     A_diff(i,i-1) = -r; A_diff(i,i) = 1+2*r; A_diff(i,i+1) = -r;
% end
% % BC au fond (z = 0) — choix via BC_bottom_type
% switch BC_bottom_type
%     case {'flux','zero_flux'}
%         % Neumann (point fantôme) : ∂C_ice/∂z|_0 = -flux_bottom/D
%         A_diff(1,1) = 1+2*r; A_diff(1,2) = -2*r;
%     case 'dirichlet'
%         % Dirichlet : C_ice(z=0) = Cice_bottom_val
%         A_diff(1,1) = 1;
%     otherwise
%         error('BC_bottom_type inconnu : %s', BC_bottom_type);
% end
% % BC en surface (z = Z) — Dirichlet (valeur atmosphérique imposée)
% A_diff(end,end) = 1;
% dA_diff = decomposition(A_diff);
% 
% %% =========================================================
% %% 12. Boucle temporelle
% %% =========================================================
% u_face    = 0.5*(u(1:end-1) + u(2:end));
% u_face(1) = 0;
% % gas_loss est directement λ [yr⁻¹] — decay par pas de temps dt [yr]
% % Équation : dC/dt = -λ*C  =>  C(t+dt) = C(t)*exp(-λ*dt)
% decay  = exp(-gas_loss * dt);
% 
% for k = 1:nt
% 
%     %% -- A. ADVECT C_bulk (forme conservative, upwind) ---------------- %%
%     C_old    = C_bulk(:,k);
%     adv_flux = zeros(nz-1,1);
%     for j = 2:nz-1
%         if u_face(j) >= 0, C_up = C_old(j); else, C_up = C_old(j+1); end
%         adv_flux(j) = u_face(j) * C_up;
%     end
%     adv_flux(nz-1) = u_face(nz-1) * Cbulk_surface;  % fix Sandra : BC surface imposée au flux upwind
%     C_bulk_adv = C_old;
%     for i = 2:nz-1
%         C_bulk_adv(i) = C_old(i) + dt*(adv_flux(i-1) - adv_flux(i))/dz;
%     end
%     C_bulk_adv(end) = Cbulk_surface;
%     ADV(:,k) = (C_bulk_adv - C_old)/dt;
% 
%     %% -- B. PARTITION ------------------------------------------------- %%
%     x_adv    = C_bulk_adv ./ (pS + n_air);
%     Cice_adv = x_adv .* pS;
% 
%     %% -- C. DIFFUSE C_ice (Backward Euler) ---------------------------- %%
%     B = Cice_adv;
%     switch BC_bottom_type
%         case 'flux'
%             B(1) = Cice_adv(1) + 2*flux_bottom*dt/dz;  % flux Neumann imposé
%         case 'zero_flux'
%             B(1) = Cice_adv(1);                          % flux nul (homogène)
%         case 'dirichlet'
%             B(1) = Cice_bottom_val;                       % valeur fixée
%     end
%     B(end) = Cice_surface;
%     Cice_new = dA_diff \ B;
% 
%     %% -- D. REBUILD C_bulk -------------------------------------------- %%
%     % Seul l'incrément de la phase lattice est ajouté au réservoir total.
%     % Les clathrates ne se re-remplissent PAS pendant l'étape de diffusion :
%     % c'est le re-équilibre thermodynamique (étape E) qui redistribue ensuite.
%     C_bulk(:,k+1) = C_bulk_adv + (Cice_new - Cice_adv);
%     DIFF(:,k)     = (C_bulk(:,k+1) - C_bulk_adv) / dt;
% 
%     %% -- E. RE-EQUILIBRATE -------------------------------------------- %%
%     X(:,k+1)     = C_bulk(:,k+1) ./ (pS + n_air);
%     C_ice(:,k+1) = X(:,k+1) .* pS;
% 
%     %% -- F. GAS LOSS (phase clathrate uniquement) --------------------- %%
%     if gas_loss > 0
%         % Des clathrates se vident : X et C_ice restent inchangés,
%         % seul le compartiment gazeux piégé diminue.
%         C_clath       = C_bulk(:,k+1) - C_ice(:,k+1);
%         C_bulk(:,k+1) = C_ice(:,k+1) + C_clath * decay;
%         % X(:,k+1) et C_ice(:,k+1) ne sont PAS modifiés
%     end
% 
% end
% fprintf('Simulation complete.\n');
% verify_BC_conditions(z, dz, Z, pS, n_air*ones(nz,1), X_atm, dt, vel_profile);
% 
% %% =========================================================
% %% 13. Retour aux unités labo
% %% =========================================================
% bulk_CC_model = C_bulk / rho_ice;
% 
% %% =========================================================
% %% 14. Diagnostics
% %% =========================================================
% partition_err = C_bulk - (C_ice + X.*n_air);
% fprintf('Partition consistency: max error = %.3e mol/m³\n', max(abs(partition_err(:))));
% M_bulk = trapz(z, C_bulk);
% fprintf('Final total Ar inventory: %.6e mol/m²\n', M_bulk(end));
% 
% %% =========================================================
% %% FIGURES 1–6  (diagnostics existants)
% %% =========================================================
% X_lab_grid = interp1(depth, X_lab/100, depth_model, 'linear','extrap');
% figure(1); clf;
% subplot(1,3,1);
%     plot(X(:,end)*1e6, z,'LineWidth',2); hold on;
%     plot(X_lab_grid*1e6, z,'o','MarkerSize',4);
%     xlabel('x (ppm)'); ylabel('Height above bed (m)');
%     title('Ar mixing ratio'); legend('Model','Lab','Location','best'); grid on; box on;
% subplot(1,3,2);
%     plot(C_ice(:,end), z,'LineWidth',2);
%     xlabel('C_{ice} (mol m^{-3})'); ylabel('Height above bed (m)');
%     title('Lattice Ar'); grid on; box on;
% subplot(1,3,3);
%     plot(C_bulk(:,end), z,'LineWidth',2); hold on;
%     plot(Cbulk_lab_profile, z,'o','MarkerSize',4);
%     xlabel('C_{bulk} (mol m^{-3})'); ylabel('Height above bed (m)');
%     title('Total Ar'); legend('Model','Lab','Location','best'); grid on; box on;
% sgtitle('Final state — thermodynamic partition');
% 
% n_snap = 6; idx_k = round(linspace(1,nt+1,n_snap));
% labels = arrayfun(@(ii) sprintf('t = %.2g yr',t(idx_k(ii))),1:n_snap,'UniformOutput',false);
% figure(2); clf;
% subplot(1,3,1); hold on;
%     for ii=1:n_snap; plot(X(:,idx_k(ii))*1e6, z,'LineWidth',1.8); end
%     xlabel('x (ppm)'); ylabel('Height above bed (m)'); title('x — mixing ratio'); grid on; box on;
% subplot(1,3,2); hold on;
%     for ii=1:n_snap; plot(C_ice(:,idx_k(ii)), z,'LineWidth',1.8); end
%     xlabel('C_{ice} (mol m^{-3})'); ylabel('Height above bed (m)'); title('C_{ice} — lattice'); grid on; box on;
% subplot(1,3,3); hold on;
%     for ii=1:n_snap; plot(C_bulk(:,idx_k(ii)), z,'LineWidth',1.8); end
%     xlabel('C_{bulk} (mol m^{-3})'); ylabel('Height above bed (m)'); title('C_{bulk} — total');
%     legend(labels,'Location','best','FontSize',8); grid on; box on;
% sgtitle('Temporal evolution');
% 
% skip_t = max(1,floor(nt/200)); idx_plot = 1:skip_t:nt;
% t_plot = t(idx_plot); ADV_plot = ADV(:,idx_plot); DIFF_plot = DIFF(:,idx_plot);
% figure(3); clf;
% subplot(1,2,1); pcolor(t_plot,z,ADV_plot); shading flat; colorbar;
%     xlabel('Time (yr)'); ylabel('Height above bed (m)');
%     title('Advective tendency [mol m^{-3} yr^{-1}]'); grid on; box on;
% subplot(1,2,2); pcolor(t_plot,z,DIFF_plot); shading flat; colorbar;
%     xlabel('Time (yr)'); ylabel('Height above bed (m)');
%     title('Diffusive tendency [mol m^{-3} yr^{-1}]'); grid on; box on;
% sgtitle('Transport tendencies');
% 
% figure(4); clf;
% subplot(1,3,1);
%     plot(G_lab,depth,'o-','LineWidth',1.5); set(gca,'YDir','reverse');
%     xlabel('Total gas content (mL/kg)'); ylabel('Depth from surface (m)');
%     title('Total gas content'); grid on; box on;
% subplot(1,3,2);
%     plot([TAC_val TAC_val],[depth_min depth_max],'r-','LineWidth',2);
%     set(gca,'YDir','reverse'); xlim([0 2*TAC_val]);
%     xlabel('TAC (m³ STP kg^{-1})'); ylabel('Depth from surface (m)');
%     title('TAC (constant — Sailer)'); grid on; box on;
% subplot(1,3,3);
%     plot(X_lab,depth,'o-','LineWidth',1.5); set(gca,'YDir','reverse');
%     xlabel('Ar mixing ratio (%)'); ylabel('Depth from surface (m)');
%     title('Ar mixing ratio'); grid on; box on;
% sgtitle('Camp Century — lab measurements');
% 
% figure(5); clf;
% subplot(1,3,1);
%     plot(X_lab/100,depth,'ko','MarkerSize',5); hold on;
%     plot(X(:,end),depth_model,'LineWidth',2); set(gca,'YDir','reverse');
%     xlabel('x (mol/mol)'); ylabel('Depth from surface (m)');
%     title('Ar mixing ratio'); legend('Lab','Model','Location','best'); grid on; box on;
% subplot(1,3,2);
%     plot(bulk_CC,depth,'ko','MarkerSize',5); hold on;
%     plot(bulk_CC_model(:,end),depth_model,'LineWidth',2); set(gca,'YDir','reverse');
%     xlabel('C_{bulk} (mol kg^{-1})'); ylabel('Depth from surface (m)');
%     title('Total Ar'); legend('Lab','Model','Location','best'); grid on; box on;
% subplot(1,3,3);
%     plot(C_ice(:,end),depth_model,'LineWidth',2); set(gca,'YDir','reverse');
%     xlabel('C_{ice} (mol m^{-3})'); ylabel('Depth from surface (m)');
%     title('Lattice Ar (model only)'); grid on; box on;
% sgtitle('Camp Century — model vs lab');
% 
% frac_lattice = C_ice(:,end) ./ C_bulk(:,end);
% frac_clath   = 1 - frac_lattice;
% figure(6); clf;
%     plot(frac_lattice*100, z,'LineWidth',2); hold on;
%     plot(frac_clath*100,   z,'LineWidth',2);
%     xlabel('Fraction of total Ar (%)'); ylabel('Height above bed (m)');
%     title('Ar partition — lattice vs clathrates (final state)');
%     legend('Lattice','Clathrates','Location','best'); grid on; box on;
% fprintf('\nFinal Ar partition:\n');
% fprintf('  Lattice:    %.4f – %.4f %%\n', 100*min(frac_lattice),100*max(frac_lattice));
% fprintf('  Clathrates: %.4f – %.4f %%\n', 100*min(frac_clath),  100*max(frac_clath));
% 
% %% =========================================================
% %% FIGURE 9 — Construction progressive des processus (état final t = T)
% %%   Courbe 1 : Diffusion + Source basale              (Umax=0, gas_loss=0)
% %%   Courbe 2 : + Advection                            (Umax≠0, gas_loss=0)
% %%   Courbe 3 : + Gas loss = modèle complet            (Umax≠0, gas_loss≠0)
% %% NOTE : X et C_ice ne changent PAS avec gas_loss (clathrate phase uniquement)
% %%        → courbes 2 et 3 identiques pour X et C_ice ; C_bulk diverge
% %% =========================================================
% fprintf('\n=== FIGURE 9 : Construction progressive des processus ===\n');
% n_air_9 = n_air * ones(nz,1);
% sim9a = run_model_attr(0,     D_m2s, 0,        flux_bottom, ...
%             z, dz, Z, pS, n_air_9, X_atm, Cbulk0, T, dt, vel_profile);   % Diff + Source
% sim9b = run_model_attr(Umax,  D_m2s, 0,        flux_bottom, ...
%             z, dz, Z, pS, n_air_9, X_atm, Cbulk0, T, dt, vel_profile);   % + Advection
% sim9c = run_model_attr(Umax,  D_m2s, gas_loss, flux_bottom, ...
%             z, dz, Z, pS, n_air_9, X_atm, Cbulk0, T, dt, vel_profile);   % + Gas loss
% 
% X9    = {sim9a.X,          sim9b.X,          sim9c.X};
% Cice9 = {sim9a.X .* pS,   sim9b.X .* pS,   sim9c.X .* pS};
% Cb9   = {sim9a.C*rho_ice,  sim9b.C*rho_ice,  sim9c.C*rho_ice};
% 
% col9 = [0.000 0.447 0.698;   % bleu      — Diff + Source
%         0.902 0.624 0.000;   % orange    — + Advection
%         0.835 0.369 0.000];  % vermillon — + Gas loss
% lbl9 = {'Diff + Source', '+ Advection', '+ Gas loss (complet)'};
% lw9  = 2.2;
% figure(9); clf; set(gcf,'Position',[100 100 1200 500]);
% subplot(1,3,1); hold on;
%     plot(X_atm*1e6*ones(2,1), [0 Z],'k:','LineWidth',1,'DisplayName','CI (t=0)');
%     for ii = 1:3
%         plot(X9{ii}*1e6, z, 'LineWidth',lw9, 'Color',col9(ii,:), 'DisplayName',lbl9{ii});
%     end
%     xlabel('x (ppm)'); ylabel('Height above bed (m)');
%     title('x — mixing ratio');
%     legend('Location','best','FontSize',8); grid on; box on;
% subplot(1,3,2); hold on;
%     plot(X_atm*pS(1)*1e3*ones(2,1), [0 Z],'k:','LineWidth',1,'HandleVisibility','off');
%     for ii = 1:3
%         plot(Cice9{ii}*1e3, z, 'LineWidth',lw9, 'Color',col9(ii,:), 'DisplayName',lbl9{ii});
%     end
%     xlabel('C_{ice} (mol m^{-3}) \times 10^{-3}'); ylabel('Height above bed (m)');
%     title('C_{ice} — lattice (inchangée par gas loss)');
%     grid on; box on;
% subplot(1,3,3); hold on;
%     plot(Cbulk_surface*ones(2,1), [0 Z],'k:','LineWidth',1,'HandleVisibility','off');
%     for ii = 1:3
%         plot(Cb9{ii}, z, 'LineWidth',lw9, 'Color',col9(ii,:), 'DisplayName',lbl9{ii});
%     end
%     xlabel('C_{bulk} (mol m^{-3})'); ylabel('Height above bed (m)');
%     title('C_{bulk} — total (affectée par gas loss)');
%     grid on; box on;
% sgtitle(sprintf('Construction progressive des processus  (t = %.2g yr)', T), 'FontSize',13);
% fprintf('Figure 9 : OK\n');
% 
% %% =========================================================
% %% FIGURE 7 — Importance des processus (leave-one-out)
% %% Valeurs de référence imposées — AUCUNE optimisation
% %% =========================================================
% fprintf('\n=== FIGURE 7 : Attribution (valeurs littérature) ===\n');
% p_ref = [Umax, D_m2s, gas_loss, flux_bottom];
% NULL_VAL    = [0,     1e-15, 0,  0];
% PROC_NAMES  = {'U_{max}','D','gas_{loss}','flux_{bot}'};
% n_air_vec = n_air * ones(nz,1);
% sim_all  = run_model_attr(p_ref(1),p_ref(2),p_ref(3),p_ref(4), ...
%                z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt,vel_profile);
% RMSE_all = calc_rmse_attr(sim_all.X, z, depth, X_lab);
% fprintf('RMSE (tous processus) : %.4e\n', RMSE_all);
% RMSE_loo = nan(4,1);
% for ip = 1:4
%     p_tmp     = p_ref;  p_tmp(ip) = NULL_VAL(ip);
%     sim_tmp   = run_model_attr(p_tmp(1),p_tmp(2),p_tmp(3),p_tmp(4), ...
%                     z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt,vel_profile);
%     RMSE_loo(ip) = calc_rmse_attr(sim_tmp.X, z, depth, X_lab);
%     fprintf('Sans %-12s : RMSE = %.4e  (Δ = +%.4e)\n', ...
%         PROC_NAMES{ip}, RMSE_loo(ip), RMSE_loo(ip)-RMSE_all);
% end
% delta = RMSE_loo - RMSE_all;
% [delta_sort, isort] = sort(delta,'descend');
% names_sort = PROC_NAMES(isort);
% CLR4 = [0.00 0.45 0.70; 0.90 0.62 0.00; 0.00 0.62 0.45; 0.80 0.40 0.00];
% clr_sort = CLR4(isort,:);
% figure(7); clf;
% ax7 = axes; hold(ax7,'on');
% for k = 1:4
%     bar(ax7,k,delta_sort(k),0.6,'FaceColor',clr_sort(k,:),'EdgeColor','none');
%     text(ax7,k,delta_sort(k)+max(delta_sort)*0.04, ...
%         sprintf('+%.2e',delta_sort(k)), ...
%         'HorizontalAlignment','center','FontSize',10,'FontWeight','bold', ...
%         'Color',clr_sort(k,:)*0.75);
% end
% set(ax7,'XTick',1:4,'XTickLabel',names_sort,'FontSize',12,'Box','off');
% ylabel(ax7,'\DeltaRMSE_X  (mol mol^{-1})','FontSize',11);
% title(ax7,{'Importance relative des processus de transport', ...
%     '(\DeltaRMSE = augmentation quand le processus est éteint)'}, ...
%     'FontSize',11,'FontWeight','bold');
% y_top = max(delta_sort) * 1.20;
% if y_top <= 0, y_top = 1e-6; end
% ylim(ax7,[0, y_top]);
% grid(ax7,'on'); ax7.GridAlpha = 0.2;
% saveas(figure(7),'figure7_importance.png');
% fprintf('Figure 7 sauvegardée : figure7_importance.png\n');
% 
% %% =========================================================
% %% FIGURE 8 — Profils X : best fit optimisé (leave-one-out)
% %% =========================================================
% fprintf('\n=== FIGURE 8 : Profils X avec optimisation ===\n');
% opts_opt = optimset('MaxIter',5000,'TolFun',1e-10,'Display','off');
% idx_all  = [1 2 3 4];
% objfun_all = @(t_vec) obj_attr(t_vec, idx_all, p_ref, LB_opt, UB_opt, ...
%     z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt,depth,X_lab,vel_profile);
% best_r = inf; best_t = [];
% for ns = 1:N_STARTS_opt
%     t0 = rand_start_attr(idx_all, LB_opt, UB_opt);
%     try
%         [t_s,r_s] = fminsearch(objfun_all,t0,opts_opt);
%         if r_s < best_r; best_r = r_s; best_t = t_s; end
%     catch; end
% end
% p_opt = p_ref;
% for k = 1:4; p_opt(k) = t2p_attr(best_t(k),k,LB_opt,UB_opt); end
% RMSE_opt = best_r;
% fprintf('Best fit : RMSE=%.3e  Umax=%.2e  D=%.2e  gas_loss=%.3f  flux_bot=%.2e\n', ...
%     RMSE_opt, p_opt(1),p_opt(2),p_opt(3),p_opt(4));
% sim_opt  = run_model_attr(p_opt(1),p_opt(2),p_opt(3),p_opt(4), ...
%                z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt,vel_profile);
% X_opt    = sim_opt.X * 100;
% X_loo_opt = cell(4,1);
% RMSE_loo_opt = nan(4,1);
% for ip = 1:4
%     p_tmp     = p_opt; p_tmp(ip) = NULL_VAL(ip);
%     sim_tmp   = run_model_attr(p_tmp(1),p_tmp(2),p_tmp(3),p_tmp(4), ...
%                     z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt,vel_profile);
%     X_loo_opt{ip}    = sim_tmp.X * 100;
%     RMSE_loo_opt(ip) = calc_rmse_attr(sim_tmp.X,z,depth,X_lab);
% end
% [~,imax_lab] = max(X_lab);
% hab_sed      = depth_max - depth(imax_lab);
% BROWN        = [0.55 0.27 0.07];
% LW = 2; MS = 6;
% all_X_fig8 = [X_lab; X_opt; cell2mat(cellfun(@(v)v,X_loo_opt,'UniformOutput',false))];
% xlim8 = [min(all_X_fig8)*0.97, max(all_X_fig8)*1.03];
% figure(8); clf;
% ax8 = axes; hold(ax8,'on');
% if hab_sed > 0
%     patch(ax8,xlim8([1 2 2 1]),[0 0 hab_sed hab_sed],BROWN, ...
%         'FaceAlpha',0.25,'EdgeColor','none');
% end
% plot(ax8,X_lab,depth_max-depth,'ko','MarkerSize',MS,'MarkerFaceColor','w', ...
%     'LineWidth',1.2,'DisplayName','Données labo');
% for ip = 1:4
%     plot(ax8,X_loo_opt{ip},z,'--','Color',CLR4(ip,:),'LineWidth',1.4, ...
%         'DisplayName',sprintf('sans %s',PROC_NAMES{ip}));
% end
% plot(ax8,X_opt,z,'-','Color',[0 0 0],'LineWidth',2.5, ...
%     'DisplayName',sprintf('Best fit (RMSE=%.2e)',RMSE_opt));
% xlabel(ax8,'Ar mixing ratio (%)','FontSize',11);
% ylabel(ax8,'Hauteur au-dessus du fond (m)','FontSize',11);
% title(ax8,{'Profils X — attribution par processus', ...
%     'Pointillés = best fit sans ce paramètre'},'FontSize',11,'FontWeight','bold');
% xlim(ax8,xlim8); ylim(ax8,[0 Z*1.05]);
% legend(ax8,'Location','northwest','FontSize',8,'Box','off');
% box(ax8,'off'); grid(ax8,'on'); ax8.GridAlpha = 0.2;
% saveas(figure(8),'figure8_profils.png');
% fprintf('Figure 8 sauvegardée : figure8_profils.png\n');
% 
% %% =========================================================
% %% FONCTIONS LOCALES
% %% =========================================================
% 
% function sim = run_model_attr(Umax, D_m2s, gas_loss, flux_bottom, ...
%         z, dz, Z, pS, n_air, X_atm, Cbulk0, T, dt, vel_profile)
% % Version vectorisée pour l'attribution (Figures 7, 8, 9)
% % sim.X = mixing ratio après re-équilibre (inchangé par gas_loss)
% % sim.C = C_bulk final / rho_ice (affecté par gas_loss)
%     rho_ice = 917;
%     D  = D_m2s * 3.1536e7;
%     nz = numel(z);
%     nt = round(T/dt);
%     u      = vel_profile(z, Z, Umax);
%     u_face = 0.5*(u(1:end-1)+u(2:end)); u_face(1) = 0;
%     u_pos  = max(0,u_face);  u_neg = min(0,u_face);
%     Cice_surf = X_atm * pS(end);
%     Cb_surf   = X_atm * (pS(end) + n_air(end));
%     C_bulk    = Cbulk0;
%     r = D*dt/dz^2; e = ones(nz,1);
%     A = spdiags([-r*e,(1+2*r)*e,-r*e],[-1,0,1],nz,nz);
%     A(1,1) = 1+2*r; A(1,2) = -2*r;
%     A(end,:) = 0;   A(end,end) = 1;
%     dA = decomposition(A);
%     decay_a = exp(-gas_loss * dt);
%     for k = 1:nt
%         F  = u_pos.*C_bulk(1:end-1) + u_neg.*C_bulk(2:end);
%         F(end) = u_face(end) * Cb_surf;
%         Ca = C_bulk;
%         Ca(2:end-1) = C_bulk(2:end-1) + dt/dz*(F(1:end-1)-F(2:end));
%         Ca(end) = Cb_surf;
%         xa  = Ca./(pS+n_air);  Cia = xa.*pS;
%         B = Cia; B(1) = Cia(1)+2*flux_bottom*dt/dz; B(end) = Cice_surf;
%         Cice_new_a = dA\B;
%         C_bulk = Ca + (Cice_new_a - Cia);
%         x_new_a = C_bulk ./ (pS + n_air);
%         if gas_loss > 0
%             C_ice_a   = x_new_a .* pS;
%             C_clath_a = C_bulk - C_ice_a;
%             C_bulk    = C_ice_a + C_clath_a * decay_a;
%         end
%     end
%     sim.X = x_new_a;
%     sim.C = C_bulk / rho_ice;
% end
% 
% function rmse = calc_rmse_attr(X_model, z, depth, X_lab)
%     depth_max = max(depth);
%     X_interp  = interp1(depth_max-z, X_model, depth,'linear','extrap');
%     rmse      = sqrt(mean((X_interp - X_lab/100).^2));
% end
% 
% function rmse = obj_attr(t_vec, idx_a, p_off, LB, UB, ...
%         z,dz,Z,pS,n_air,X_atm,Cbulk0,T,dt,depth,X_lab,vel_profile)
%     p = p_off;
%     for k = 1:numel(idx_a)
%         p(idx_a(k)) = t2p_attr(t_vec(k),idx_a(k),LB,UB);
%     end
%     try
%         sim  = run_model_attr(p(1),p(2),p(3),p(4),z,dz,Z,pS,n_air,X_atm,Cbulk0,T,dt,vel_profile);
%         rmse = calc_rmse_attr(sim.X,z,depth,X_lab);
%     catch
%         rmse = 1e6;
%     end
% end
% 
% function t0 = rand_start_attr(idx_a, LB, UB)
%     t0 = zeros(1,numel(idx_a));
%     for k = 1:numel(idx_a)
%         ip = idx_a(k);
%         switch ip
%             case 1
%                 lo = min(log(abs(LB(1))),log(abs(UB(1))));
%                 hi = max(log(abs(LB(1))),log(abs(UB(1))));
%                 t0(k) = lo+(hi-lo)*rand();
%             case {2,4}
%                 t0(k) = log(LB(ip))+(log(UB(ip))-log(LB(ip)))*rand();
%             case 3
%                 t0(k) = LB(3)+(UB(3)-LB(3))*rand();
%         end
%     end
% end
% 
% function p = t2p_attr(t, ip, LB, UB)
%     switch ip
%         case 1,     p = -exp(t); p = max(LB(1),min(UB(1),p));
%         case {2,4}, p =  exp(t); p = max(LB(ip),min(UB(ip),p));
%         case 3,     p = max(LB(3),min(UB(3),t));
%     end
% end
% 
% %% =========================================================
% %% VÉRIFICATION DES 3 CONDITIONS BASALES
% %% =========================================================
% function verify_BC_conditions(z, dz, Z, pS, n_air_vec, X_atm, dt, vel_profile)
%     fprintf('\n========================================\n');
%     fprintf('  VÉRIFICATION DES 3 CONDITIONS BASALES\n');
%     fprintf('========================================\n');
%     nz       = numel(z);
%     D_test   = 1e-11 * 3.1536e7;
%     T_test   = 1e4;
%     nt_test  = round(T_test / dt);
%     Cice_surf = X_atm * pS(end);
%     Cb_surf   = X_atm * (pS(end) + n_air_vec(end));
%     Cb0       = Cb_surf * ones(nz, 1);
%     all_pass = true;
%     fprintf('\n[TEST 1]  BC_bottom = ''zero_flux''\n');
%     fprintf('  Config  : Umax=0, flux=0, CI=BC_surf (état d''équilibre)\n');
%     fprintf('  Attendu : max|ΔC_bulk| < 1e-10 mol/m³\n');
%     Cb1    = mini_sim(Cb0, 0, D_test, 0, 'zero_flux', Cice_surf, ...
%                       z, dz, Z, pS, n_air_vec, Cb_surf, Cice_surf, dt, nt_test, vel_profile);
%     drift1 = max(abs(Cb1 - Cb0));
%     pass1  = drift1 < 1e-10;
%     all_pass = all_pass & pass1;
%     fprintf('  Résultat: max|ΔC_bulk| = %.2e mol/m³  →  %s\n', drift1, ok(pass1));
%     fprintf('\n[TEST 2]  BC_bottom = ''dirichlet''\n');
%     fprintf('  Config  : Umax=0, BC_bot=BC_surf=CI (équilibre symétrique)\n');
%     fprintf('  Attendu : max|ΔC_bulk| < 1e-10 mol/m³\n');
%     Cb2    = mini_sim(Cb0, 0, D_test, 0, 'dirichlet', Cice_surf, ...
%                       z, dz, Z, pS, n_air_vec, Cb_surf, Cice_surf, dt, nt_test, vel_profile);
%     drift2 = max(abs(Cb2 - Cb0));
%     pass2  = drift2 < 1e-10;
%     all_pass = all_pass & pass2;
%     fprintf('  Résultat: max|ΔC_bulk| = %.2e mol/m³  →  %s\n', drift2, ok(pass2));
%     F_test  = 1e-6;
%     nt_mass = 50;
%     T_mass  = nt_mass * dt;
%     fprintf('\n[TEST 3]  BC_bottom = ''flux''\n');
%     fprintf('  Config  : Umax=0, D=0, flux=%.1e mol/m²/yr, T=%.0f yr\n', F_test, T_mass);
%     fprintf('  Attendu : ΔM_ice = %.6e mol/m²\n', F_test * T_mass);
%     Cb3 = mini_sim(Cb0, 0, 0, F_test, 'flux', 0, ...
%                    z, dz, Z, pS, n_air_vec, Cb_surf, Cice_surf, dt, nt_mass, vel_profile);
%     Cice0_v = Cb0  .* pS ./ (pS + n_air_vec);
%     Cice3   = Cb3  .* pS ./ (pS + n_air_vec);
%     dM_ice      = trapz(z, Cice3 - Cice0_v);
%     dM_expected = F_test * T_mass;
%     err3   = abs(dM_ice - dM_expected) / dM_expected;
%     pass3  = err3 < 1e-2;
%     all_pass = all_pass & pass3;
%     fprintf('  ΔM_ice  = %.6e mol/m²  (attendu %.6e)\n', dM_ice, dM_expected);
%     fprintf('  Erreur relative = %.2e  →  %s\n', err3, ok(pass3));
%     fprintf('\n----------------------------------------\n');
%     if all_pass
%         fprintf('  BILAN : TOUS LES TESTS RÉUSSIS  ✓\n');
%     else
%         fprintf('  BILAN : CERTAINS TESTS ONT ÉCHOUÉ  ✗\n');
%         fprintf('          → Vérifier la matrice A_diff (Section 11)\n');
%     end
%     fprintf('========================================\n\n');
% end
% 
% function Cb_out = mini_sim(Cb0, Umax, D, flux_bot, bc_type, Cice_bot_val, ...
%                             z, dz, Z, pS, n_air, Cb_surf, Cice_surf, dt, nt, vel_profile)
%     nz        = numel(z);
%     u         = vel_profile(z, Z, Umax);
%     u_face    = 0.5*(u(1:end-1)+u(2:end));  u_face(1) = 0;
%     u_pos     = max(0, u_face);  u_neg = min(0, u_face);
%     r = D * dt / dz^2;
%     e = ones(nz,1);
%     A = spdiags([-r*e, (1+2*r)*e, -r*e], [-1,0,1], nz, nz);
%     switch bc_type
%         case {'flux','zero_flux'}
%             A(1,1) = 1+2*r;  A(1,2) = -2*r;
%         case 'dirichlet'
%             A(1,:) = 0;  A(1,1) = 1;
%     end
%     A(end,:) = 0;  A(end,end) = 1;
%     dA = decomposition(A);
%     Cb = Cb0;
%     for k = 1:nt
%         F = u_pos.*Cb(1:end-1) + u_neg.*Cb(2:end);
%         F(end) = u_face(end) * Cb_surf;
%         Ca = Cb;
%         Ca(2:end-1) = Cb(2:end-1) + dt/dz*(F(1:end-1)-F(2:end));
%         Ca(end) = Cb_surf;
%         Cia = Ca ./ (pS + n_air) .* pS;
%         B = Cia;
%         switch bc_type
%             case 'flux',      B(1) = Cia(1) + 2*flux_bot*dt/dz;
%             case 'zero_flux', B(1) = Cia(1);
%             case 'dirichlet', B(1) = Cice_bot_val;
%         end
%         B(end) = Cice_surf;
%         Ci_new = dA \ B;
%         Cb = Ca + (Ci_new - Cia);
%     end
%     Cb_out = Cb;
% end
% 
% function s = ok(b)
%     if b, s = 'PASS  ✓'; else, s = 'FAIL  ✗'; end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%% SANS DJ %%%%% 
% %% 1D EULERIAN MODEL — ARGON TRANSPORT IN ICE (advection + lattice diffusion)
% %%
% %% State variables:
% %%   x       : Ar mixing ratio in clathrate gas    [mol/mol]
% %%   C_ice   : Ar dissolved in ice lattice          [mol/m³ ice]
% %%   C_bulk  : total Ar = clathrate + lattice       [mol/m³ ice]
% %%
% %% Thermodynamic equilibrium (instantaneous, at each timestep):
% %%   C_ice  = x * P_diss * S  =  x * pS
% %%   C_bulk = C_ice + x * n_air  =  x * (pS + n_air)
% %%   =>  x  = C_bulk / (pS + n_air)
% %%
% %% Grid convention:
% %%   z = 0   -> bed / fond du glacier   (MATLAB index 1, deepest)
% %%   z = Z   -> surface of the ice      (MATLAB index end, shallowest)
% %%   depth   -> depth from ice surface  (standard core convention, = depth_max - z)
% %%
% %% Reconstruction de C_bulk après diffusion (méthode Code 1) :
% %%   Seul l'incrément de la phase lattice est ajouté au réservoir total.
% %%   C_bulk(:,k+1) = C_bulk_adv + (Cice_new - Cice_adv)
% %%   Le ré-équilibre clathrate/lattice n'intervient qu'en étape E
% %%   via X = C_bulk / (pS + n_air).
% %%
% %% =========================================================
% clear; clc; close all;
% %% =========================================================
% %% *** TOUS LES PARAMÈTRES À MODIFIER SONT ICI ***
% %% =========================================================
% %  Ne modifier que ce bloc. Le reste du code s'adapte automatiquement.
% % --- Constantes physiques ---
% rho_ice    = 917;         % densité de la glace [kg/m³]
% Vm_STP     = 22.414e-3;  % volume molaire gaz idéal STP [m³/mol]
% P_diss_val = 5e6;         % pression dissociation clathrate Ar [Pa]
% S_Ar_val   = 10e-9;      % solubilité Ar dans glace [mol/m³/Pa]
% TAC_val    = 0.091e-3;   % teneur totale en air [m³ STP/kg glace] (Sailer)
% % --- Paramètres de transport (valeurs de référence littérature) ---
% D_m2s       = 1e-11;   % diffusivité lattice Ar [m²/s]
% Umax        = -1e-6;   % vitesse max glace en surface [m/yr]  (< 0 = vers le bas)
% flux_bottom = 1.8e-9;  % flux basal Ar [mol/m²/yr]
% % gas_loss : taux de perte gazeuse λ [yr⁻¹], appliqué à la PHASE CLATHRATE uniquement
% %   Physique : des clathrates se vident entièrement → X et C_ice inchangés,
% %              seul C_bulk diminue via C_bulk = C_ice + (X*n_air)*exp(-λ*dt)
% % Relation avec le gradient observé dG/dz [ml/kg/m] :
% %   gas_loss = |dG/dz| * |Umax| / G_ref
% % Exemple : dG/dz = 0.9 ml/kg/m, |Umax|=1e-4 m/yr, G_ref=50 ml/kg
% %   => gas_loss ≈ 0.9 * 1e-4 / 50 = 1.8e-6 yr⁻¹
% gas_loss    = 0;        % taux de perte gazeuse  [yr⁻¹]  (0 = inactif)
% % --- Condition aux limites au fond (z = 0) ---
% % 'flux'      : flux Neumann imposé  = flux_bottom [mol/m²/yr]  (physique par défaut)
% % 'zero_flux' : flux nul (Neumann homogène)  — test : profil doit rester constant si CI uniforme
% % 'dirichlet' : valeur fixée C_ice = Cice_bottom_val            — test : BC symétrique
% BC_bottom_type  = 'flux';
% Cice_bottom_val = 0;    % [mol/m³] — utilisé uniquement si BC_bottom_type = 'dirichlet'
% % --- Numérique ---
% % NOTE : le schéma upwind 1er ordre introduit une diffusion numérique ∝ Umax*dz/2.
% %        Un schéma TVD (ex. van Leer) serait plus précis aux grandes échelles de temps.
% %        Pour l'instant l'upwind est conservé (Sandra, sept. 2026).
% nz = 101;     % nombre de nœuds sur la verticale
% T  = 8e7;    % durée totale de simulation [yr]
% dt = 100;    % pas de temps [yr]
% % --- Bornes d'optimisation (utilisées uniquement en Figure 8) ---
% %   ordre : [Umax   D_m2s   gas_loss   flux_bottom]
% LB_opt = [-2e-4,  2e-12, 0,      1e-10];
% UB_opt = [-2e-5,  5e-11, 1e-4,  2e-9 ];
% N_STARTS_opt = 10;   % nombre de points de départ aléatoires
% %% =========================================================
% %% — FIN DU BLOC PARAMÈTRES — NE PAS MODIFIER EN DESSOUS —
% %% =========================================================
% %% =========================================================
% %% 1. Chargement des données Camp Century
% %% =========================================================
% data_file  = fullfile(fileparts(mfilename('fullpath')), ...
% 'Table_S1_122025_for manuscript.xlsx');
% sheet_name = 'CampCentury';
% raw = readmatrix(data_file, 'Sheet', sheet_name, 'Range', 'C5:AD59');
% depth     = raw(:,1);    X_lab     = raw(:,10);   X_u_lab   = raw(:,15);
% G_lab     = raw(:,2);    G_u_lab   = raw(:,3);    debris    = raw(:,4);
% bulk_CC   = raw(:,21);   bulk_CC_u = raw(:,26);
% valid   = ~isnan(depth) & ~isnan(X_lab) & ~isnan(bulk_CC);
% depth   = depth(valid);   G_lab   = G_lab(valid);   G_u_lab   = G_u_lab(valid);
% debris  = debris(valid);  X_lab   = X_lab(valid);   X_u_lab   = X_u_lab(valid);
% bulk_CC = bulk_CC(valid); bulk_CC_u = bulk_CC_u(valid);
% [depth, idx] = sort(depth);
% G_lab = G_lab(idx); G_u_lab = G_u_lab(idx); debris = debris(idx);
% X_lab = X_lab(idx); X_u_lab = X_u_lab(idx);
% bulk_CC = bulk_CC(idx); bulk_CC_u = bulk_CC_u(idx);
% depth_min = min(depth);  depth_max = max(depth);
% Z = depth_max - depth_min;
% fprintf('\nDataset: %s\n', data_file);
% fprintf('Depth range: %.2f – %.2f m  (column Z = %.2f m)\n', depth_min, depth_max, Z);
% %% =========================================================
% %% 2. Grille verticale
% %% =========================================================
% z  = linspace(0, Z, nz)';
% dz = z(2) - z(1);
% depth_model = depth_max - z;
% %% =========================================================
% %% 3. Temps
% %% =========================================================
% nt = round(T/dt);
% t  = (0:nt)*dt;
% %% =========================================================
% %% 4. Grandeurs dérivées (calculées automatiquement depuis PARAMÈTRES)
% %% =========================================================
% D = D_m2s * 3.1536e7;           % [m²/yr]
% u = Umax * zeros(nz, 1); %u = Umax * (z/Z).^2;
% CFL_adv = max(abs(u)) * dt / dz;
% SC_diff  = D * dt / dz^2;
% fprintf('CFL (advection) = %.3g   |   Diffusion number = %.3g\n', CFL_adv, SC_diff);
% if CFL_adv > 1, warning('Advection CFL > 1 — reduce dt or increase dz.'); end
% P_diss = P_diss_val * ones(nz,1);
% S_Ar   = S_Ar_val   * ones(nz,1);
% pS     = P_diss .* S_Ar;
% %% =========================================================
% %% 5. Teneur en air
% %% =========================================================
% n_air = TAC_val * rho_ice / Vm_STP;   % [mol air/m³ ice] — scalaire
% %% =========================================================
% %% 6. Conversion données labo → unités modèle
% %% =========================================================
% Cbulk_lab         = bulk_CC * rho_ice;
% Cbulk_lab_profile = interp1(depth, Cbulk_lab, depth_model, 'linear','extrap');
% %% =========================================================
% %% 7. Condition aux limites en surface (z = Z)
% %% =========================================================
% X_atm         = X_lab(1) / 100;
% Cice_surface  = X_atm * pS(end);
% Cbulk_surface = X_atm * (pS(end) + n_air);
% %% =========================================================
% %% 8. Condition initiale (t = 0)
% %% =========================================================
% % CI cohérente avec BC surface : les deux utilisent X_lab(1) (échantillon le plus haut)
% % => à t=0, colonne uniforme au niveau atmosphérique (même valeur que la BC Dirichlet)
% Cbulk0 = Cbulk_surface * ones(nz,1);
% x0     = Cbulk0 ./ (pS + n_air);
% Cice0  = x0 .* pS;
% fprintf('\nIC = BC surface = %.4e mol/m³  (X_atm = %.2f ppm)\n', Cbulk_surface, X_atm*1e6);
% fprintf('IC partition check: max |C_bulk - (C_ice + x*n_air)| = %.3e mol/m³\n', ...
%     max(abs(Cbulk0 - (Cice0 + x0.*n_air))));
% %% =========================================================
% %% 9. Flux basal (z = 0)
% %% =========================================================
% % flux_bottom défini dans la section PARAMÈTRES
% %% =========================================================
% %% 10. Pré-allocation
% %% =========================================================
% C_bulk = zeros(nz, nt+1);  C_ice = zeros(nz, nt+1);
% X      = zeros(nz, nt+1);
% ADV    = zeros(nz, nt);    DIFF  = zeros(nz, nt);
% C_bulk(:,1) = Cbulk0;  C_ice(:,1) = Cice0;  X(:,1) = x0;
% %% =========================================================
% %% 11. Matrice de diffusion implicite (Backward Euler)
% %% =========================================================
% % Physique (Sandra) : seul C_ice diffuse (grain boundary diffusion).
% % C_bulk est le "grand réservoir" qui ne diffuse pas directement.
% % Après chaque pas de diffusion, l'équilibre thermodynamique C_ice = X*pS
% % est restauré via X = C_bulk / (pS + n_air)  (étape E).
% r      = D * dt / dz^2;
% A_diff = sparse(nz, nz);
% for i = 2:nz-1
%     A_diff(i,i-1) = -r; A_diff(i,i) = 1+2*r; A_diff(i,i+1) = -r;
% end
% % BC au fond (z = 0) — choix via BC_bottom_type
% switch BC_bottom_type
% case {'flux','zero_flux'}
% % Neumann (point fantôme) : ∂C_ice/∂z|_0 = -flux_bottom/D
%         A_diff(1,1) = 1+2*r; A_diff(1,2) = -2*r;
% case 'dirichlet'
% % Dirichlet : C_ice(z=0) = Cice_bottom_val
%         A_diff(1,1) = 1;
% otherwise
%         error('BC_bottom_type inconnu : %s', BC_bottom_type);
% end
% % BC en surface (z = Z) — Dirichlet (valeur atmosphérique imposée)
% A_diff(end,end) = 1;
% dA_diff = decomposition(A_diff);
% %% =========================================================
% %% 12. Boucle temporelle
% %% =========================================================
% u_face    = 0.5*(u(1:end-1) + u(2:end));
% u_face(1) = 0;
% % gas_loss est directement λ [yr⁻¹] — decay par pas de temps dt [yr]
% % Équation : dC/dt = -λ*C  =>  C(t+dt) = C(t)*exp(-λ*dt)
% decay  = exp(-gas_loss * dt);
% for k = 1:nt
%     %% -- A. ADVECT C_bulk (forme conservative, upwind) ---------------- %%
%     C_old    = C_bulk(:,k);
%     adv_flux = zeros(nz-1,1);
% for j = 2:nz-1
% if u_face(j) >= 0, C_up = C_old(j); else, C_up = C_old(j+1); end
%         adv_flux(j) = u_face(j) * C_up;
% end
%     adv_flux(nz-1) = u_face(nz-1) * Cbulk_surface;  % fix Sandra : BC surface imposée au flux upwind
%     C_bulk_adv = C_old;
% for i = 2:nz-1
%         C_bulk_adv(i) = C_old(i) + dt*(adv_flux(i-1) - adv_flux(i))/dz;
% end
%     C_bulk_adv(end) = Cbulk_surface;
%     ADV(:,k) = (C_bulk_adv - C_old)/dt;
%     %% -- B. PARTITION ------------------------------------------------- %%
%     x_adv    = C_bulk_adv ./ (pS + n_air);
%     Cice_adv = x_adv .* pS;
%     %% -- C. DIFFUSE C_ice (Backward Euler) ---------------------------- %%
%     B = Cice_adv;
% switch BC_bottom_type
% case 'flux'
%             B(1) = Cice_adv(1) + 2*flux_bottom*dt/dz;  % flux Neumann imposé
% case 'zero_flux'
%             B(1) = Cice_adv(1);                          % flux nul (homogène)
% case 'dirichlet'
%             B(1) = Cice_bottom_val;                       % valeur fixée
% end
%     B(end) = Cice_surface;
%     Cice_new = dA_diff \ B;
%     %% -- D. REBUILD C_bulk -------------------------------------------- %%
% % Seul l'incrément de la phase lattice est ajouté au réservoir total.
% % Les clathrates ne se re-remplissent PAS pendant l'étape de diffusion :
% % c'est le re-équilibre thermodynamique (étape E) qui redistribue ensuite.
%     C_bulk(:,k+1) = C_bulk_adv + (Cice_new - Cice_adv);
%     DIFF(:,k)     = (C_bulk(:,k+1) - C_bulk_adv) / dt;
%     %% -- E. RE-EQUILIBRATE -------------------------------------------- %%
%     X(:,k+1)     = C_bulk(:,k+1) ./ (pS + n_air);
%     C_ice(:,k+1) = X(:,k+1) .* pS;
%     %% -- F. GAS LOSS (phase clathrate uniquement) --------------------- %%
% if gas_loss > 0
% % Des clathrates se vident : X et C_ice restent inchangés,
% % seul le compartiment gazeux piégé diminue.
%         C_clath       = C_bulk(:,k+1) - C_ice(:,k+1);
%         C_bulk(:,k+1) = C_ice(:,k+1) + C_clath * decay;
% % X(:,k+1) et C_ice(:,k+1) ne sont PAS modifiés
% end
% end
% fprintf('Simulation complete.\n');
% verify_BC_conditions(z, dz, Z, pS, n_air*ones(nz,1), X_atm, dt);
% %% =========================================================
% %% 13. Retour aux unités labo
% %% =========================================================
% bulk_CC_model = C_bulk / rho_ice;
% %% =========================================================
% %% 14. Diagnostics
% %% =========================================================
% partition_err = C_bulk - (C_ice + X.*n_air);
% fprintf('Partition consistency: max error = %.3e mol/m³\n', max(abs(partition_err(:))));
% M_bulk = trapz(z, C_bulk);
% fprintf('Final total Ar inventory: %.6e mol/m²\n', M_bulk(end));
% %% =========================================================
% %% FIGURES 1–6  (diagnostics existants)
% %% =========================================================
% X_lab_grid = interp1(depth, X_lab/100, depth_model, 'linear','extrap');
% figure(1); clf;
% subplot(1,3,1);
%     plot(X(:,end)*1e6, z,'LineWidth',2); hold on;
%     plot(X_lab_grid*1e6, z,'o','MarkerSize',4);
%     xlabel('x (ppm)'); ylabel('Height above bed (m)');
%     title('Ar mixing ratio'); legend('Model','Lab','Location','best'); grid on; box on;
% subplot(1,3,2);
%     plot(C_ice(:,end), z,'LineWidth',2);
%     xlabel('C_{ice} (mol m^{-3})'); ylabel('Height above bed (m)');
%     title('Lattice Ar'); grid on; box on;
% subplot(1,3,3);
%     plot(C_bulk(:,end), z,'LineWidth',2); hold on;
%     plot(Cbulk_lab_profile, z,'o','MarkerSize',4);
%     xlabel('C_{bulk} (mol m^{-3})'); ylabel('Height above bed (m)');
%     title('Total Ar'); legend('Model','Lab','Location','best'); grid on; box on;
% sgtitle('Final state — thermodynamic partition');
% n_snap = 6; idx_k = round(linspace(1,nt+1,n_snap));
% labels = arrayfun(@(ii) sprintf('t = %.2g yr',t(idx_k(ii))),1:n_snap,'UniformOutput',false);
% figure(2); clf;
% subplot(1,3,1); hold on;
% for ii=1:n_snap; plot(X(:,idx_k(ii))*1e6, z,'LineWidth',1.8); end
%     xlabel('x (ppm)'); ylabel('Height above bed (m)'); title('x — mixing ratio'); grid on; box on;
% subplot(1,3,2); hold on;
% for ii=1:n_snap; plot(C_ice(:,idx_k(ii)), z,'LineWidth',1.8); end
%     xlabel('C_{ice} (mol m^{-3})'); ylabel('Height above bed (m)'); title('C_{ice} — lattice'); grid on; box on;
% subplot(1,3,3); hold on;
% for ii=1:n_snap; plot(C_bulk(:,idx_k(ii)), z,'LineWidth',1.8); end
%     xlabel('C_{bulk} (mol m^{-3})'); ylabel('Height above bed (m)'); title('C_{bulk} — total');
%     legend(labels,'Location','best','FontSize',8); grid on; box on;
% sgtitle('Temporal evolution');
% skip_t = max(1,floor(nt/200)); idx_plot = 1:skip_t:nt;
% t_plot = t(idx_plot); ADV_plot = ADV(:,idx_plot); DIFF_plot = DIFF(:,idx_plot);
% figure(3); clf;
% subplot(1,2,1); pcolor(t_plot,z,ADV_plot); shading flat; colorbar;
%     xlabel('Time (yr)'); ylabel('Height above bed (m)');
%     title('Advective tendency [mol m^{-3} yr^{-1}]'); grid on; box on;
% subplot(1,2,2); pcolor(t_plot,z,DIFF_plot); shading flat; colorbar;
%     xlabel('Time (yr)'); ylabel('Height above bed (m)');
%     title('Diffusive tendency [mol m^{-3} yr^{-1}]'); grid on; box on;
% sgtitle('Transport tendencies');
% figure(4); clf;
% subplot(1,3,1);
%     plot(G_lab,depth,'o-','LineWidth',1.5); set(gca,'YDir','reverse');
%     xlabel('Total gas content (mL/kg)'); ylabel('Depth from surface (m)');
%     title('Total gas content'); grid on; box on;
% subplot(1,3,2);
%     plot([TAC_val TAC_val],[depth_min depth_max],'r-','LineWidth',2);
%     set(gca,'YDir','reverse'); xlim([0 2*TAC_val]);
%     xlabel('TAC (m³ STP kg^{-1})'); ylabel('Depth from surface (m)');
%     title('TAC (constant — Sailer)'); grid on; box on;
% subplot(1,3,3);
%     plot(X_lab,depth,'o-','LineWidth',1.5); set(gca,'YDir','reverse');
%     xlabel('Ar mixing ratio (%)'); ylabel('Depth from surface (m)');
%     title('Ar mixing ratio'); grid on; box on;
% sgtitle('Camp Century — lab measurements');
% figure(5); clf;
% subplot(1,3,1);
%     plot(X_lab/100,depth,'ko','MarkerSize',5); hold on;
%     plot(X(:,end),depth_model,'LineWidth',2); set(gca,'YDir','reverse');
%     xlabel('x (mol/mol)'); ylabel('Depth from surface (m)');
%     title('Ar mixing ratio'); legend('Lab','Model','Location','best'); grid on; box on;
% subplot(1,3,2);
%     plot(bulk_CC,depth,'ko','MarkerSize',5); hold on;
%     plot(bulk_CC_model(:,end),depth_model,'LineWidth',2); set(gca,'YDir','reverse');
%     xlabel('C_{bulk} (mol kg^{-1})'); ylabel('Depth from surface (m)');
%     title('Total Ar'); legend('Lab','Model','Location','best'); grid on; box on;
% subplot(1,3,3);
%     plot(C_ice(:,end),depth_model,'LineWidth',2); set(gca,'YDir','reverse');
%     xlabel('C_{ice} (mol m^{-3})'); ylabel('Depth from surface (m)');
%     title('Lattice Ar (model only)'); grid on; box on;
% sgtitle('Camp Century — model vs lab');
% frac_lattice = C_ice(:,end) ./ C_bulk(:,end);
% frac_clath   = 1 - frac_lattice;
% figure(6); clf;
%     plot(frac_lattice*100, z,'LineWidth',2); hold on;
%     plot(frac_clath*100,   z,'LineWidth',2);
%     xlabel('Fraction of total Ar (%)'); ylabel('Height above bed (m)');
%     title('Ar partition — lattice vs clathrates (final state)');
%     legend('Lattice','Clathrates','Location','best'); grid on; box on;
% fprintf('\nFinal Ar partition:\n');
% fprintf('  Lattice:    %.4f – %.4f %%\n', 100*min(frac_lattice),100*max(frac_lattice));
% fprintf('  Clathrates: %.4f – %.4f %%\n', 100*min(frac_clath),  100*max(frac_clath));
% %% =========================================================
% %% FIGURE 9 — Construction progressive des processus (état final t = T)
% %%   Courbe 1 : Diffusion + Source basale              (Umax=0, gas_loss=0)
% %%   Courbe 2 : + Advection                            (Umax≠0, gas_loss=0)
% %%   Courbe 3 : + Gas loss = modèle complet            (Umax≠0, gas_loss≠0)
% %% NOTE : X et C_ice ne changent PAS avec gas_loss (clathrate phase uniquement)
% %%        → courbes 2 et 3 identiques pour X et C_ice ; C_bulk diverge
% %% =========================================================
% fprintf('\n=== FIGURE 9 : Construction progressive des processus ===\n');
% n_air_9 = n_air * ones(nz,1);
% sim9a = run_model_attr(0,     D_m2s, 0,        flux_bottom, ...
%             z, dz, Z, pS, n_air_9, X_atm, Cbulk0, T, dt);   % Diff + Source
% sim9b = run_model_attr(Umax,  D_m2s, 0,        flux_bottom, ...
%             z, dz, Z, pS, n_air_9, X_atm, Cbulk0, T, dt);   % + Advection
% sim9c = run_model_attr(Umax,  D_m2s, gas_loss, flux_bottom, ...
%             z, dz, Z, pS, n_air_9, X_atm, Cbulk0, T, dt);   % + Gas loss
% % Reconstruction des 3 variables
% % sim.X = vrai mixing ratio (inchangé par gas_loss) ; sim.C = C_bulk/rho_ice
% X9    = {sim9a.X,          sim9b.X,          sim9c.X};
% Cice9 = {sim9a.X .* pS,   sim9b.X .* pS,   sim9c.X .* pS};
% Cb9   = {sim9a.C*rho_ice,  sim9b.C*rho_ice,  sim9c.C*rho_ice};
% % Palette Okabe-Ito (CVD-safe)
% col9 = [0.000 0.447 0.698;   % bleu      — Diff + Source
%         0.902 0.624 0.000;   % orange    — + Advection
%         0.835 0.369 0.000];  % vermillon — + Gas loss
% lbl9 = {'Diff + Source', '+ Advection', '+ Gas loss (complet)'};
% lw9  = 2.2;
% figure(9); clf; set(gcf,'Position',[100 100 1200 500]);
% subplot(1,3,1); hold on;
%     plot(X_atm*1e6*ones(2,1), [0 Z],'k:','LineWidth',1,'DisplayName','CI (t=0)');
% for ii = 1:3
%         plot(X9{ii}*1e6, z, 'LineWidth',lw9, 'Color',col9(ii,:), 'DisplayName',lbl9{ii});
% end
%     xlabel('x (ppm)'); ylabel('Height above bed (m)');
%     title('x — mixing ratio');
%     legend('Location','best','FontSize',8); grid on; box on;
% subplot(1,3,2); hold on;
%     plot(X_atm*pS(1)*1e3*ones(2,1), [0 Z],'k:','LineWidth',1,'HandleVisibility','off');
% for ii = 1:3
%         plot(Cice9{ii}*1e3, z, 'LineWidth',lw9, 'Color',col9(ii,:), 'DisplayName',lbl9{ii});
% end
%     xlabel('C_{ice} (mol m^{-3}) \times 10^{-3}'); ylabel('Height above bed (m)');
%     title('C_{ice} — lattice (inchangée par gas loss)');
%     grid on; box on;
% subplot(1,3,3); hold on;
%     plot(Cbulk_surface*ones(2,1), [0 Z],'k:','LineWidth',1,'HandleVisibility','off');
% for ii = 1:3
%         plot(Cb9{ii}, z, 'LineWidth',lw9, 'Color',col9(ii,:), 'DisplayName',lbl9{ii});
% end
%     xlabel('C_{bulk} (mol m^{-3})'); ylabel('Height above bed (m)');
%     title('C_{bulk} — total (affectée par gas loss)');
%     grid on; box on;
% sgtitle(sprintf('Construction progressive des processus  (t = %.2g yr)', T), 'FontSize',13);
% fprintf('Figure 9 : OK\n');
% %% =========================================================
% %% FIGURE 7 — Importance des processus (leave-one-out)
% %% Valeurs de référence imposées — AUCUNE optimisation
% %% =========================================================
% fprintf('\n=== FIGURE 7 : Attribution (valeurs littérature) ===\n');
% % Vecteur de référence [Umax  D_m2s  gas_loss  flux_bottom]
% p_ref = [Umax, D_m2s, gas_loss, flux_bottom];
% % Paramètre "éteint" pour chaque processus
% NULL_VAL    = [0,     1e-15, 0,  0];
% PROC_NAMES  = {'U_{max}','D','gas_{loss}','flux_{bot}'};
% % RMSE avec tous les processus actifs
% n_air_vec = n_air * ones(nz,1);
% sim_all  = run_model_attr(p_ref(1),p_ref(2),p_ref(3),p_ref(4), ...
%                z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt);
% RMSE_all = calc_rmse_attr(sim_all.X, z, depth, X_lab);
% fprintf('RMSE (tous processus) : %.4e\n', RMSE_all);
% % Leave-one-out
% RMSE_loo = nan(4,1);
% for ip = 1:4
%     p_tmp     = p_ref;  p_tmp(ip) = NULL_VAL(ip);
%     sim_tmp   = run_model_attr(p_tmp(1),p_tmp(2),p_tmp(3),p_tmp(4), ...
%                     z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt);
%     RMSE_loo(ip) = calc_rmse_attr(sim_tmp.X, z, depth, X_lab);
%     fprintf('Sans %-12s : RMSE = %.4e  (Δ = +%.4e)\n', ...
%         PROC_NAMES{ip}, RMSE_loo(ip), RMSE_loo(ip)-RMSE_all);
% end
% % Bar chart
% delta = RMSE_loo - RMSE_all;
% [delta_sort, isort] = sort(delta,'descend');
% names_sort = PROC_NAMES(isort);
% CLR4 = [0.00 0.45 0.70; 0.90 0.62 0.00; 0.00 0.62 0.45; 0.80 0.40 0.00];
% clr_sort = CLR4(isort,:);
% figure(7); clf;
% ax7 = axes; hold(ax7,'on');
% for k = 1:4
%     bar(ax7,k,delta_sort(k),0.6,'FaceColor',clr_sort(k,:),'EdgeColor','none');
%     text(ax7,k,delta_sort(k)+max(delta_sort)*0.04, ...
%         sprintf('+%.2e',delta_sort(k)), ...
% 'HorizontalAlignment','center','FontSize',10,'FontWeight','bold', ...
% 'Color',clr_sort(k,:)*0.75);
% end
% set(ax7,'XTick',1:4,'XTickLabel',names_sort,'FontSize',12,'Box','off');
% ylabel(ax7,'\DeltaRMSE_X  (mol mol^{-1})','FontSize',11);
% title(ax7,{'Importance relative des processus de transport', ...
% '(\DeltaRMSE = augmentation quand le processus est éteint)'}, ...
% 'FontSize',11,'FontWeight','bold');
% ylim(ax7,[0, max(delta_sort)*1.20]);
% grid(ax7,'on'); ax7.GridAlpha = 0.2;
% saveas(figure(7),'figure7_importance.png');
% fprintf('Figure 7 sauvegardée : figure7_importance.png\n');
% %% =========================================================
% %% FIGURE 8 — Profils X : best fit optimisé (leave-one-out)
% %% =========================================================
% fprintf('\n=== FIGURE 8 : Profils X avec optimisation ===\n');
% opts_opt = optimset('MaxIter',5000,'TolFun',1e-10,'Display','off');
% idx_all  = [1 2 3 4];
% objfun_all = @(t_vec) obj_attr(t_vec, idx_all, p_ref, LB_opt, UB_opt, ...
%     z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt,depth,X_lab);
% best_r = inf; best_t = [];
% for ns = 1:N_STARTS_opt
%     t0 = rand_start_attr(idx_all, LB_opt, UB_opt);
% try
%         [t_s,r_s] = fminsearch(objfun_all,t0,opts_opt);
% if r_s < best_r; best_r = r_s; best_t = t_s; end
% catch; end
% end
% p_opt = p_ref;
% for k = 1:4; p_opt(k) = t2p_attr(best_t(k),k,LB_opt,UB_opt); end
% RMSE_opt = best_r;
% fprintf('Best fit : RMSE=%.3e  Umax=%.2e  D=%.2e  gas_loss=%.3f  flux_bot=%.2e\n', ...
%     RMSE_opt, p_opt(1),p_opt(2),p_opt(3),p_opt(4));
% % Profil best fit
% sim_opt  = run_model_attr(p_opt(1),p_opt(2),p_opt(3),p_opt(4), ...
%                z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt);
% X_opt    = sim_opt.X * 100;
% % Profils leave-one-out avec paramètres optimisés
% X_loo_opt = cell(4,1);
% RMSE_loo_opt = nan(4,1);
% for ip = 1:4
%     p_tmp     = p_opt; p_tmp(ip) = NULL_VAL(ip);
%     sim_tmp   = run_model_attr(p_tmp(1),p_tmp(2),p_tmp(3),p_tmp(4), ...
%                     z,dz,Z,pS,n_air_vec,X_atm,Cbulk0,T,dt);
%     X_loo_opt{ip}    = sim_tmp.X * 100;
%     RMSE_loo_opt(ip) = calc_rmse_attr(sim_tmp.X,z,depth,X_lab);
% end
% % Hauteur bande sédimentaire
% [~,imax_lab] = max(X_lab);
% hab_sed      = depth_max - depth(imax_lab);
% BROWN        = [0.55 0.27 0.07];
% LW = 2; MS = 6;
% all_X_fig8 = [X_lab; X_opt; cell2mat(cellfun(@(v)v,X_loo_opt,'UniformOutput',false))];
% xlim8 = [min(all_X_fig8)*0.97, max(all_X_fig8)*1.03];
% figure(8); clf;
% ax8 = axes; hold(ax8,'on');
% if hab_sed > 0
%     patch(ax8,xlim8([1 2 2 1]),[0 0 hab_sed hab_sed],BROWN, ...
% 'FaceAlpha',0.25,'EdgeColor','none');
% end
% plot(ax8,X_lab,depth_max-depth,'ko','MarkerSize',MS,'MarkerFaceColor','w', ...
% 'LineWidth',1.2,'DisplayName','Données labo');
% for ip = 1:4
%     plot(ax8,X_loo_opt{ip},z,'--','Color',CLR4(ip,:),'LineWidth',1.4, ...
% 'DisplayName',sprintf('sans %s',PROC_NAMES{ip}));
% end
% plot(ax8,X_opt,z,'-','Color',[0 0 0],'LineWidth',2.5, ...
% 'DisplayName',sprintf('Best fit (RMSE=%.2e)',RMSE_opt));
% xlabel(ax8,'Ar mixing ratio (%)','FontSize',11);
% ylabel(ax8,'Hauteur au-dessus du fond (m)','FontSize',11);
% title(ax8,{'Profils X — attribution par processus', ...
% 'Pointillés = best fit sans ce paramètre'},'FontSize',11,'FontWeight','bold');
% xlim(ax8,xlim8); ylim(ax8,[0 Z*1.05]);
% legend(ax8,'Location','northwest','FontSize',8,'Box','off');
% box(ax8,'off'); grid(ax8,'on'); ax8.GridAlpha = 0.2;
% saveas(figure(8),'figure8_profils.png');
% fprintf('Figure 8 sauvegardée : figure8_profils.png\n');
% %% =========================================================
% %% FONCTIONS LOCALES
% %% =========================================================
% function sim = run_model_attr(Umax, D_m2s, gas_loss, flux_bottom, ...
%         z, dz, Z, pS, n_air, X_atm, Cbulk0, T, dt)
% % Version vectorisée pour l'attribution (Figures 7, 8, 9)
% % Reconstruction C_bulk par incrément lattice (méthode Code 1) :
% %   C_bulk = C_bulk_adv + (Cice_new - Cice_adv)
% % sim.X = mixing ratio après re-équilibre (inchangé par gas_loss)
% % sim.C = C_bulk final / rho_ice (affecté par gas_loss)
%     rho_ice = 917;            % [kg/m³] — constante locale
%     D  = D_m2s * 3.1536e7;
%     nz = numel(z);
%     nt = round(T/dt);
%     u = Umax * ones(nz, 1);  %u      = Umax * (z/Z).^2;
%     u_face = 0.5*(u(1:end-1)+u(2:end)); u_face(1) = 0;
%     u_pos  = max(0,u_face);  u_neg = min(0,u_face);
%     Cice_surf = X_atm * pS(end);
%     Cb_surf   = X_atm * (pS(end) + n_air(end));
%     C_bulk    = Cbulk0;
%     r = D*dt/dz^2; e = ones(nz,1);
%     A = spdiags([-r*e,(1+2*r)*e,-r*e],[-1,0,1],nz,nz);
%     A(1,1) = 1+2*r; A(1,2) = -2*r;
%     A(end,:) = 0;   A(end,end) = 1;
%     dA = decomposition(A);
% % gas_loss est λ [yr⁻¹] : dC/dt = -λ*C => facteur exp(-λ*dt) par pas
%     decay_a = exp(-gas_loss * dt);
% for k = 1:nt
%         %% -- A. Advection (forme conservative, upwind) ----------- %%
%         F  = u_pos.*C_bulk(1:end-1) + u_neg.*C_bulk(2:end);
%         F(end) = u_face(end) * Cb_surf;       % fix Sandra : BC surface au flux upwind
%         Ca = C_bulk;
%         Ca(2:end-1) = C_bulk(2:end-1) + dt/dz*(F(1:end-1)-F(2:end));
%         Ca(end) = Cb_surf;
%         %% -- B. Partition ---------------------------------------- %%
%         xa  = Ca./(pS+n_air);  Cia = xa.*pS;
%         %% -- C. Diffusion C_ice ---------------------------------- %%
%         B = Cia; B(1) = Cia(1)+2*flux_bottom*dt/dz; B(end) = Cice_surf;
%         Cice_new_a = dA\B;
%         %% -- D. Rebuild C_bulk (lattice increment only) ---------- %%
%         C_bulk = Ca + (Cice_new_a - Cia);
%         %% -- E. Re-equilibrate ----------------------------------- %%
%         x_new_a = C_bulk ./ (pS + n_air);
%         %% -- F. Gas loss (phase clathrate uniquement) ------------ %%
% if gas_loss > 0
%             C_ice_a   = x_new_a .* pS;
%             C_clath_a = C_bulk - C_ice_a;
%             C_bulk    = C_ice_a + C_clath_a * decay_a;
% % x_new_a inchangé : X ne change pas avec gas_loss
% end
% end
%     sim.X = x_new_a;          % mixing ratio après re-équilibre (inchangé par gas_loss)
%     sim.C = C_bulk / rho_ice; % C_bulk final en unités labo
% end
% function rmse = calc_rmse_attr(X_model, z, depth, X_lab)
%     depth_max = max(depth);
%     X_interp  = interp1(depth_max-z, X_model, depth,'linear','extrap');
%     rmse      = sqrt(mean((X_interp - X_lab/100).^2));
% end
% function rmse = obj_attr(t_vec, idx_a, p_off, LB, UB, ...
%         z,dz,Z,pS,n_air,X_atm,Cbulk0,T,dt,depth,X_lab)
%     p = p_off;
% for k = 1:numel(idx_a)
%         p(idx_a(k)) = t2p_attr(t_vec(k),idx_a(k),LB,UB);
% end
% try
%         sim  = run_model_attr(p(1),p(2),p(3),p(4),z,dz,Z,pS,n_air,X_atm,Cbulk0,T,dt);
%         rmse = calc_rmse_attr(sim.X,z,depth,X_lab);
% catch
%         rmse = 1e6;
% end
% end
% function t0 = rand_start_attr(idx_a, LB, UB)
%     t0 = zeros(1,numel(idx_a));
% for k = 1:numel(idx_a)
%         ip = idx_a(k);
% switch ip
% case 1
%                 lo = min(log(abs(LB(1))),log(abs(UB(1))));
%                 hi = max(log(abs(LB(1))),log(abs(UB(1))));
%                 t0(k) = lo+(hi-lo)*rand();
% case {2,4}
%                 t0(k) = log(LB(ip))+(log(UB(ip))-log(LB(ip)))*rand();
% case 3
%                 t0(k) = LB(3)+(UB(3)-LB(3))*rand();
% end
% end
% end
% function p = t2p_attr(t, ip, LB, UB)
% switch ip
% case 1,     p = -exp(t); p = max(LB(1),min(UB(1),p));
% case {2,4}, p =  exp(t); p = max(LB(ip),min(UB(ip),p));
% case 3,     p = max(LB(3),min(UB(3),t));
% end
% end
% %% =========================================================
% %% VÉRIFICATION DES 3 CONDITIONS BASALES
% %% =========================================================
% %  Appel (après la boucle principale, avant les figures) :
% %    verify_BC_conditions(z, dz, Z, pS, n_air*ones(nz,1), X_atm, dt)
% %
% %  TEST 1 — 'zero_flux'  : CI=BC_surf, Umax=0, flux=0  → profil constant
% %  TEST 2 — 'dirichlet'  : CI=BC_surf, Umax=0, BC_bot=BC_surf → profil constant
% %  TEST 3 — 'flux'       : D=0, Umax=0, flux=F → ΔM_ice = F·T (conservation)
% %% =========================================================
% function verify_BC_conditions(z, dz, Z, pS, n_air_vec, X_atm, dt)
%     fprintf('\n========================================\n');
%     fprintf('  VÉRIFICATION DES 3 CONDITIONS BASALES\n');
%     fprintf('========================================\n');
%     nz       = numel(z);
%     D_test   = 1e-11 * 3.1536e7;  % [m²/yr]
%     T_test   = 1e4;                % durée rapide [yr]
%     nt_test  = round(T_test / dt);
%     Cice_surf = X_atm * pS(end);
%     Cb_surf   = X_atm * (pS(end) + n_air_vec(end));
%     Cb0       = Cb_surf * ones(nz, 1);  % CI uniforme = BC surface
%     all_pass = true;
%     %% ── TEST 1 : zero_flux ──────────────────────────────────────────
%     fprintf('\n[TEST 1]  BC_bottom = ''zero_flux''\n');
%     fprintf('  Config  : Umax=0, flux=0, CI=BC_surf (état d''équilibre)\n');
%     fprintf('  Attendu : max|ΔC_bulk| < 1e-10 mol/m³\n');
%     Cb1    = mini_sim(Cb0, 0, D_test, 0, 'zero_flux', Cice_surf, ...
%                       z, dz, Z, pS, n_air_vec, Cb_surf, Cice_surf, dt, nt_test);
%     drift1 = max(abs(Cb1 - Cb0));
%     pass1  = drift1 < 1e-10;
%     all_pass = all_pass & pass1;
%     fprintf('  Résultat: max|ΔC_bulk| = %.2e mol/m³  →  %s\n', drift1, ok(pass1));
%     %% ── TEST 2 : dirichlet ──────────────────────────────────────────
%     fprintf('\n[TEST 2]  BC_bottom = ''dirichlet''\n');
%     fprintf('  Config  : Umax=0, BC_bot=BC_surf=CI (équilibre symétrique)\n');
%     fprintf('  Attendu : max|ΔC_bulk| < 1e-10 mol/m³\n');
%     Cb2    = mini_sim(Cb0, 0, D_test, 0, 'dirichlet', Cice_surf, ...
%                       z, dz, Z, pS, n_air_vec, Cb_surf, Cice_surf, dt, nt_test);
%     drift2 = max(abs(Cb2 - Cb0));
%     pass2  = drift2 < 1e-10;
%     all_pass = all_pass & pass2;
%     fprintf('  Résultat: max|ΔC_bulk| = %.2e mol/m³  →  %s\n', drift2, ok(pass2));
%     %% ── TEST 3 : flux — conservation de masse ───────────────────────
%     F_test  = 1e-6;   % flux test [mol/m²/yr]
%     nt_mass = 50;
%     T_mass  = nt_mass * dt;
%     fprintf('\n[TEST 3]  BC_bottom = ''flux''\n');
%     fprintf('  Config  : Umax=0, D=0, flux=%.1e mol/m²/yr, T=%.0f yr\n', F_test, T_mass);
%     fprintf('  Attendu : ΔM_ice = %.6e mol/m²\n', F_test * T_mass);
%     Cb3 = mini_sim(Cb0, 0, 0, F_test, 'flux', 0, ...
%                    z, dz, Z, pS, n_air_vec, Cb_surf, Cice_surf, dt, nt_mass);
%     Cice0_v = Cb0  .* pS ./ (pS + n_air_vec);
%     Cice3   = Cb3  .* pS ./ (pS + n_air_vec);
%     dM_ice      = trapz(z, Cice3 - Cice0_v);
%     dM_expected = F_test * T_mass;
%     err3   = abs(dM_ice - dM_expected) / dM_expected;
%     pass3  = err3 < 1e-2;
%     all_pass = all_pass & pass3;
%     fprintf('  ΔM_ice  = %.6e mol/m²  (attendu %.6e)\n', dM_ice, dM_expected);
%     fprintf('  Erreur relative = %.2e  →  %s\n', err3, ok(pass3));
%     %% ── Bilan ───────────────────────────────────────────────────────
%     fprintf('\n----------------------------------------\n');
% if all_pass
%         fprintf('  BILAN : TOUS LES TESTS RÉUSSIS  ✓\n');
% else
%         fprintf('  BILAN : CERTAINS TESTS ONT ÉCHOUÉ  ✗\n');
%         fprintf('          → Vérifier la matrice A_diff (Section 11)\n');
% end
%     fprintf('========================================\n\n');
% end
% %% ──────────────────────────────────────────────────────────────────────────
% function Cb_out = mini_sim(Cb0, Umax, D, flux_bot, bc_type, Cice_bot_val, ...
%                             z, dz, Z, pS, n_air, Cb_surf, Cice_surf, dt, nt)
% %MINI_SIM  Boucle temporelle minimale pour les tests de vérification BC.
% %  Reconstruction C_bulk par incrément lattice (cohérence avec boucle principale).
%     nz        = numel(z);
%     u         = Umax * (z/Z).^2;
%     u_face    = 0.5*(u(1:end-1)+u(2:end));  u_face(1) = 0;
%     u_pos     = max(0, u_face);  u_neg = min(0, u_face);
%     r = D * dt / dz^2;
%     e = ones(nz,1);
%     A = spdiags([-r*e, (1+2*r)*e, -r*e], [-1,0,1], nz, nz);
% switch bc_type
% case {'flux','zero_flux'}
%             A(1,1) = 1+2*r;  A(1,2) = -2*r;   % Neumann (point fantôme)
% case 'dirichlet'
%             A(1,:) = 0;  A(1,1) = 1;           % Dirichlet
% end
%     A(end,:) = 0;  A(end,end) = 1;             % surface : Dirichlet
%     dA = decomposition(A);
%     Cb = Cb0;
% for k = 1:nt
%         %% -- A. Advection (forme conservative, upwind) ----------- %%
%         F = u_pos.*Cb(1:end-1) + u_neg.*Cb(2:end);
%         F(end) = u_face(end) * Cb_surf;
%         Ca = Cb;
%         Ca(2:end-1) = Cb(2:end-1) + dt/dz*(F(1:end-1)-F(2:end));
%         Ca(end) = Cb_surf;
%         %% -- B. Partition ---------------------------------------- %%
%         Cia = Ca ./ (pS + n_air) .* pS;
%         %% -- C. Diffusion C_ice ---------------------------------- %%
%         B = Cia;
% switch bc_type
% case 'flux',      B(1) = Cia(1) + 2*flux_bot*dt/dz;
% case 'zero_flux', B(1) = Cia(1);
% case 'dirichlet', B(1) = Cice_bot_val;
% end
%         B(end) = Cice_surf;
%         Ci_new = dA \ B;
%         %% -- D. Rebuild C_bulk (lattice increment only) ---------- %%
%         Cb = Ca + (Ci_new - Cia);
%         %% -- E. Re-equilibrate (implicite via Cb) ---------------- %%
% % x = Cb ./ (pS + n_air)  si besoin en dehors de la boucle
% end
%     Cb_out = Cb;
% end
% %% ──────────────────────────────────────────────────────────────────────────
% function s = ok(b)
% if b, s = 'PASS  ✓'; else, s = 'FAIL  ✗'; end
% end