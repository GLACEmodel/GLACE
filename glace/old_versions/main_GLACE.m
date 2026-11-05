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
%%   z = 0   -> bed                (MATLAB index 1)
%%   z = Z   -> surface of ice     (MATLAB index end)
%%
%% Operator splitting per timestep:
%%   A : advect C_bulk (upwind, advective form)
%%   B : partition -> C_ice, C_clath
%%   C : diffuse C_ice (implicit backward Euler)
%%   D : rebuild C_bulk from diffusion increment
%%   E : re-equilibrate x, C_ice
%%   F : gas loss — decay C_ice only, clathrates unchanged
%% =========================================================
clear; clc; close all;

%% =========================================================
%% 1. Vertical grid
%% =========================================================
Z  = 30;
nz = 101;
z  = linspace(0, Z, nz)';
dz = z(2) - z(1);

%% =========================================================
%% 2. Time
%% =========================================================
T  = 2e6;
dt = 100;
nt = round(T/dt);
t  = (0:nt)*dt;

%% =========================================================
%% 4. Physical parameters
%% =========================================================
rho_ice = 917;
Vm_STP  = 22.414e-3;
D_m2s = 1e-11;
D     = D_m2s * 3.1536e7;          % [m²/yr]
Umax = -1e-4;                       % [m/yr]  negative = downward
u    = Umax * (z/Z).^1;             % linear velocity profile
CFL_adv = max(abs(u)) * dt / dz;
SC_diff = D * dt / dz^2;
fprintf('CFL (advection) = %.3g   |   Diffusion number = %.3g\n', CFL_adv, SC_diff);
if CFL_adv > 1
    warning('Advection CFL > 1 — reduce dt or increase dz.');
end
P_diss = 5e6   * ones(nz,1);       % [Pa]
S_Ar   = 10e-9 * ones(nz,1);       % [mol/mol/Pa]
pS     = P_diss .* S_Ar;           % [mol/mol]

%% =========================================================
%% 5. Total air content
%% =========================================================
TAC   = 0.091e-3;                   % [m³ air / m³ ice]
n_air = TAC * rho_ice / Vm_STP;    % [mol air / m³ ice]

%% =========================================================
%% 5b. GAS LOSS CONFIGURATION
%% =========================================================
lambda_max  = 5e-6;   % [yr⁻¹]
z_char      = 8.0;    % [m]
z_gl_limit  = 20.0;   % [m]

%% =========================================================
%% 6. Boundary conditions
%% =========================================================
X_surf = 0.934e-2;    % atmospheric Ar mixing ratio [mol/mol]  = 0.934 %
X_bed  = 0.934e-2;

bc_surf.type  = 'dirichlet';
bc_surf.C     = [];
bc_surf.F     = [];
bc_bed.type   = 'flux';
bc_bed.C      = [];
bc_bed.F      = [];

bc_surf.C     = X_surf * (pS(end) + n_air); %%% pas de terme de volume dans ces équations ??
bc_surf.C_ice = X_surf * pS(end);

if strcmp(bc_bed.type, 'dirichlet')
    bc_bed.C     = X_bed * (pS(1) + n_air);
    bc_bed.C_ice = X_bed * pS(1);
elseif strcmp(bc_bed.type, 'flux')
    F_Ar_si  = 1e-12;
    bc_bed.F = F_Ar_si * 3.1536e7;
elseif strcmp(bc_bed.type, 'zeroflux')
    bc_bed.F = 0;
else
    bc_bed.C     = NaN;
    bc_bed.C_ice = NaN;
end

%% =========================================================
%% 7. Initial condition
%% =========================================================
Cbulk0 = X_surf * (pS + n_air);
x0     = Cbulk0 ./ (pS + n_air);
Cice0  = x0 .* pS;
fprintf('IC partition check: max |C_bulk - (C_ice + x.*n_air)| = %.3e mol/m³\n', ...
    max(abs(Cbulk0 - (Cice0 + x0.*n_air))));

%% =========================================================
%% 7b. Spatially-varying gas loss rate
%% =========================================================
lambda_z = zeros(nz,1);
mask     = (z < z_gl_limit);
lambda_z(mask) = lambda_max * exp(-z(mask) / z_char);
decay_factor = exp(-lambda_z * dt);
fprintf('Gas loss — lambda_max = %.1e yr⁻¹  |  half-life at bed = %.0f yr  |  z_char = %.1f m\n', ...
    lambda_max, log(2)/lambda_max, z_char);

%% =========================================================
%% 8. Pre-allocate storage
%% =========================================================
C_bulk = zeros(nz, nt+1);
C_ice  = zeros(nz, nt+1);
X      = zeros(nz, nt+1);
ADV    = zeros(nz, nt);
DIFF   = zeros(nz, nt);
C_bulk(:,1) = Cbulk0;
C_ice(:,1)  = Cice0;
X(:,1)      = x0;

%% =========================================================
%% 9. Build implicit diffusion matrix
%% =========================================================
r      = D * dt / dz^2;
A_diff = sparse(nz, nz);
for i = 2:nz-1
    A_diff(i, i-1) = -r;
    A_diff(i, i  ) =  1 + 2*r;
    A_diff(i, i+1) = -r;
end
switch bc_bed.type
    case 'dirichlet'
        A_diff(1,1) = 1;
    case {'flux','zeroflux'}
        A_diff(1,1) =  1 + 2*r;
        A_diff(1,2) =    - 2*r;
end
switch bc_surf.type
    case 'dirichlet'
        A_diff(end,end) = 1;
    case {'flux','zeroflux'}
        A_diff(end,end  ) =  1 + 2*r;
        A_diff(end,end-1) =    - 2*r;
end
try
    dA_diff    = decomposition(A_diff);
    use_decomp = true;
catch
    use_decomp = false;
    fprintf('decomposition() unavailable — using A\\b directly.\n');
end

%% =========================================================
%% 10. Time loop
%% =========================================================
for k = 1:nt
    %% -- A. ADVECT C_bulk -------------------------------------------- %%
    C_old = C_bulk(:,k);
    dCdz  = zeros(nz,1);
    for i = 2:nz-1
        if u(i) >= 0
            dCdz(i) = (C_old(i)   - C_old(i-1)) / dz;
        else
            dCdz(i) = (C_old(i+1) - C_old(i)  ) / dz;
        end
    end
    C_bulk_adv = C_old - dt * u .* dCdz;
    switch bc_bed.type
        case 'dirichlet',         C_bulk_adv(1)   = bc_bed.C;
        case {'flux','zeroflux'}, C_bulk_adv(1)   = C_old(1);
    end
    switch bc_surf.type
        case 'dirichlet',         C_bulk_adv(end) = bc_surf.C;
        case {'flux','zeroflux'}
            C_bulk_adv(end) = C_old(end) - dt*u(end)*(C_old(end)-C_old(end-1))/dz;
    end
    ADV(:,k) = (C_bulk_adv - C_old) / dt;

    %% -- B. PARTITION ------------------------------------------------- %%
    x_adv    = C_bulk_adv ./ (pS + n_air);
    Cice_adv = x_adv .* pS;

    %% -- C. DIFFUSE C_ice --------------------------------------------- %%
    B = Cice_adv;
    switch bc_bed.type
        case 'dirichlet', B(1) = bc_bed.C_ice;
        case 'flux',      B(1) = Cice_adv(1) + 2*bc_bed.F*dt/dz;
        case 'zeroflux',  B(1) = Cice_adv(1);
    end
    switch bc_surf.type
        case 'dirichlet', B(end) = bc_surf.C_ice;
        case 'flux',      B(end) = Cice_adv(end) - 2*bc_surf.F*dt/dz;
        case 'zeroflux',  B(end) = Cice_adv(end);
    end
    if use_decomp
        Cice_new = dA_diff \ B;
    else
        Cice_new = A_diff \ B;
    end

    %% -- D. REBUILD C_bulk -------------------------------------------- %%
    C_bulk(:,k+1) = C_bulk_adv + (Cice_new - Cice_adv);
    DIFF(:,k)     = (C_bulk(:,k+1) - C_bulk_adv) / dt;

    %% -- E. RE-EQUILIBRATE -------------------------------------------- %%
    X(:,k+1)     = C_bulk(:,k+1) ./ (pS + n_air);
    C_ice(:,k+1) = X(:,k+1) .* pS;

    %% -- F. GAS LOSS -------------------------------------------------- %%
    C_clath_k     = C_bulk(:,k+1) - C_ice(:,k+1);
    C_ice(:,k+1)  = C_ice(:,k+1) .* decay_factor;
    C_bulk(:,k+1) = C_ice(:,k+1) + C_clath_k;
    X(:,k+1)      = C_bulk(:,k+1) ./ (pS + n_air);
end
fprintf('Simulation complete.\n');

%% =========================================================
%% 11. Convert to lab units
%% =========================================================
bulk_CC_model = C_bulk / rho_ice;

%% =========================================================
%% 12. Diagnostics
%% =========================================================
partition_err = C_bulk - (C_ice + X.*n_air);
fprintf('Partition consistency: max error = %.3e mol/m³\n', max(abs(partition_err(:))));
M_bulk = trapz(z, C_bulk);
fprintf('Final total Ar inventory: %.6e mol/m²\n', M_bulk(end));

%% =========================================================
%% FIGURE 1 — Final thermodynamic partition
%% =========================================================
figure(1); clf;
subplot(1,3,1);
    plot(X(:,end)*100, z, 'LineWidth',2);
    xlabel('x (%)'); ylabel('Hauteur au-dessus du lit (m)');
    title('Fraction molaire Ar  x'); grid on; box on;
subplot(1,3,2);
    plot(C_ice(:,end), z, 'LineWidth',2);
    xlabel('C_{ice} (mol m^{-3})'); ylabel('Hauteur au-dessus du lit (m)');
    title('Ar en réseau cristallin  C_{ice}'); grid on; box on;
subplot(1,3,3);
    plot(C_bulk(:,end), z, 'LineWidth',2);
    xlabel('C_{bulk} (mol m^{-3})'); ylabel('Hauteur au-dessus du lit (m)');
    title('Ar total  C_{bulk}'); grid on; box on;
sgtitle('État final — partition thermodynamique');

%% =========================================================
%% FIGURE 2 — Temporal evolution (6 snapshots)
%% =========================================================
n_snap = 6;
idx_k  = round(linspace(1, nt+1, n_snap));
labels = arrayfun(@(ii) sprintf('t = %.2g yr', t(idx_k(ii))), 1:n_snap, 'UniformOutput',false);
figure(2); clf;
subplot(1,3,1); hold on;
for ii=1:n_snap; plot(X(:,idx_k(ii))*100, z, 'LineWidth',1.8); end
    xlabel('x (%)'); ylabel('Hauteur au-dessus du lit (m)'); title('x — fraction molaire');
    legend(labels,'Location','best','FontSize',8); grid on; box on;
subplot(1,3,2); hold on;
for ii=1:n_snap; plot(C_ice(:,idx_k(ii)), z, 'LineWidth',1.8); end
    xlabel('C_{ice} (mol m^{-3})'); ylabel('Hauteur au-dessus du lit (m)');
    title('C_{ice} — réseau'); grid on; box on;
subplot(1,3,3); hold on;
for ii=1:n_snap; plot(C_bulk(:,idx_k(ii)), z, 'LineWidth',1.8); end
    xlabel('C_{bulk} (mol m^{-3})'); ylabel('Hauteur au-dessus du lit (m)');
    title('C_{bulk} — total'); grid on; box on;
sgtitle('Évolution temporelle');

%% =========================================================
%% FIGURE 3 — Transport tendencies
%% =========================================================
skip_t   = max(1, floor(nt/200));
idx_plot = 1:skip_t:nt;
t_plot   = t(idx_plot);
n_cmap   = 256;
r_c = [linspace(0,1,n_cmap/2), ones(1,n_cmap/2)];
g_c = [linspace(0,1,n_cmap/2), linspace(1,0,n_cmap/2)];
b_c = [ones(1,n_cmap/2),       linspace(1,0,n_cmap/2)];
bwr = [r_c', g_c', b_c'];
figure(3); clf;
subplot(1,2,1);
    pcolor(t_plot, z, ADV(:,idx_plot)); shading flat; colorbar;
    adv_max = max(abs(ADV(:)));
    if adv_max > 0, clim([-1 1]*adv_max); end
    colormap(gca, bwr);
    xlabel('Temps (yr)'); ylabel('Hauteur au-dessus du lit (m)');
    title('Tendance advective  [mol m^{-3} yr^{-1}]');
    grid on; box on;
subplot(1,2,2);
    pcolor(t_plot, z, DIFF(:,idx_plot)); shading flat; colorbar;
    diff_max = max(abs(DIFF(:)));
    if diff_max > 0, clim([-1 1]*diff_max); end
    colormap(gca, bwr);
    xlabel('Temps (yr)'); ylabel('Hauteur au-dessus du lit (m)');
    title('Tendance diffusive  [mol m^{-3} yr^{-1}]');
    grid on; box on;

%% =========================================================
%% FIGURE 4 — Ar partition fractions
%% =========================================================
frac_lattice = C_ice(:,end) ./ C_bulk(:,end);
frac_clath   = 1 - frac_lattice;
figure(4); clf;
    plot(frac_lattice*100, z, 'LineWidth',2); hold on;
    plot(frac_clath*100,   z, 'LineWidth',2);
    xlabel('Fraction de l''Ar total (%)'); ylabel('Hauteur au-dessus du lit (m)');
    title('Partition Ar — réseau vs clathrates  (état final)');
    legend('Réseau cristallin','Clathrates','Location','best');
    grid on; box on;

%% =========================================================
%% FIGURE 9 — POSTER: Sensitivity analysis C_bulk
%%            Temporal evolution on each panel
%% =========================================================
SENS_adv  = [1e-2,  1,  1e+2];   % multipliers on Umax   (rows)
SENS_diff = [1e-2,  1,  1e+2];   % multipliers on D_m2s  (columns)
n_sub     = numel(SENS_adv);

%% -- Actual parameter values for labels --------------------
U_vals    = Umax  .* SENS_adv;               % [m yr⁻¹]
D_vals_yr = D_m2s .* SENS_diff * 3.1536e7;  % [m² yr⁻¹] — display unit

%% -- Column labels (D, one per column) ---------------------
adv_str  = { sprintf('U = %.2e m yr^{-1}', U_vals(1)), ...
             sprintf('U = %.2e m yr^{-1}', U_vals(2)), ...
             sprintf('U = %.2e m yr^{-1}', U_vals(3)) };

%% -- Row labels (U, one per row) ---------------------------
diff_str = { sprintf('D = %.2e m^{2} yr^{-1}', D_vals_yr(1)), ...
             sprintf('D = %.2e m^{2} yr^{-1}', D_vals_yr(2)), ...
             sprintf('D = %.2e m^{2} yr^{-1}', D_vals_yr(3)) };

%% -- Snapshots shared by all panels ------------------------
n_snap9   = 5;
idx_snap9 = round(linspace(1, nt, n_snap9));
t_snap9   = t(idx_snap9);
cmap9     = [linspace(0.85,0.08,n_snap9)', linspace(0.92,0.38,n_snap9)', ones(n_snap9,1)];  % light→dark blue gradient

%% -- Layout constants --------------------------------------
CLR_text    = [0 0 0];
LW_snap     = 1.8;
FS_sup      = 18;  FS_label = 13;  FS_title = 10;  FS_tick = 9;
x_lim_fixed = 15;

%% -- Run 9 simulations -------------------------------------
fprintf('\n=== Running 9 sensitivity simulations ===\n');
Cb_grid = cell(n_sub, n_sub);
for ir = 1:n_sub
    for ic = 1:n_sub
        Umax_s = Umax  * SENS_adv(ir);
        Dm2s_s = D_m2s * SENS_diff(ic);
        fprintf('  (%d,%d) U=%.1e m/yr  D=%.1e m²/s ... ', ir, ic, Umax_s, Dm2s_s);
        Cb_grid{ir,ic} = run_sens_code1(Umax_s, Dm2s_s, z, dz, Z, pS, n_air, ...
                                         bc_surf, bc_bed, Cbulk0, nt, dt, ...
                                         lambda_z, false, idx_snap9);
        fprintf('OK\n');
    end
end

%% -- Figure ------------------------------------------------
fig9 = figure(9); clf;
set(fig9, 'Color','w', 'Position',[50 50 1050 1000]);
tl = tiledlayout(4, n_sub, 'TileSpacing','compact', 'Padding','compact');
title(tl, 'Sensitivity analysis — C_{bulk} profiles', ...
    'FontSize',FS_sup,'FontWeight','bold','Color',CLR_text);

ax_all = gobjects(n_sub, n_sub);

for ir = 1:n_sub
    for ic = 1:n_sub
        ax            = nexttile((ir-1)*n_sub + ic);
        ax_all(ir,ic) = ax;
        hold(ax,'on');

        %% -- Temporal profiles ----------------------------------
        Cb_snaps = Cb_grid{ir,ic};
        for is = 1:n_snap9
            plot(ax, Cb_snaps(:,is), z, '-', 'Color',cmap9(is,:), 'LineWidth',LW_snap);
        end

        %% -- Axes formatting ------------------------------------
        if ic > 1, set(ax,'YTickLabel',{}); end
        if ir < n_sub, set(ax,'XTickLabel',{}); end
        xlim(ax, [0  x_lim_fixed]);
        ylim(ax, [0  Z*1.05]);
        set(ax,'FontSize',FS_tick,'FontWeight','bold');
        grid(ax,'on');  ax.GridAlpha = 0.15;  box(ax,'on');

        %% -- x-label on bottom row only -------------------------
        if ir == n_sub
            xlabel(ax, 'C_{bulk}  (mol m^{-3})', ...
                'FontSize',FS_label,'FontWeight','bold','Interpreter','tex');
        end

        %% -- y-label on left column only ------------------------
        if ic == 1
            ylabel(ax, 'Height above bed  z  (m)', ...
                'FontSize',FS_label,'FontWeight','bold');
        end
    end
end

%% -- Column headers (D) — placed AFTER loop using stored handles,
%    so each title is correctly centred on its column
for ic = 1:n_sub
    title(ax_all(1, ic), diff_str{ic}, ...
        'FontSize',FS_title+1, 'FontWeight','bold', ...
        'Color',CLR_text,      'Interpreter','tex');
end

%% -- Row labels (U) — placed AFTER the loop so ax_all positions are set.
%    Each label is centred on its row by anchoring to (1.04, 0.5) in the
%    normalised coordinates of the rightmost panel of that row.
for ir = 1:n_sub
    text(ax_all(ir, n_sub), 1.04, 0.5, adv_str{ir}, ...
        'Units',               'normalized', ...
        'HorizontalAlignment', 'left',       ...
        'VerticalAlignment',   'middle',     ...
        'FontSize',            FS_title,     ...
        'FontWeight',          'bold',       ...
        'Color',               CLR_text,     ...
        'Interpreter',         'tex',        ...
        'Rotation',            90,           ...
        'Clipping',            'off');
end

%% -- Directional arrows (figure-normalised coordinates) ----
%    Horizontal: D increases left → right
%    Vertical:   |U| increases top → bottom (more compaction downward)
%    ⚠ Adjust [x1 x2] / [y1 y2] if layout changes.
annotation(fig9,'doublearrow',[0.08 0.85],[0.962 0.962], ...
    'LineWidth',1.4,'Color',[0.4 0.4 0.4], ...
    'Head1Length',7,'Head2Length',7,'Head1Width',4,'Head2Width',4);
annotation(fig9,'doublearrow',[0.972 0.972],[0.13 0.90], ...
    'LineWidth',1.4,'Color',[0.4 0.4 0.4], ...
    'Head1Length',7,'Head2Length',7,'Head1Width',4,'Head2Width',4);

% Flèche verticale — |U| croissant de haut en bas (compaction croissante)
annotation(fig9,'doublearrow',[0.975 0.975],[0.14 0.91], ...
    'LineWidth',1.4,'Color',[0.35 0.35 0.35], ...
    'Head1Length',8,'Head2Length',8,'Head1Width',5,'Head2Width',5);

%% -- Légende commune — 4ème rangée, largeur des 3 colonnes de panneaux -
ax_leg = nexttile(tl, n_sub*n_sub+1, [1 n_sub]);
axis(ax_leg, 'off');
hold(ax_leg, 'on');
for is = 1:n_snap9
    plot(ax_leg, NaN, NaN, '-', 'Color',cmap9(is,:), 'LineWidth',3.0, ...
        'DisplayName', sprintf('t = %.2g yr', t_snap9(is)));
end
lgd = legend(ax_leg, 'Orientation','horizontal', 'FontSize',11, ...
    'Box','off', 'Location','north');
% Aligner la légende horizontalement sur la largeur des panneaux
lgd.Position(1) = ax_all(1,1).Position(1);               % bord gauche = panneau (1,1)
lgd.Position(3) = ax_all(1,n_sub).Position(1) + ...
                  ax_all(1,n_sub).Position(3) - ...
                  ax_all(1,1).Position(1);                % largeur = 3 colonnes
hold(ax_leg, 'off');
fprintf('Figure 9 (poster) : OK\n');

%% =========================================================
%% FIGURE 10 — Attribution des processus (3 courbes)
%% =========================================================
fprintf('\nCas 1 : Diffusion + flux basal seulement (Umax=0, sans gas loss) ... ');
Cb_1 = run_sens_code1(0,    D_m2s, z, dz, Z, pS, n_air, bc_surf, bc_bed, ...
                       Cbulk0, nt, dt, lambda_z, false);
fprintf('OK\n');
fprintf('Cas 2 : + Advection  (sans gas loss) ... ');
Cb_2 = run_sens_code1(Umax, D_m2s, z, dz, Z, pS, n_air, bc_surf, bc_bed, ...
                       Cbulk0, nt, dt, lambda_z, false);
fprintf('OK\n');
fprintf('Cas 3 : + Gas loss  (modèle complet) ... ');
Cb_3 = run_sens_code1(Umax, D_m2s, z, dz, Z, pS, n_air, bc_surf, bc_bed, ...
                       Cbulk0, nt, dt, lambda_z, true);
fprintf('OK\n');

X_1 = Cb_1./(pS+n_air);  Cice_1 = X_1.*pS;
X_2 = Cb_2./(pS+n_air);  Cice_2 = X_2.*pS;
X_3 = Cb_3./(pS+n_air);  Cice_3 = X_3.*pS;

CLR1 = [0.08 0.30 0.65];
CLR2 = [0.84 0.15 0.16];
CLR3 = [0.13 0.60 0.25];
LW10 = 3.5;
FS_sup10 = 22;  FS_lab10 = 17;  FS_tit10 = 16;  FS_tck10 = 14;  FS_leg10 = 13;
LBL1 = 'Diffusion + flux basal';
LBL2 = '+ Advection';
LBL3 = '+ Gas loss  (modèle complet)';

figure(10); clf;
set(gcf,'Color','w','Position',[50 50 1350 550]);
subplot(1,3,1); hold on;
plot(X_1*100, z, '-',  'Color',CLR1, 'LineWidth',LW10, 'DisplayName',LBL1);
plot(X_2*100, z, '--', 'Color',CLR2, 'LineWidth',LW10, 'DisplayName',LBL2);
plot(X_3*100, z, ':',  'Color',CLR3, 'LineWidth',LW10, 'DisplayName',LBL3);
xlabel('x  (%)',                             'FontSize',FS_lab10,'FontWeight','bold');
ylabel('Hauteur au-dessus du lit  z  (m)',   'FontSize',FS_lab10,'FontWeight','bold');
title('Fraction molaire Ar  x',              'FontSize',FS_tit10,'FontWeight','bold');
legend('Location','best','FontSize',FS_leg10,'Box','off');
set(gca,'FontSize',FS_tck10); grid on; box on;

subplot(1,3,2); hold on;
plot(Cice_1, z, '-',  'Color',CLR1, 'LineWidth',LW10, 'HandleVisibility','off');
plot(Cice_2, z, '--', 'Color',CLR2, 'LineWidth',LW10, 'HandleVisibility','off');
plot(Cice_3, z, ':',  'Color',CLR3, 'LineWidth',LW10, 'HandleVisibility','off');
xlabel('C_{ice}  (mol m^{-3})',  'FontSize',FS_lab10,'FontWeight','bold');
title('Ar réseau  C_{ice}',      'FontSize',FS_tit10,'FontWeight','bold');
set(gca,'FontSize',FS_tck10,'YTickLabel',{}); grid on; box on;

subplot(1,3,3); hold on;
plot(Cb_1, z, '-',  'Color',CLR1, 'LineWidth',LW10, 'HandleVisibility','off');
plot(Cb_2, z, '--', 'Color',CLR2, 'LineWidth',LW10, 'HandleVisibility','off');
plot(Cb_3, z, ':',  'Color',CLR3, 'LineWidth',LW10, 'HandleVisibility','off');
xlabel('C_{bulk}  (mol m^{-3})', 'FontSize',FS_lab10,'FontWeight','bold');
title('Ar total  C_{bulk}',      'FontSize',FS_tit10,'FontWeight','bold');
set(gca,'FontSize',FS_tck10,'YTickLabel',{}); grid on; box on;

sgtitle(sprintf('Attribution des processus  —  t = %.2g yr', T), ...
    'FontSize',FS_sup10,'FontWeight','bold');
fprintf('Figure 10 (attribution des processus) : OK\n');

%% =========================================================
%% Fin du script principal
%% =========================================================


%% =========================================================
%% FONCTION LOCALE — run_sens_code1
%% =========================================================
function Cb_out = run_sens_code1(Umax_s, D_m2s_s, z, dz, Z, pS, n_air, ...
                                  bc_surf, bc_bed, Cbulk0, nt, dt, ...
                                  lambda_z, apply_gas_loss, idx_snap)
% Lance la simulation avec Umax et D_m2s modifiés.
%   idx_snap (optionnel) : [1×n_snap] indices dans 1…nt+1
%                          Absent → retourne uniquement l'état final [nz×1]
%   apply_gas_loss       : logical

    %% -- Gestion des snapshots ----------------------------
    if nargin < 15 || isempty(idx_snap)
        idx_snap          = nt + 1;
        return_final_only = true;
    else
        return_final_only = false;
    end
    n_save = numel(idx_snap);
    nz     = numel(z);
    Cb_out = zeros(nz, n_save);

    %% -- Paramètres dérivés ------------------------------
    D_s = D_m2s_s * 3.1536e7;          % [m²/yr]
    u_s = Umax_s * (z / Z).^1;         % profil linéaire — identique au script principal

    %% -- Sous-cyclage adaptatif pour la stabilité CFL ----
    CFL_s = max(abs(u_s)) * dt / dz;
    n_cyc = max(1, ceil(CFL_s * 1.05)); % marge de sécurité 5 %
    dt_a  = dt / n_cyc;
    if CFL_s > 1
        fprintf('[sous-cyclage : CFL=%.2f → %d sous-pas, dt_a=%.1f yr] ', ...
                CFL_s, n_cyc, dt_a);
    end

    %% -- Matrice de diffusion implicite (pas complet dt) -
    r_s = D_s * dt / dz^2;
    A_s = sparse(nz, nz);
    for i = 2:nz-1
        A_s(i,i-1) = -r_s;  A_s(i,i) = 1+2*r_s;  A_s(i,i+1) = -r_s;
    end
    switch bc_bed.type
        case 'dirichlet',         A_s(1,1) = 1;
        case {'flux','zeroflux'}, A_s(1,1) = 1+2*r_s;  A_s(1,2) = -2*r_s;
    end
    switch bc_surf.type
        case 'dirichlet',         A_s(end,end) = 1;
        case {'flux','zeroflux'}, A_s(end,end) = 1+2*r_s;  A_s(end,end-1) = -2*r_s;
    end
    try
        dA_s   = decomposition(A_s);  use_dc = true;
    catch
        use_dc = false;
    end

    decay_s  = exp(-lambda_z * dt);
    C_bulk   = Cbulk0;
    save_col = 1;

    %% -- Boucle temporelle --------------------------------
    for k = 1:nt

        if save_col <= n_save && idx_snap(save_col) == k
            Cb_out(:, save_col) = C_bulk;
            save_col = save_col + 1;
        end

        %% A. Advection — n_cyc sous-pas (chaque CFL ≤ 1)
        C_adv = C_bulk;
        for isc = 1:n_cyc
            C_old  = C_adv;
            u_pos  = (u_s >= 0);
            dCdz_s = zeros(nz,1);
            dCdz_s(2:end-1) = ...
                u_pos(2:end-1)    .* (C_old(2:end-1) - C_old(1:end-2)) / dz + ...
                (~u_pos(2:end-1)) .* (C_old(3:end)   - C_old(2:end-1)) / dz;
            C_adv = C_old - dt_a * u_s .* dCdz_s;
            switch bc_bed.type
                case 'dirichlet',         C_adv(1)   = bc_bed.C;
                case {'flux','zeroflux'}, C_adv(1)   = C_old(1);
            end
            switch bc_surf.type
                case 'dirichlet',         C_adv(end) = bc_surf.C;
                case {'flux','zeroflux'}, C_adv(end) = C_old(end) ...
                                          - dt_a*u_s(end)*(C_old(end)-C_old(end-1))/dz;
            end
        end

        %% B. Partition
        x_adv    = C_adv ./ (pS + n_air);
        Cice_adv = x_adv .* pS;

        %% C. Diffusion C_ice
        B = Cice_adv;
        switch bc_bed.type
            case 'dirichlet', B(1) = bc_bed.C_ice;
            case 'flux',      B(1) = Cice_adv(1) + 2*bc_bed.F*dt/dz;
            case 'zeroflux',  B(1) = Cice_adv(1);
        end
        switch bc_surf.type
            case 'dirichlet', B(end) = bc_surf.C_ice;
            case 'flux',      B(end) = Cice_adv(end) - 2*bc_surf.F*dt/dz;
            case 'zeroflux',  B(end) = Cice_adv(end);
        end
        if use_dc,  Cice_new = dA_s \ B;
        else,       Cice_new = A_s  \ B;
        end

        %% D. Reconstruction de C_bulk
        C_bulk = C_adv + (Cice_new - Cice_adv);

        %% E. Ré-équilibrage
        X_k    = C_bulk ./ (pS + n_air);
        Cice_k = X_k .* pS;

        %% F. Gas loss — C_ice uniquement, clathrates inchangés
        if apply_gas_loss
            C_clath_k = C_bulk - Cice_k;
            Cice_k    = Cice_k .* decay_s;
            C_bulk    = Cice_k + C_clath_k;
        end
    end

    while save_col <= n_save
        Cb_out(:, save_col) = C_bulk;
        save_col = save_col + 1;
    end

    if return_final_only
        Cb_out = Cb_out(:,1);
    end
end