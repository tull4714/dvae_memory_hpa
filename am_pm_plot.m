%% AM/AM, AM/PM 특성 그리기 (메모리 다항식 HPA, 16QAM OFDM)
% 필요 파일: hpa_mp.m (같은 폴더)
%
% [선택] DPD 적용 후 특성도 그리려면, 파이썬 평가 파일 끝에 아래를 추가해
%        dpd_vtae.mat / dpd_trans.mat 을 만든 뒤 이 폴더에 두세요.
%   from scipy.io import savemat
%   n = 1_000_000   # 전체(2,560만 샘플)는 너무 크므로 앞부분만 저장
%   savemat('dpd_vtae.mat', {'x_ref': IQ_out_r[:n].flatten(), 'z': dvae_r[:n].flatten()})
%   (Transformer 평가 파일에서는 파일명을 'dpd_trans.mat'로)

clear; close all; rng(1);

%% 설정
N      = 64;          % 부반송파 수
OS     = 4;           % 오버샘플링 (Fs/Fd)
Nfft   = N*OS;
Nsym   = 2000;        % OFDM 심볼 수
IBO_dB = 8;           % 입력 백오프 [dB] (포화 입력 전력 기준, 파이썬 PA_IBO_DB와 같게)
model  = 'standard';  % 'standard': 논문 식 (2) (현재 파이썬 코드와 동일), 'code': 이전 코드의 변형 모델
Nplot  = 20000;       % 산점도에 찍을 샘플 수

%% 메모리 다항식 계수: Ding et al. (2004) Class-AB PA (Table I)
%  hpa_mp 안에서 c10으로 나누어 소신호 이득을 1로 정규화함
%        q = 0                q = 1                q = 2
C = [  1.0513+0.0904j,     -0.0680-0.0023j,      0.0289-0.0054j;    % k = 1
      -0.0542-0.2900j,      0.2234+0.2317j,     -0.0621-0.0932j;    % k = 3
      -0.9657-0.7028j,     -0.2451-0.3735j,      0.1229+0.1508j];   % k = 5

%% 16QAM OFDM 신호 생성
lv = [-3 -1 1 3];
X  = (lv(randi(4,N,Nsym)) + 1j*lv(randi(4,N,Nsym))) / sqrt(10);
Xp = zeros(Nfft, Nsym);
Xp(1:N/2,:)         = X(1:N/2,:);        % 가운데에 0을 넣어 오버샘플링
Xp(end-N/2+1:end,:) = X(N/2+1:end,:);
xt = ifft(Xp);
x  = xt(:);   % MATLAB의 (:)는 열 우선 → 심볼 단위의 물리적 전송 순서

%% HPA 통과
[y, A_sat] = hpa_mp(x, C, IBO_dB, model);
s   = sqrt(mean(abs(x).^2) * 10^(IBO_dB/10)) / A_sat;   % PA 입력의 정규화 스케일 (hpa_mp와 동일)
fprintf('HPA only EVM (%s model): %.2f %%\n', model, norm(y-x)/norm(x)*100);

idx  = randperm(numel(x), min(Nplot, numel(x)));
ain  = abs(x(idx)) / s;                                % 정규화 입력 진폭
aout = abs(y(idx)) / s;                                % 정규화 출력 진폭
dphi = angle(y(idx) .* conj(x(idx))) * 180/pi;         % 위상 변화 [deg]

ag = linspace(0, max(ain), 300).';                     % 메모리 없는(q=0) 기준 곡선
Cn = C / C(1,1);
agc = min(ag, A_sat);                                  % 진폭 제한기와 같게
F0 = Cn(1,1)*agc + Cn(2,1)*agc.^3 + Cn(3,1)*agc.^5;   % 포화 진폭 이상에서는 출력이 일정

%% Fig. 1: HPA 특성 (메모리 효과)
figure('Units','centimeters','Position',[2 2 18 7.5]);
subplot(1,2,1);
plot(ain, aout, '.', 'MarkerSize', 3, 'Color', [0.30 0.55 0.85]); hold on;
plot(ag, abs(F0), 'k-', 'LineWidth', 1.5);
plot([0 ag(end)], [0 ag(end)], 'k--', 'LineWidth', 1);
xlim([0 ag(end)]); ylim([0 ag(end)]); grid on;
xlabel('Normalized input amplitude'); ylabel('Normalized output amplitude');
title('(a) AM/AM');
legend({'With memory', 'Memoryless (q = 0)', 'Linear'}, 'Location', 'northwest');

subplot(1,2,2);
plot(ain, dphi, '.', 'MarkerSize', 3, 'Color', [0.30 0.55 0.85]); hold on;
plot(ag, angle(F0)*180/pi, 'k-', 'LineWidth', 1.5);
xlim([0 ag(end)]); grid on;
xlabel('Normalized input amplitude'); ylabel('Phase deviation (deg)');
title('(b) AM/PM');
legend({'With memory', 'Memoryless (q = 0)'}, 'Location', 'southwest');
% 그림 저장 (MATLAB R2020a 이상은 exportgraphics, 그 외 버전·Octave는 print)
if exist('exportgraphics') > 0
    exportgraphics(gcf, 'amam_ampm_hpa.png', 'Resolution', 300);
    exportgraphics(gcf, 'amam_ampm_hpa.pdf', 'ContentType', 'vector');
else
    print(gcf, 'amam_ampm_hpa.png', '-dpng', '-r300');
    print(gcf, 'amam_ampm_hpa.pdf', '-dpdf');
end

%% Fig. 2 (선택): DPD 적용 전후 비교
files  = {'dpd_trans.mat', 'dpd_vtae.mat'};
labels = {'Transformer DPD', 'VT-AE DPD (proposed)'};
colors = {[0.20 0.65 0.30], [0.80 0.20 0.20]};
have   = cellfun(@(f) exist(f, 'file') == 2, files);

if any(have)
    first = find(have, 1);
    S0    = load(files{first});
    xr    = S0.x_ref(:);
    sr    = sqrt(mean(abs(xr).^2) * 10^(IBO_dB/10)) / A_sat;
    Pref  = mean(abs(xr).^2);                              % PA 기준 전력 (원신호)
    y0    = hpa_mp(xr, C, IBO_dB, model, Pref);
    id2   = randperm(numel(xr), min(Nplot, numel(xr)));

    figure('Units','centimeters','Position',[2 2 18 7.5]);
    for p = 1:2
        subplot(1,2,p); hold on; grid on;
        ain2 = abs(xr(id2)) / sr;
        if p == 1, v0 = abs(y0(id2))/sr; else, v0 = angle(y0(id2).*conj(xr(id2)))*180/pi; end
        plot(ain2, v0, '.', 'MarkerSize', 3, 'Color', [0.65 0.65 0.65]);
        leg = {'Without DPD'};
        for f = find(have)
            S  = load(files{f});
            yd = hpa_mp(S.z(:), C, IBO_dB, model, Pref);  % DPD 출력을 같은 PA에 통과
            if p == 1, v = abs(yd(id2))/sr; else, v = angle(yd(id2).*conj(xr(id2)))*180/pi; end
            plot(ain2, v, '.', 'MarkerSize', 3, 'Color', colors{f});
            leg{end+1} = labels{f};
            if p == 1
                fprintf('%s EVM: %.2f %%\n', labels{f}, norm(yd-xr)/norm(xr)*100);
            end
        end
        xlabel('Normalized input amplitude');
        if p == 1
            ylabel('Normalized output amplitude'); title('(a) AM/AM');
            plot([0 max(ain2)], [0 max(ain2)], 'k--', 'LineWidth', 1); leg{end+1} = 'Linear';
            legend(leg, 'Location', 'northwest');
        else
            ylabel('Phase deviation (deg)'); title('(b) AM/PM');
            legend(leg, 'Location', 'southwest');
        end
    end
    % 그림 저장 (MATLAB R2020a 이상은 exportgraphics, 그 외 버전·Octave는 print)
    if exist('exportgraphics') > 0
        exportgraphics(gcf, 'amam_ampm_dpd.png', 'Resolution', 300);
        exportgraphics(gcf, 'amam_ampm_dpd.pdf', 'ContentType', 'vector');
    else
        print(gcf, 'amam_ampm_dpd.png', '-dpng', '-r300');
        print(gcf, 'amam_ampm_dpd.pdf', '-dpdf');
    end
end
