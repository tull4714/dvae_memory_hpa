%% Fig. 1: EVM of the uncompensated HPA, Transformer DPD, and VT-AE DPD
% 16-QAM OFDM, memory-polynomial HPA, IBO = 8 dB (noise-free, time domain)
clear; close all;

%% 설정
show_title     = false;   % IEEE 논문은 캡션을 쓰므로 그림 안 제목은 끄는 것을 권장
show_reduction = true;    % 막대 위에 보상 전 대비 감소율 표시
font_name      = 'Times New Roman';
font_size      = 18;
bar_w          = 0.58;    % 막대 폭

%% 데이터 (평가 로그 값)

evm_hpa = 8.0078;                                % 잡음 없는 EVM [%]: HPA
% 잡음 없는 EVM [%], 5회 독립 학습 (seed none, 2, 3, 4, 5)
runs_T = [5.8025, 5.7288, 5.7626, 5.7777, 5.6259];   % Transformer
runs_V = [5.0886, 4.8421, 4.9614, 4.8196, 4.8719];   % VT-AE

%% ===================== 1) EVM 막대 =====================
mu  = [evm_hpa, mean(runs_T), mean(runs_V)];      % 막대 = 평균
sd  = [0, std(runs_T), std(runs_V)];              % 오차 막대 = 표준편차 (n-1 정규화)
top = [evm_hpa, max(runs_T), max(runs_V)];        % 값 표시 위치
R   = {[], runs_T, runs_V};

redu    = (mu(1) - mu) / mu(1) * 100;              % 보상 전 대비 감소율 [%]

%% 그림
line1  = {'HPA',      'Transformer', 'VT-AE DPD'};  % x축 라벨 첫 줄
line2  = {'(no DPD)', 'DPD',         '(proposed)'}; % x축 라벨 둘째 줄
colors = [0.69 0.69 0.69;      % HPA
    0.42 0.56 0.75;      % Transformer
    0.75 0.22 0.17];     % VT-AE

for i = 1:3
    % 막대 (rectangle로 그려 MATLAB·Octave 모두 같은 모양이 되도록 함)
    rectangle('Position', [i - bar_w/2, 0, bar_w, mu(i)], ...
        'FaceColor', colors(i,:), 'EdgeColor', [0.15 0.15 0.15], 'LineWidth', 0.6);
    % 막대 위 값 라벨
    if show_reduction && i > 1
        text(i, mu(i) + 0.85, sprintf('%.2f%%', mu(i)), 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'FontName', font_name, 'FontSize', font_size, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'Margin', 0.5);
        text(i, mu(i) + 0.35, sprintf('(%.1f%%)', -redu(i)), 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'FontName', font_name, 'FontSize', font_size, 'BackgroundColor', 'w', 'Margin', 0.5);
    else
        text(i, mu(i) + 0.35, sprintf('%.2f%%', mu(i)), 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'FontName', font_name, 'FontSize', font_size, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'Margin', 0.5);
    end
    % 두 줄 x축 라벨
    text(i, -0.25, line1{i}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
        'FontName', font_name, 'FontSize', font_size);
    text(i, -0.75, line2{i}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
        'FontName', font_name, 'FontSize', font_size);
end
hold on
for i = 2:3                                       % 오차 막대(표준편차)와 각 학습 결과(점)
    plot([i i], mu(i) + sd(i) * [-1 1], 'k-', 'LineWidth', 2);
    plot(i + 0.25 * [-1 1], (mu(i) - sd(i)) * [1 1], 'k-', 'LineWidth', 2);
    plot(i + 0.25 * [-1 1], (mu(i) + sd(i)) * [1 1], 'k-', 'LineWidth', 2);
    xs = i + linspace(-0.2, 0.2, numel(R{i}));
    plot(xs, R{i}, 'o', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
end

set(gca, 'XTick', 1:3, 'XTickLabel', {}, 'XLim', [0.4 3.6], 'YLim', [0 10], 'YTick', 0:2:14, ...
    'FontName', font_name, 'FontSize', font_size, 'LineWidth', 0.6, ...
    'YGrid', 'on', 'GridLineStyle', '--', 'Layer', 'bottom');   % 격자를 막대 뒤로
ylabel('Noise-free EVM (%)', 'FontName', font_name, 'FontSize', font_size);
if show_title
    title('EVM comparison (16-QAM, IBO = 1 dB)', 'FontName', font_name, 'FontSize', font_size);
end