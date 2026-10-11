%% Fig. 1: EVM of the uncompensated HPA, Transformer DPD, and VT-AE DPD
% 16-QAM OFDM, memory-polynomial HPA, IBO = 1 dB (noise-free, time domain)
clear; close all;

%% 설정
show_title     = false;   % IEEE 논문은 캡션을 쓰므로 그림 안 제목은 끄는 것을 권장
show_reduction = true;    % 막대 위에 보상 전 대비 감소율 표시
font_name      = 'Times New Roman';
font_size      = 9;
bar_w          = 0.58;    % 막대 폭

%% 데이터 (평가 로그 값)
%evm    = [12.3530, 7.2974, 5.5982];                 % [%]
evm = [8.0078, 5.8025, 4.2251];   % HPA, Transformer-DPD, VT-AE
line1  = {'HPA',      'Transformer', 'VT-AE DPD'};  % x축 라벨 첫 줄
line2  = {'(no DPD)', 'DPD',         '(proposed)'}; % x축 라벨 둘째 줄
colors = [0.69 0.69 0.69;      % HPA
          0.42 0.56 0.75;      % Transformer
          0.75 0.22 0.17];     % VT-AE
red    = (evm(1) - evm) / evm(1) * 100;              % 보상 전 대비 감소율 [%]

%% 그림 (IEEE 한 단 폭: 8.8 cm)
figure('Units', 'centimeters', 'Position', [2 2 8.8 6.4], 'Color', 'w');
axes('Units', 'normalized', 'Position', [0.14 0.24 0.82 0.72]);   % 아래쪽에 두 줄 라벨 공간
hold on; box on;

for i = 1:3
    % 막대 (rectangle로 그려 MATLAB·Octave 모두 같은 모양이 되도록 함)
    rectangle('Position', [i - bar_w/2, 0, bar_w, evm(i)], ...
              'FaceColor', colors(i,:), 'EdgeColor', [0.15 0.15 0.15], 'LineWidth', 0.6);
    % 막대 위 값 라벨
    if show_reduction && i > 1
        text(i, evm(i) + 1.45, sprintf('%.2f%%', evm(i)), 'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom', 'FontName', font_name, 'FontSize', font_size, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'Margin', 0.5);
        text(i, evm(i) + 0.35, sprintf('(%.1f%%)', -red(i)), 'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom', 'FontName', font_name, 'FontSize', font_size, 'BackgroundColor', 'w', 'Margin', 0.5);
    else
        text(i, evm(i) + 0.35, sprintf('%.2f%%', evm(i)), 'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom', 'FontName', font_name, 'FontSize', font_size, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'Margin', 0.5);
    end
    % 두 줄 x축 라벨
    text(i, -0.45, line1{i}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
         'FontName', font_name, 'FontSize', font_size);
    text(i, -1.65, line2{i}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
         'FontName', font_name, 'FontSize', font_size);
end

set(gca, 'XTick', 1:3, 'XTickLabel', {}, 'XLim', [0.4 3.6], 'YLim', [0 15], 'YTick', 0:2:14, ...
         'FontName', font_name, 'FontSize', font_size, 'LineWidth', 0.6, ...
         'YGrid', 'on', 'GridLineStyle', '--', 'Layer', 'bottom');   % 격자를 막대 뒤로
ylabel('EVM (%)', 'FontName', font_name, 'FontSize', font_size);
if show_title
    title('EVM comparison (16-QAM, IBO = 1 dB)', 'FontName', font_name, 'FontSize', font_size);
end

%% 저장 (MATLAB R2020a 이상은 exportgraphics, 그 외 버전·Octave는 print)
if exist('exportgraphics') > 0
    exportgraphics(gcf, 'fig_evm_bar.png', 'Resolution', 600);
    exportgraphics(gcf, 'fig_evm_bar.pdf', 'ContentType', 'vector');
else
    print(gcf, 'fig_evm_bar.png', '-dpng', '-r600');
    print(gcf, 'fig_evm_bar.pdf', '-dpdf');
end
