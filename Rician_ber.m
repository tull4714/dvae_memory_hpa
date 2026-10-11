clear
clc

snr = 0: 2: 18;

% BER vs Eb/N0 = 0:2:18 dB, rows = Rician K = 4, 8, 12, 16, 20
Ideal_BER = [[0.344793, 0.303124, 0.257399, 0.210039, 0.164336, 0.120928, 0.0818629, 0.0491086, 0.0264141, 0.0136242]
    [0.215231, 0.167725, 0.123158, 0.0817746, 0.0467549, 0.0217181, 0.00847895, 0.00322391, 0.00155137, 0.000755859]];
hpa = [[0.346005, 0.304688, 0.259456, 0.212551, 0.167557, 0.125086, 0.0869033, 0.0545175, 0.0311821, 0.0169984]
    [0.21763, 0.170739, 0.12705, 0.0868719, 0.0526331, 0.0272417, 0.0124543, 0.00514957, 0.00237605, 0.00108961]];
transform = [[0.345456, 0.304053, 0.258605, 0.211454, 0.166104, 0.123193, 0.0845845, 0.0519735, 0.0288714, 0.0152568]
    [0.216594, 0.169387, 0.125273, 0.0845542, 0.0499294, 0.0246143, 0.0104382, 0.00400289, 0.00185211, 0.000862305]];
vt_ae = [[0.345165, 0.303618, 0.258053, 0.2108, 0.16532, 0.122163, 0.0833116, 0.0506209, 0.0276946, 0.0144568]
    [0.215994, 0.168612, 0.124276, 0.0832729, 0.048457, 0.0232569, 0.00948367, 0.00355938, 0.00168891, 0.000806211]];

figure(2)
semilogy(snr, Ideal_BER(1, :), 'k-', 'LineWidth', 1.5, 'MarkerSize', 12)
hold on
semilogy(snr, hpa(1, :), 'b--+', 'LineWidth', 1.5, 'MarkerSize', 12)
hold on
semilogy(snr, transform(1, :), 'gs', 'LineStyle', 'none', 'LineWidth', 3, 'MarkerSize', 12)
hold on
semilogy(snr, vt_ae(1, :), 'r:x', 'LineWidth', 1.5, 'MarkerSize', 12)
hold on
axis([0 18 10^-2 1])
%axis square
xlabel('E_b/N_0 [dB]', 'FontName', 'Times New Roman', 'FontSize', 14);
ylabel('Bit Error Rate (BER)', 'FontName', 'Times New Roman', 'FontSize', 14);
legend('Ideal (Clean)', 'HPA (No DPD)', 'Transformer-DPD', 'VT-AE DPD (Proposed)', ...
    'FontName', 'Times New Roman', 'FontSize', 12, 'Location', 'best');
%title('Rayleigh Fading Channel', 'FontName', 'Times New Roman', 'FontSize', 15)
%legend('FontSize',8,'Location','best');
grid on;

figure(3)
semilogy(snr, Ideal_BER(2, :), 'k-', 'LineWidth', 1.5, 'MarkerSize', 12)
hold on
semilogy(snr, hpa(2, :), 'b--+', 'LineWidth', 1.5, 'MarkerSize', 12)
hold on
semilogy(snr, transform(2, :), 'gs', 'LineStyle', 'none', 'LineWidth', 2, 'MarkerSize', 12)
hold on
semilogy(snr, vt_ae(2, :), 'r:x', 'LineWidth', 1.5, 'MarkerSize', 12)
axis([0 18 10^-4 1])
%axis square
xlabel('E_b/N_0 [dB]', 'FontName', 'Times New Roman', 'FontSize', 14);
ylabel('Bit Error Rate (BER)', 'FontName', 'Times New Roman', 'FontSize', 14);
legend('Ideal (Clean)', 'HPA (No DPD)', 'Transformer-DPD', 'VT-AE DPD (Proposed)', ...
    'FontName', 'Times New Roman', 'FontSize', 12, 'Location', 'best');
%title('Rician Fading Channel ({\it K} = 4)', 'FontName', 'Times New Roman', 'FontSize', 15)
%legend('FontSize',8,'Location','best');
grid on;