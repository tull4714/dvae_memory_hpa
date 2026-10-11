clear
clc

snr = 0: 2: 18;

ideal       = [49.99, 39.716, 31.547, 25.061, 19.9, 15.815, 12.559, 9.9775, 7.9238, 6.2949];
hpa         = [50.876, 40.714, 32.705, 26.431, 21.545, 17.799, 14.949, 12.832, 11.293, 10.205];
transform   = [50.478, 40.259, 32.173, 25.801, 20.788, 16.892, 13.87, 11.568, 9.8415, 8.5765];
vt_ae       = [50.252, 40.008, 31.883, 25.458, 20.378, 16.395, 13.271, 10.85, 8.9926, 7.5913];

figure(3)
plot(snr, ideal, '-vk', 'LineWidth', 1.5, 'MarkerSize', 10); hold on
plot(snr, hpa, '-sb', 'LineWidth', 1.5, 'MarkerSize', 10); hold on
plot(snr, transform, '-+g', 'LineWidth', 1.5, 'MarkerSize', 10); hold on
plot(snr, vt_ae, '-xr', 'LineWidth', 1.5, 'MarkerSize', 10); hold on
grid on;
xlabel('E_b/N_0 [dB]', 'FontName', 'Times New Roman', 'FontSize', 14);
ylabel('Error Vector Magnitude (EVM) [%]', 'FontName', 'Times New Roman', 'FontSize', 14);
legend('Ideal (Clean)', 'HPA (No DPD)', 'Transformer-DPD', 'VT-AE DPD (Proposed)', ...
    'FontName', 'Times New Roman', 'FontSize', 12);
title('AWGN EVM vs. E_b/N_0 (16-QAM, IBO = 1 dB)', 'FontName', 'Times New Roman', 'FontSize', 14)
axis([10, 18, 6, 20])