clc;
clear;
trend_line = readtable('trend_doseenergy1.csv');
trend_line = table2array(trend_line);
sample_rate = 10;
figure_num = length(trend_line) / sample_rate;
for i = 1:figure_num
    % linear
    figure('units','centimeter','position',[8, 4, 12, 10])
    x = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 5);
    y_true = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 9);
    y_NE_pred = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 7);
    y_mlp_pred = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 8);
    y_min = min(trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 8));
    y_max = max(trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 8));
    simulate = plot(x, y_true, 'bo', 'MarkerSize', 5);
    hold on
    pred_mlp = plot(x, y_mlp_pred, 'Color', 'r', 'LineWidth', 1);
    hold on
    pred_NE = plot(x, y_NE_pred, 'Color', 'b', 'LineWidth', 1);
    hold on
    set(gca,'FontSize', 20, 'LineWidth', 4.0)
    xlim([20, 420]),ylim([(y_min/2) (y_max + y_min/2)])%,xticks(0:0.3:0.9), yticks(0:5e-4:1e-3);
    xlabel('\bfDose Energy 2(eV)', 'FontSize',20), ylabel('\bfI_c (A) ', 'FontSize',20)
    legend([pred_mlp, pred_NE, simulate], {'\bfMLP', '\bfNE', '\bfTrue'},'FontSize',14, 'Location','northwest','Box','off');
    % log scale
    figure('units','centimeter','position',[8, 4, 12, 10])
    x = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 5);
    y_true = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 9);
    y_NE_pred = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 7);
    y_mlp_pred = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 8);
    simulate = plot(x, y_true, 'bo', 'MarkerSize', 5);
    hold on
    pred_mlp = plot(x, y_mlp_pred, 'Color', 'r', 'LineWidth', 1);
    hold on
    pred_NE = plot(x, y_NE_pred, 'Color', 'b', 'LineWidth', 1);
    hold on
    set(gca,'FontSize', 20, 'LineWidth', 4.0,'YScale','log');
    xlim([20, 420]),ylim([(y_min/2) (y_max + y_min * 2)])%,xticks(0:0.3:0.9), yticks(0:5e-4:1e-3);
    xlabel('\bfDose Energy 2(eV)', 'FontSize',20), ylabel('\bfI_c (A) ', 'FontSize',20)
    legend([pred_mlp, pred_NE, simulate], {'\bfMLP', '\bfNE', '\bfTrue'},'FontSize',14, 'Location','northwest','Box','off');
end

trend_line = readtable('trend_doseenergy2.csv');
trend_line = table2array(trend_line);
sample_rate = 8;
figure_num = length(trend_line) / sample_rate;
for i = 1:figure_num
    % linear
    figure('units','centimeter','position',[8, 4, 12, 10])
    x = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 6);
    y_true = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 9);
    y_NE_pred = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 7);
    y_mlp_pred = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 8);
    y_min = min(trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 8));
    y_max = max(trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 8));
    simulate = plot(x, y_true, 'bo', 'MarkerSize', 5);
    hold on
    pred_mlp = plot(x, y_mlp_pred, 'Color', 'r', 'LineWidth', 1);
    hold on
    pred_NE = plot(x, y_NE_pred, 'Color', 'b', 'LineWidth', 1);
    hold on
    set(gca,'FontSize', 20, 'LineWidth', 4.0)
    xlim([20, 110]),ylim([(15 / 16 * y_min) (y_max + y_min / 16)])%,ylim([(y_min/2) (y_max + y_min/2)])%,xticks(0:0.3:0.9), yticks(0:5e-4:1e-3);
    xlabel('\bfDose Energy 1(eV)', 'FontSize',20), ylabel('\bfI_c (A) ', 'FontSize',20)
    legend([pred_mlp, pred_NE, simulate], {'\bfMLP', '\bfNE', '\bfTrue'},'FontSize',14, 'Location','northwest','Box','off');
    % log scale
    figure('units','centimeter','position',[8, 4, 12, 10])
    x = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 6);
    y_true = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 9);
    y_NE_pred = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 7);
    y_mlp_pred = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 8);
    simulate = plot(x, y_true, 'bo', 'MarkerSize', 5);
    hold on
    pred_mlp = plot(x, y_mlp_pred, 'Color', 'r', 'LineWidth', 1);
    hold on
    pred_NE = plot(x, y_NE_pred, 'Color', 'b', 'LineWidth', 1);
    hold on
    set(gca,'FontSize', 20, 'LineWidth', 4.0,'YScale','log');
    xlim([20, 110]),ylim([(15 / 16 * y_min) (y_max + y_min / 16)])%,ylim([(y_min/2) (y_max + y_min)])%,xticks(0:0.3:0.9), yticks(0:5e-4:1e-3);
    xlabel('\bfDose Energy 1(eV)', 'FontSize',20), ylabel('\bfI_c (A) ', 'FontSize',20)
    legend([pred_mlp, pred_NE, simulate], {'\bfMLP', '\bfNE', '\bfTrue'},'FontSize',14, 'Location','northwest','Box','off');
end

trend_line = readtable('trend_cdose2.csv');
trend_line = table2array(trend_line);
sample_rate = 6;
figure_num = length(trend_line) / sample_rate;
for i = 1:figure_num
    % linear
    figure('units','centimeter','position',[8, 4, 12, 10])
    x = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 3);
    y_true = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 9);
    y_NE_pred = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 7);
    y_mlp_pred = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 8);
    y_min = min(trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 8));
    y_max = max(trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 8));
    simulate = plot(x, y_true, 'bo', 'MarkerSize', 5);
    hold on
    pred_mlp = plot(x, y_mlp_pred, 'Color', 'r', 'LineWidth', 1);
    hold on
    pred_NE = plot(x, y_NE_pred, 'Color', 'b', 'LineWidth', 1);
    hold on
    set(gca,'FontSize', 20, 'LineWidth', 4.0)
    xlim([4.5E15, 11E15]),ylim([(15 / 16 * y_min) (y_max + y_min / 16)])%,xticks(0:0.3:0.9), yticks(0:5e-4:1e-3);
    xlabel('\bfCdose 2(cm^-^2)', 'FontSize',20), ylabel('\bfI_c (A) ', 'FontSize',20)
    legend([pred_mlp, pred_NE, simulate], {'\bfMLP', '\bfNE', '\bfTrue'},'FontSize',14, 'Location','northwest','Box','off');
    % log scale
    figure('units','centimeter','position',[8, 4, 12, 10])
    x = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 3);
    y_true = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 9);
    y_NE_pred = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 7);
    y_mlp_pred = trend_line(1 + (i - 1) * sample_rate:  i * sample_rate, 8);
    simulate = plot(x, y_true, 'bo', 'MarkerSize', 5);
    hold on
    pred_mlp = plot(x, y_mlp_pred, 'Color', 'r', 'LineWidth', 1);
    hold on
    pred_NE = plot(x, y_NE_pred, 'Color', 'b', 'LineWidth', 1);
    hold on
    set(gca,'FontSize', 20, 'LineWidth', 4.0,'YScale','log');
    xlim([4.5E15, 11E15]),ylim([(15 / 16 * y_min) (y_max + y_min / 16)])%ylim([(y_min/2) (y_max + y_min)])%,xticks(0:0.3:0.9), yticks(0:5e-4:1e-3);
    xlabel('\bfCdose 2(cm^-^2)', 'FontSize',20), ylabel('\bfI_c (A) ', 'FontSize',20)
    legend([pred_mlp, pred_NE, simulate], {'\bfMLP', '\bfNE', '\bfTrue'},'FontSize',14, 'Location','northwest','Box','off');
end




