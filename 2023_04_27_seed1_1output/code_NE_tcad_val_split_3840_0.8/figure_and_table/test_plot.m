clc;
close all;
clear all;
generation = 10;
indiv = 10;
data = readtable("./NE_performance.csv");
data = table2array(data);
data_mlp = readtable("./mlp_performance.csv");
data_mlp = table2array(data_mlp);

figure('units','centimeter','position',[8, 4, 12, 10])
data_y_mlp = data_mlp(2:end, 1) * 100;
for i=1:generation
    data_y = 100 * data(2:end, i);
    data_x = i * ones(indiv,1);
    scatter(data_x, data_y, 60,'filled');
    hold on
end
mlp = yline(data_y_mlp,'-','MLP', 'Color', 'black', 'LineWidth', 1.5);
set(gca,'FontSize', 20, 'LineWidth', 4.0,'fontweight','bold')

xlim([0 generation + 1]);
ylim([0.05 0.40]);
xlabel('\bfGeneration', 'FontSize',20);
ylabel('\bfTest set(mape)%', 'FontSize',20);
legend('Individual in each generation','FontSize',12, 'Location','northeast','Box','off')
box on;

figure('units','centimeter','position',[8, 4, 12, 10])
data_x = linspace(1, generation, generation);
data_y = data(2,1:end) * 100;
data_y_mlp = data_mlp(2:end, 1) * 100;
NE = plot(data_x, data_y,'-b.', 'MarkerSize',20, 'LineWidth', 1.5);
hold on 
mlp = yline(data_y_mlp,'-','MLP', 'Color', 'black', 'LineWidth', 1.5);%plot(data_x, data_y_mlp, 'Color', 'black', 'LineWidth', 1.5);
set(gca,'FontSize', 20, 'LineWidth', 4.0,'fontweight','bold')

xlim([0 generation + 1]);
ylim([0 2*data_y_mlp]);
xlabel('\bfGeneration', 'FontSize',20);
ylabel('\bfTest set(mape)%', 'FontSize',20);
legend('dataset trend','FontSize',14, 'Location','northeast','Box','off')
legend([NE, mlp], {'\bfNeuroEvolution','\bfMLP'},'FontSize',14, 'Location','northeast','Box','off');
box on;
