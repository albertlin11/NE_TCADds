clc;
close all;

generation = 10;

legend_names = cell(1, 4);
legend_names{1} = '\bfMLP';
legend_names{2} = '\bfNE 1st generation';
k = 3;
for i = [floor((1+generation)/2), generation]
    legend_names{k} = append('\bfNE ', int2str(i), 'th generation');
    k = k + 1;
end
k = 3;

data = readtable('./loss_gen_csv/loss_gen.csv');
data = table2array(data);
% loss versus fitness
figure('units','centimeter','position',[8, 4, 12, 10])
gen_fitness_mlp = data(1:end, 12);
gen_fitness_mlp = gen_fitness_mlp(~isnan(gen_fitness_mlp));
epochs = length(gen_fitness_mlp);
epochs_list = linspace(1, epochs, epochs)';
epoch_all = epochs;
mlp = plot(epochs_list, gen_fitness_mlp, 'Color', 'black', 'LineWidth', 1.5);
hold on
for i = [1,floor((1+generation)/2), generation]
    gen_fitness_NE = data(1:end, i + 1);
    gen_fitness_NE = gen_fitness_NE(~isnan(gen_fitness_NE));
    epochs = length(gen_fitness_NE);
    epochs_list = linspace(1, epochs, epochs)';
    epoch_all = [epoch_all, epochs];
    NE = plot(epochs_list, gen_fitness_NE,'MarkerSize',20, 'LineWidth', 1.5);
    hold on
    set(gca,'FontSize', 20, 'LineWidth', 4.0,'YScale', 'log','fontweight','bold')
end
xlim([0, max(epoch_all)]),xticks(0:5000:max(epoch_all)), ylim([0 2]);
xlabel('\bfEpochs', 'FontSize',20), ylabel('\bfLoss Fitness score', 'FontSize',20)
legend(legend_names,'FontSize',14, 'Location','northeast','Box','off');

% val loss versus fitness
val_data = readtable('./loss_gen_csv/val_loss_gen.csv');
val_data = table2array(val_data);

figure('units','centimeter','position',[8, 4, 12, 10])
gen_val_fitness_mlp = val_data(1:end, 12);
gen_val_fitness_mlp = gen_val_fitness_mlp(~isnan(gen_val_fitness_mlp));
epochs = length(gen_val_fitness_mlp);
epochs_list = linspace(1, epochs, epochs)';
epoch_all = epochs;
mlp = plot(epochs_list, gen_val_fitness_mlp, 'Color', 'black', 'LineWidth', 1.5);
hold on
for i = [1,floor((1+generation)/2), generation]
    gen_val_fitness_NE = val_data(1:end, i + 1);
    gen_val_fitness_NE = gen_val_fitness_NE(~isnan(gen_val_fitness_NE));
    epochs = length(gen_val_fitness_NE);
    epochs_list = linspace(1, epochs, epochs)';
    epoch_all = [epoch_all, epochs];
    hold on
    NE = plot(epochs_list, gen_val_fitness_NE,'MarkerSize',20, 'LineWidth', 1.5);
    set(gca,'FontSize', 20, 'LineWidth', 4.0,'YScale', 'log','fontweight','bold')
end
xlim([0, max(epoch_all)]),xticks(0:5000: max(epoch_all)), ylim([0 2]);
xlabel('\bfEpochs', 'FontSize',20), ylabel('\bfVal Loss Fitness score', 'FontSize',20)

% legend([mlp, NE], {'\bfMLP','\bfNeuroEvolution'},'FontSize',14, 'Location','northeast','Box','off');
legend(legend_names,'FontSize',14, 'Location','northeast','Box','off');

% MLP fitting and  NE last gen fitting plot
epochs = length(gen_fitness_mlp);
epochs_list = linspace(1, epochs, epochs)';
figure('units','centimeter','position',[8, 4, 12, 10])

loss_fitting = plot(epochs_list, gen_fitness_mlp,'MarkerSize',20, 'LineWidth', 1.5);
hold on
val_loss_fitting = plot(epochs_list, gen_val_fitness_mlp,'MarkerSize',20, 'LineWidth', 1.5);
hold on
set(gca,'FontSize', 20, 'LineWidth', 4.0,'YScale', 'log','fontweight','bold')
xlim([0, epochs]),xticks(0:2500:epochs)%, ylim([0 2]);
xlabel('\bfEpochs', 'FontSize',20), ylabel('\bfFitness score', 'FontSize',20)
annotation('textbox', [0.25 0.8 0.1 0.1], 'String','\bfMLP', 'FontSize',20)

legend([loss_fitting, val_loss_fitting], {'\bfTraining Loss','\bfValidation Loss'},'FontSize',14, 'Location','northeast','Box','off');


epochs = length(gen_fitness_NE);
epochs_list = linspace(1, epochs, epochs)';
figure('units','centimeter','position',[8, 4, 12, 10])

loss_fitting = plot(epochs_list, gen_fitness_NE,'MarkerSize',20, 'LineWidth', 1.5);
hold on
val_loss_fitting = plot(epochs_list, gen_val_fitness_NE,'MarkerSize',20, 'LineWidth', 1.5);
set(gca,'FontSize', 20, 'LineWidth', 4.0,'YScale', 'log','fontweight','bold')
xlim([0, epochs]),xticks(0:2000:epochs), ylim([0 2]);
xlabel('\bfEpochs', 'FontSize',20), ylabel('\bfFitness score', 'FontSize',20)
annotation('textbox', [0.25 0.8 0.1 0.1], 'String','\bfNE', 'FontSize',20)

legend([loss_fitting, val_loss_fitting], {'\bfTraining Loss','\bfValidation Loss'},'FontSize',14, 'Location','northeast','Box','off');




