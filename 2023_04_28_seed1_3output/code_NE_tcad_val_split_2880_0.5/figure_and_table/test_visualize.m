clc;
close all;

testset = readtable("../train_model/dataset/test_set.csv");
testset = table2array(testset);
trainset = readtable("../train_model/dataset/train_set.csv");
trainset = table2array(trainset);

train_dopant = trainset(:,1);
train_cdose1 = trainset(:,2);
train_cdose2 = trainset(:,3);
train_edose = trainset(:,4);
train_doseenergy1 = trainset(:,5);
train_doseenergy2 = trainset(:,6);
train_ic = trainset(:,7);

test_dopant = testset(:,1);
test_cdose1 = testset(:,2);
test_cdose2 = testset(:,3);
test_edose = testset(:,4);
test_doseenergy1 = testset(:,5);
test_doseenergy2 = testset(:,6);
test_ic = testset(:,7);

figure('units','centimeter','position',[8, 4, 14, 12])
train = scatter3(train_cdose2,train_dopant, train_ic,25,'black','filled');
hold on
test = scatter3(test_cdose2,test_dopant, test_ic,25,'red');

set(gca,'FontSize', 14, 'LineWidth', 2.0);
xlabel('\bfC_d_o_s_e_2(eV)', 'FontSize',20, 'rotation', 15);
ylabel('\bfDopant', 'FontSize',20, 'rotation', -30);
zlabel('\bfI_c(A)', 'FontSize',20);
xticks(4E15:3E15:1E16);xticklabels({'4x10^1^5','7x10^1^5','1x10^1^6'});
legend([train, test], {'\bfTrain set point','\bfTest set point'}','FontSize',14, 'Location','northeast','Box','off');
% box on;
figure('units','centimeter','position',[8, 4, 14, 12])
train = scatter3(train_cdose2,train_cdose1, train_ic,25,'black','filled');
hold on
test = scatter3(test_cdose2,test_cdose1, test_ic,25,'red');

set(gca,'FontSize', 14, 'LineWidth', 2.0);
xlabel('\bfC_d_o_s_e_2(eV)', 'FontSize',20, 'rotation', 15);
ylabel('\bfC_d_o_s_e_1(eV)', 'FontSize',20, 'rotation', -30);
zlabel('\bfI_c(A)', 'FontSize',20);
xticks(4E15:3E15:1E16);xticklabels({'4x10^1^5','7x10^1^5','1x10^1^6'});
yticks(0:3E15:6E15);yticklabels({'0','3x10^1^5','6x10^1^5'});
legend([train, test], {'\bfTrain set point','\bfTest set point'}','FontSize',14, 'Location','northeast','Box','off');

%%%
figure('units','centimeter','position',[8, 4, 14, 12])
train = scatter3(train_cdose2,train_edose, train_ic,25,'black','filled');
hold on
test = scatter3(test_cdose2,test_edose, test_ic,25,'red');

set(gca,'FontSize', 14, 'LineWidth', 2.0);
xlabel('\bfC_d_o_s_e_2(eV)', 'FontSize',20, 'rotation', 15);
ylabel('\bfE_d_o_s_e_1(eV)', 'FontSize',20, 'rotation', -30);
zlabel('\bfI_c(A)', 'FontSize',20);
xticks(4E15:3E15:1E16);xticklabels({'4x10^1^5','7x10^1^5','1x10^1^6'});
yticks(0:3E20:6E20);yticklabels({'0','3x10^2^0','6x10^2^0'});
legend([train, test], {'\bfTrain set point','\bfTest set point'}','FontSize',14, 'Location','northeast','Box','off');

%%%
figure('units','centimeter','position',[8, 4, 14, 12])
train = scatter3(train_doseenergy1,train_doseenergy2, train_ic,25,'black','filled');
hold on
test = scatter3(test_doseenergy1,test_doseenergy2, test_ic,25,'red');

set(gca,'FontSize', 14, 'LineWidth', 2.0);
xlabel('\bfEn_D_o_s_e_1(eV)', 'FontSize',20, 'rotation', 15);
ylabel('\bfEn_D_o_s_e_2(eV)', 'FontSize',20, 'rotation', -30);
zlabel('\bfI_c(A)', 'FontSize',20);
legend([train, test], {'\bfTrain set point','\bfTest set point'}','FontSize',14, 'Location','northeast','Box','off');

