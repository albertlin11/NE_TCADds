2023_04_27_seed1_1output folder - model architecture with 6 inputs and 1 output
2023_04_28_seed1_3output folder - model architecture with 6 inputs and 3 output

Main folders:
    generate_dataset
    train_model
    figures_and_table

How to use:
step1:
    Go to generate_dataset folder, and then execute generate_dataset.py to delete unnecessary data and generate
    datasets.

    Dataset = 1920, use result_v3.csv.
    Dataset = 2880, use result_v4.csv.
    Dataset = 3840, use result_v5.csv and result_0501.csv.


step2:
    Go to train_model folder, and use the following commands to train and generate models:
        python optimize.py 

step3:
    figures_and_table folder contains the following matlab files to plot figures for manuscript and results table:
        loss_plot.m
        test_plot.m
        trend.m
	test_visualize.m
	train_record.csv
    