import os
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

plt.style.use('general.mplstyle')

#General comparison figure with absolute and relative values
datasets=['PKIS', 'HTSFP5', 'HTSFP10']
techniques=['DNN', 'Macau']
xlim_value=(-0.05,1.05)


light_colors={'DNN':'#9BC9E4', 'Macau':'#FB849E'}
dark_colors={'DNN':'#348ABD', 'Macau':'#A60628'}
measures={'PKIS': 'RMSD', 'HTSFP5': 'mcc', 'HTSFP10':'mcc'}
locations={'PKIS': 'upper left', 'HTSFP5': 'lower left', 'HTSFP10':'lower left'}
ylim_values={'PKIS': (-5, 65), 'HTSFP5': (-105,5), 'HTSFP10':(-105,5)}


for dataset in datasets:
    #Absolute values
    plt.figure()
    for technique in techniques:
        y_list=[]
        files=[os.path.join('..', 'results', dataset, technique, '{0}_{1}_{2}_labelremovalmodel.dat'.format(dataset, technique,i)) for i in range(1,11)]
        for file in files:
            dat = pd.read_csv(file, sep='\t')
            dat=dat.sort_values('percent_to_delete')
            plt.plot(dat.percent_to_delete, 
              dat.filter(regex='{0}_Test'.format(measures[dataset])).quantile(0.5, axis=1), 
              lw=1.25, c=light_colors[technique], zorder=15, label='')
            y_list.append(dat.filter(regex='{0}_Test'.format(measures[dataset])).quantile(0.5, axis=1))
        y_list=np.array(y_list)
        plt.plot(dat.percent_to_delete, y_list.mean(axis=0), c=dark_colors[technique], lw=2, label=technique, zorder=20)
    plt.xlabel('Deleted activity labels')
    plt.ylabel('Median '+measures[dataset].upper())
    plt.xlim(xlim_value)
    plt.legend(loc=locations[dataset])
    plt.savefig('{0}_absolute.png'.format(dataset))
    
    #Relative values
    plt.figure()
    for i,technique in enumerate(['DNN', 'Macau']):
        y_list=[]
        files=[os.path.join('..', 'results', dataset, technique, '{0}_{1}_{2}_labelremovalmodel.dat'.format(dataset, technique,i)) for i in range(1,11)]
        for file in files:
            dat = pd.read_csv(file, sep='\t')
            dat=dat.sort_values('percent_to_delete')
            start=dat.filter(regex='{0}_Test'.format(measures[dataset])).quantile(0.5, axis=1).values[0]
            plt.plot(dat.percent_to_delete, 
              100*(dat.filter(regex='{0}_Test'.format(measures[dataset])).quantile(0.5, axis=1)-start)/start, 
              lw=1.25, c=light_colors[technique], zorder=15, label='')
            y_list.append(100*(dat.filter(regex='{0}_Test'.format(measures[dataset])).quantile(0.5, axis=1)-start)/start)
        y_list=np.array(y_list)
        plt.plot(dat.percent_to_delete, y_list.mean(axis=0), c=dark_colors[technique], lw=2, label=technique, zorder=20)
    plt.xlabel('Deleted activity labels')
    plt.ylabel('Percent Change '+measures[dataset].upper())
    plt.xlim(xlim_value)
    plt.ylim(ylim_values[dataset])
    plt.legend(loc=locations[dataset])
    plt.savefig('{0}_relative.png'.format(dataset))
    
#Comparison figure for data removal models


datasets=['PKIS', 'HTSFP5', 'HTSFP10']
techniques=['DNN', 'Macau']
models=['label', 'compound']

light_colors={'label':'#9BC9E4', 'compound':'#FB849E'}
dark_colors={'label':'#348ABD', 'compound':'#A60628'}
measures={'PKIS': 'RMSD', 'HTSFP5': 'mcc', 'HTSFP10':'mcc'}
locations={'PKIS': 'upper left', 'HTSFP5': 'lower left', 'HTSFP10':'lower left'}

for dataset in ['PKIS', 'HTSFP5', 'HTSFP10']:
    for technique in techniques:
        plt.figure()
        for model in models:
            y_list=[]
            files=[os.path.join('..', 'results', dataset, technique, '{0}_{1}_{2}_{3}removalmodel.dat'.format(dataset, technique,i, model)) for i in range(1,11)]
            for file in files:
                dat = pd.read_csv(file, sep='\t')
                dat=dat.sort_values('percent_to_delete')
                start=dat.filter(regex='{0}_Test'.format(measures[dataset])).quantile(0.5, axis=1).values[0]
                plt.plot(dat.percent_to_delete, 
                  100*(dat.filter(regex='{0}_Test'.format(measures[dataset])).quantile(0.5, axis=1)-start)/start, 
                  lw=1.25, c=light_colors[model], zorder=15, label='')
                y_list.append(100*(dat.filter(regex='{0}_Test'.format(measures[dataset])).quantile(0.5, axis=1)-start)/start)
            y_list=np.array(y_list)
            plt.plot(dat.percent_to_delete, y_list.mean(axis=0), c=dark_colors[model], lw=2, label=model+'s', zorder=20)
        plt.xlabel('Deleted activity labels')
        plt.ylabel('Percent Change '+measures[dataset].upper())
        plt.xlim(xlim_value)
        plt.ylim(ylim_values[dataset])
        plt.legend(loc=locations[dataset])
        plt.savefig('{0}_{1}_models.png'.format(dataset, technique))
        
        
#Comparison figure for seed values

datasets=['PKIS', 'HTSFP5', 'HTSFP10']
techniques=['DNN', 'Macau']
seed_list=[123456,234567,345678,456789]

dark_colors={123456:'#348ABD', 234567:'#A60628', 345678:'#7A68A6', 456789:'#467821'}
linestyles = {123456:'--',234567: '-.',345678: ':',456789: '-'}


measures={'PKIS': 'RMSD', 'HTSFP5': 'mcc', 'HTSFP10':'mcc'}
locations={'PKIS': 'upper left', 'HTSFP5': 'lower left', 'HTSFP10':'lower left'}
bboxes={'PKIS': (0.002, 0.7), 'HTSFP5': (-0.0015, 0.64), 'HTSFP10':(-0.0015, 0.64)}

for dataset in ['PKIS', 'HTSFP5', 'HTSFP10']:
    for technique in techniques:
        plt.figure()
        
        for i,split_seed in enumerate(seed_list):
            plot_lines=[]
            for j,remove_seed in enumerate(seed_list):
                file=os.path.join('..', 'results', dataset, technique, '{0}_{1}_{2}_{3}.dat'.format(dataset, technique,split_seed, remove_seed))
                dat = pd.read_csv(file, sep='\t')
                dat=dat.sort_values('percent_to_delete')
                start=dat.filter(regex='{0}_Test'.format(measures[dataset])).quantile(0.5, axis=1).values[0]
                plt.plot(dat.percent_to_delete, 
                  100*(dat.filter(regex='{0}_Test'.format(measures[dataset])).quantile(0.5, axis=1)-start)/start, 
                  lw=1.25, c=dark_colors[split_seed], ls=linestyles[remove_seed], zorder=15, label='seed '+str(i+1))
                plot_lines.append(mlines.Line2D([], [], color='black', lw=1.25, zorder=15, ls=linestyles[remove_seed]))  
              
        plt.xlabel('Deleted activity labels')
        plt.ylabel('Percent Change '+measures[dataset].upper())
        plt.xlim(xlim_value)
        plt.ylim(ylim_values[dataset])
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        
        plot_lines=[mlines.Line2D([], [], color='black', lw=1.25, zorder=15, ls=linestyles[remove_seed]) for remove_seed in seed_list]
        plot_labels=["removal "+str(i) for i in range(1,5)]
        legend1 = plt.legend(plot_lines, plot_labels, 
                     loc='upper left', bbox_to_anchor=bboxes[dataset])
        plt.gca().add_artist(legend1)

        plt.legend(by_label.values(), by_label.keys(), loc=locations[dataset])
        plt.savefig('{0}_{1}_seeds.png'.format(dataset, technique))