import matplotlib.pyplot as plt
import numpy as np
import pickle
import os




def build_tables():
    cwd = os.getcwd()
    root_dir = cwd + '/results'


    for dataset_folders in next(os.walk(root_dir))[1]:        #loop through data sets
        current_path = os.path.join(root_dir, dataset_folders)
        # need to build a 2d grid . we will build one for each thing. turn into a numpy array
        # and add to the previos numpy array and then average across all random seeds
        # do this by building up a list and appending to a bigger list
        final_table_list = np.zeros((15, 2))
        for random_seed_folder in  next(os.walk(current_path))[1]:  #loop through all 10 random seeds
            table_list = []
            current_path2 = os.path.join(current_path, random_seed_folder)
            for _, _, files in os.walk(current_path2):
                for file in files:    # loop through each iteration of continual learning process
                    with open(os.path.join(current_path2, file), 'rb') as f:
                        state = pickle.load(f)
                        optimal_id_accuracy = state['best_id_accuracy']
                        print('optimal id accuracy', optimal_id_accuracy)
                        optimal_ood_accuracy = state['best_ood_accuracy']
                        print('optimal ood accuracy', optimal_ood_accuracy)
                        optimal_total_accuracy = state['best_accuracy']
                        print('best accuracy', optimal_total_accuracy)
                        original_id_accuracy = state['id_accuracy_original']
                        print('id acc original', original_id_accuracy)
                        original_ood_accuracy = state['ood_accuracy_original']
                        print('ood_accuracy original', original_ood_accuracy)
                        original_total_accuracy = state['total_accuracy_original']
                        print('original total accuracy', original_total_accuracy)
                        table_list.append([optimal_id_accuracy, original_id_accuracy])
                        table_list.append([optimal_ood_accuracy, original_ood_accuracy])
                        table_list.append([optimal_total_accuracy, original_total_accuracy])                        
            table_list = np.array(table_list)
            final_table_list += table_list
        final_table_list /= 10 # average on 10 random seeds
        final_table_list = final_table_list.tolist()
        final_table_list = [[f'{elem}' for elem in row] for row in final_table_list]
        
        row_names = ['id accuracy', 'ood accuracy', 'total accuracy']
        row_labels = []
        for i in range(5, 10):
            for row_name in row_names:
               row_labels.append(row_name + f' {i}')
        print(row_labels)
        col_labels  = ['Optimal', '1 std dev']
        
        fig, ax = plt.subplots()
        ax.set_axis_off()
        #print(len(final_table_list))
        #print(len(final_table_list[0]))
        #print(len(row_labels))
        #print(len(col_labels)) 
        table = ax.table( cellText = final_table_list, rowLabels = row_labels, colLabels = col_labels, rowColours =["blue"] * 15, colColours =["blue"] * 2, colWidths=[.3, .3,.3], cellLoc ='right',  loc ='center right')         
   
        ax.set_title(state['dataset'], fontweight ="bold") 
        # fig = plt.figure()
        # generate your plot
        plt.savefig(cwd + '/results//' + state['dataset'] + '.jpg', dpi = 800)
                        
    

build_tables()
