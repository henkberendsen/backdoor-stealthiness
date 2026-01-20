import argparse
import json
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--old", type=str, required=True, help="Old hardcoded path")
parser.add_argument("--new", type=str, required=True, help="New harcoded path")
parser.add_argument("--attack_result", type=str, required=True, help="Path to attack_result.pt file")
args = parser.parse_args()

atk_dict = torch.load(args.attack_result, weights_only=False)

for key in ['bd_train', 'bd_test', 'cross_test']:
    try:
        atk_dict[key]['save_folder_path'] = atk_dict[key]['save_folder_path'].replace(args.old, args.new)
        bd_data_container = atk_dict[key]['bd_data_container']
        bd_data_container['save_folder_path'] = bd_data_container['save_folder_path'].replace(args.old, args.new)
        data_dict = bd_data_container['data_dict']

        indices = data_dict.keys()

        for i in indices: 
            data_dict[i]['path'] = data_dict[i]['path'].replace(args.old, args.new)
    except Exception:
        continue

torch.save(atk_dict, args.attack_result)