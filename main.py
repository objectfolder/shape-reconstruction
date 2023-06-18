import argparse
import yaml
from easydict import EasyDict as edict
from Engine import Engine

def parse_args():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modality", type=str, default="joint")
    parser.add_argument("--modality_list", nargs='+', default=['vision','touch','audio'])
    parser.add_argument("--model", type=str, default="PCN")
    parser.add_argument("--config_location", type=str, default="./configs/default.yml")
    parser.add_argument('--eval', action='store_true', default=False, help='if True, only perform testing')
    parser.add_argument("--impact_num", type=int, default=4)
    # DATA_new Locations
    parser.add_argument("--global_gt_points_location", type=str, default='../DATA_new/global_gt_points')
    parser.add_argument("--local_gt_points_location", type=str, default='../DATA_new/local_gt_points_down_sampled')
    parser.add_argument("--visual_images_location", type=str, default='../DATA_new/vision')
    parser.add_argument("--tactile_images_location", type=str, default='../DATA_new/touch')
    parser.add_argument("--tactile_depth_location", type=str, default='../DATA_new/depth')
    parser.add_argument("--audio_spectrogram_location", type=str, default='../DATA_new/audio_spectrogram')
    parser.add_argument("--camera_info_location", type=str, default='../DATA_new/camera_info')
    parser.add_argument("--split_location", type=str, default='../DATA_new/split_cross_object.json')
    # Train & Evaluation
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--global_sample_num', type=int, default=10000)
    parser.add_argument('--local_sample_num', type=int, default=64)
    parser.add_argument('--normalize', action='store_true', default=False, help='if True, normalize the GT point cloud during training')
    # Exp
    parser.add_argument("--exp", type=str, default='test', help = 'The directory to save checkpoints')
    
    args = parser.parse_args()
    return args

def get_config(args):
    cfg_path = args.config_location
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    return edict(config)

def main():
    args = parse_args()
    cfg = get_config(args)
    engine = Engine(args, cfg)
    engine()
    
if __name__ == "__main__":
    main()