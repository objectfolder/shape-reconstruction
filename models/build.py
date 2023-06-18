import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.optim as optim

def build(args, cfg):
    print("Building model: {}".format(args.model))
    if args.model == 'PCN':
        from PCN import joint_model
        modality_list=args.modality_list
        model = joint_model.AutoEncoder(args, cfg, use_vision='vision' in modality_list,
                                        use_touch='touch' in modality_list,
                                        use_audio='audio' in modality_list)
        optim_params = [{'params': model.linear1.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
                        {'params': model.linear2.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
        # optim_params = [{'params': model.linear.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
        if 'vision' in modality_list:
            if 'vision_checkpoint' in cfg.keys():
                print(f"loading vision ckpt from {cfg.vision_checkpoint}")
                vision_state_dict = torch.load(cfg.vision_checkpoint,map_location='cpu')
                vision_state_dict = {
                    k: v for k, v in vision_state_dict.items() if 'vision_encoder' in k}
                model.load_state_dict(vision_state_dict, strict=False)
                optim_params.append({'params': model.vision_encoder.parameters(), 'lr': args.lr*1e-2, 'weight_decay': args.weight_decay*1e-2})
            else:
                optim_params.append({'params': model.vision_encoder.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        if 'audio' in modality_list:
            if 'audio_checkpoint' in cfg.keys():
                print(f"loading audio ckpt from {cfg.audio_checkpoint}")
                audio_state_dict = torch.load(cfg.audio_checkpoint,map_location='cpu')
                audio_state_dict = {
                    k: v for k, v in audio_state_dict.items() if 'audio_encoder' in k}
                model.load_state_dict(audio_state_dict, strict=False)
                optim_params.append({'params': model.audio_encoder.parameters(), 'lr': args.lr*1e-2, 'weight_decay': args.weight_decay*1e-2})
            else:
                optim_params.append({'params': model.audio_encoder.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        if 'touch' in modality_list:
            if 'touch_checkpoint' in cfg.keys():
                print(f"loading touch ckpt from {cfg.touch_checkpoint}")
                touch_state_dict = torch.load(cfg.touch_checkpoint,map_location='cpu')
                touch_state_dict = {
                    k: v for k, v in touch_state_dict.items() if 'tactile_encoder' in k}
                model.load_state_dict(touch_state_dict, strict=False)
                optim_params.append({'params': model.tactile_encoder.parameters(), 'lr': args.lr*1e-2, 'weight_decay': args.weight_decay*1e-2})
            else:
                optim_params.append({'params': model.tactile_encoder.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        optimizer = optim.AdamW(optim_params)
        
    elif args.model == 'MDN':
        from MDN import joint_model
        modality_list=args.modality_list
        model = joint_model.Encoder(args, cfg, use_vision='vision' in modality_list,
                                    use_touch='touch' in modality_list,
                                    use_audio='audio' in modality_list)
        optim_params = [{'params': model.mesh_decoder.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
        if 'vision' in modality_list:
            if 'vision_checkpoint' in cfg.keys():
                print(f"loading vision ckpt from {cfg.vision_checkpoint}")
                vision_state_dict = torch.load(cfg.vision_checkpoint,map_location='cpu')
                vision_state_dict = {
                    k: v for k, v in vision_state_dict.items() if 'img_encoder' in k}
                model.load_state_dict(vision_state_dict, strict=False)
                optim_params.append({'params': model.img_encoder.parameters(), 'lr': args.lr*1e-3, 'weight_decay': args.weight_decay})
            else:
                optim_params.append({'params': model.img_encoder.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        if 'audio' in modality_list:
            if 'audio_checkpoint' in cfg.keys():
                print(f"loading audio ckpt from {cfg.audio_checkpoint}")
                audio_state_dict = torch.load(cfg.audio_checkpoint,map_location='cpu')
                audio_state_dict = {
                    k: v for k, v in audio_state_dict.items() if 'audio_encoder' in k or 'position_encoder' in k}
                model.load_state_dict(audio_state_dict, strict=False)
                optim_params.append({'params': model.audio_encoder.parameters(), 'lr': args.lr*1e-3, 'weight_decay': 0})
                optim_params.append({'params': model.position_encoder.parameters(), 'lr': args.lr*1e-3, 'weight_decay': 0})
            else:
                optim_params.append({'params': model.audio_encoder.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
                optim_params.append({'params': model.position_encoder.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        optimizer = optim.AdamW(optim_params)
    elif args.model == 'ReconTransformer':
        from ReconTransformer import ReconTransformer
        modality_list = args.modality_list
        model = ReconTransformer.ReconstructionTransformer(args, cfg, 
                                        use_vision='vision' in modality_list,
                                        use_touch='touch' in modality_list,
                                        use_audio='audio' in modality_list)
        optim_params = []
        
        optim_params.append({'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr, 'weight_decay': args.weight_decay})
        optimizer = optim.AdamW(optim_params)
        
    return model, optimizer
    