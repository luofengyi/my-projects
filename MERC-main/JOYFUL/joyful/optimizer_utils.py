"""
优化器工具函数
用于配置分层学习率，适应层次化门控网络的需求
"""

import torch
import torch.optim as optim


def create_hierarchical_optimizer(model, modelF, args):
    """
    创建分层学习率的优化器
    
    门控网络（utterance_gate, dialogue_gate）使用较小的学习率，
    其他部分使用正常学习率，以提高训练稳定性
    
    Args:
        model: JOYFUL主模型
        modelF: AutoFusion融合模型（可能是AutoFusion_Hierarchical）
        args: 配置参数
    
    Returns:
        optimizer: 配置好的优化器
    """
    # 检查是否使用层次化融合
    use_hierarchical = hasattr(modelF, 'utterance_gate') and hasattr(modelF, 'dialogue_gate')
    
    if not use_hierarchical:
        # 如果使用原始AutoFusion，使用标准优化器
        params = list(model.parameters()) + list(modelF.parameters())
        if args.optimizer1 == "adam":
            optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer1 == "adamw":
            optimizer = optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer1 == "sgd":
            optimizer = optim.SGD(params, lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        elif args.optimizer1 == "rmsprop":
            optimizer = optim.RMSprop(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        
        return optimizer
    
    # 使用分层学习率
    # 1. 主模型参数（正常学习率）
    model_params = {
        'params': model.parameters(),
        'lr': args.learning_rate,
        'weight_decay': args.weight_decay
    }
    
    # 2. 融合模型的局部特征学习部分（正常学习率）
    # 需要展开参数生成器
    inter_params_list = []
    inter_params_list.extend(list(modelF.fuse_inInter.parameters()))
    inter_params_list.extend(list(modelF.fuse_outInter.parameters()))
    inter_params_list.extend(list(modelF.projectA.parameters()))
    inter_params_list.extend(list(modelF.projectT.parameters()))
    inter_params_list.extend(list(modelF.projectV.parameters()))
    inter_params_list.extend(list(modelF.projectB.parameters()))
    
    inter_params = {
        'params': inter_params_list,
        'lr': args.learning_rate,
        'weight_decay': args.weight_decay
    }
    
    # 3. 全局情感状态（较小学习率，因为初始化较小）
    global_state_params = {
        'params': [modelF.global_emotion_state],
        'lr': args.learning_rate * 0.1,  # 较小学习率
        'weight_decay': args.weight_decay
    }
    
    # 4. 内层门控网络（较小学习率，避免过度调整）
    utterance_gate_params = {
        'params': modelF.utterance_gate.parameters(),
        'lr': args.learning_rate * 0.1,  # 较小学习率
        'weight_decay': args.weight_decay
    }
    
    # 5. 外层门控网络（较小学习率，避免过度调整）
    dialogue_gate_params = {
        'params': modelF.dialogue_gate.parameters(),
        'lr': args.learning_rate * 0.1,  # 较小学习率
        'weight_decay': args.weight_decay
    }
    
    # 组合所有参数组
    param_groups = [
        model_params,
        inter_params,
        global_state_params,
        utterance_gate_params,
        dialogue_gate_params
    ]
    
    # 创建优化器
    if args.optimizer1 == "adam":
        optimizer = optim.Adam(param_groups)
    elif args.optimizer1 == "adamw":
        optimizer = optim.AdamW(param_groups)
    elif args.optimizer1 == "sgd":
        optimizer = optim.SGD(param_groups, momentum=0.9)
    elif args.optimizer1 == "rmsprop":
        optimizer = optim.RMSprop(param_groups)
    else:
        optimizer = optim.Adam(param_groups)
    
    return optimizer


def print_optimizer_info(optimizer):
    """
    打印优化器的参数组信息，用于调试
    
    Args:
        optimizer: 优化器对象
    """
    print("\n=== Optimizer Parameter Groups ===")
    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i}:")
        print(f"  Learning Rate: {group['lr']}")
        print(f"  Weight Decay: {group.get('weight_decay', 0)}")
        print(f"  Number of Parameters: {sum(p.numel() for p in group['params'])}")
    print("=" * 40 + "\n")

