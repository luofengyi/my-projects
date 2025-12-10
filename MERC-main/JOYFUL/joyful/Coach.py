import copy
import time

import numpy as np
from numpy.core import overrides
import torch
from tqdm import tqdm
from sklearn import metrics

import joyful
from joyful.loss_utils import LossWeightConfig

log = joyful.utils.get_logger()


class Coach:
    def __init__(self, trainset, devset, testset, model, modelF, opt1 , sched1, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.modelF = modelF
        self.opt1 = opt1
        self.scheduler = sched1
        self.args = args
        self.dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
            "meld": {"Neutral": 0, "Surprise": 1, "Fear": 2, "Sadness": 3, "Joy": 4, "Disgust": 5, "Angry": 6}
        }

        if args.dataset and args.emotion == "multilabel":
            self.dataset_label_dict["mosei"] = {
                "happiness": 0,
                "sadness": 1,
                "anger": 2,
                "surprise": 3,
                "disgust": 4,
                "fear": 5,
            }

        if args.emotion == "7class":
            self.label_to_idx = {
                "Strong Negative": 0,
                "Weak Negative": 1,
                "Negative": 2,
                "Neutral": 3,
                "Positive": 4,
                "Weak Positive": 5,
                "Strong Positive": 6,
            }
        else:
            self.label_to_idx = self.dataset_label_dict[args.dataset]

        self.best_dev_f1 = None
        self.best_epoch = None
        self.best_state = None
        
        # 使用基础优化方案的损失权重配置
        self.loss_weight_config = LossWeightConfig.from_args(args)
        
        # 训练过程记录（用于可视化）
        self.training_history = {
            'epochs': [],
            'train_losses': [],
            'train_f1s': [],
            'dev_f1s': [],
            'test_f1s': [],
            'class_f1s': {}  # 每个类别的F1分数
        }
        # ULGM权重调度参数
        self.unimodal_init_weight = getattr(args, 'unimodal_init_weight', 0.0)
        self.unimodal_target_weight = getattr(args, 'unimodal_loss_weight', 0.001)
        self.unimodal_warmup_epochs = max(1, getattr(args, 'unimodal_warmup_epochs', 5))
        self.unimodal_delay_epochs = max(0, getattr(args, 'unimodal_delay_epochs', 0))

    def load_ckpt(self, ckpt):
        print('')

    def train(self):
        log.debug(self.model)

        best_dev_f1, best_epoch, best_state = (
            self.best_dev_f1,
            self.best_epoch,
            self.best_state,
        )

        dev_f1s = []
        test_f1s = []
        train_losses = []
        best_test_f1 = None

        # Train
        for epoch in range(1, self.args.epochs + 1):
            train_loss, train_f1 = self.train_epoch(epoch)
            dev_f1, dev_loss, dev_class_f1s = self.evaluate()
            self.scheduler.step(dev_loss)
            test_f1, test_loss, test_class_f1s = self.evaluate(test=True)
            if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
                test_f1 = np.array(list(test_f1.values())).mean()
            log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))

            # 记录训练历史
            self.training_history['epochs'].append(epoch)
            self.training_history['train_losses'].append(train_loss)
            self.training_history['train_f1s'].append(train_f1)
            self.training_history['dev_f1s'].append(dev_f1)
            self.training_history['test_f1s'].append(test_f1 if isinstance(test_f1, (int, float)) else np.array(list(test_f1.values())).mean() if isinstance(test_f1, dict) else test_f1)
            
            # 记录各类别F1分数（如果是IEMOCAP_4）
            if self.args.dataset in ["iemocap_4", "iemocap"] and isinstance(test_class_f1s, dict):
                for class_name, f1_score in test_class_f1s.items():
                    if class_name not in self.training_history['class_f1s']:
                        self.training_history['class_f1s'][class_name] = []
                    self.training_history['class_f1s'][class_name].append(f1_score)

            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                if self.args.dataset == "mosei":
                    torch.save(
                        {"args": self.args, "state_dict": self.model},
                        "model_checkpoints/mosei_best_dev_f1_model_"
                        + self.args.modalities
                        + "_"
                        + self.args.emotion
                        + ".pt",
                    )
                else:
                    torch.save({
                        "args": self.args,
                        'modelN_state_dict': self.model,
                        'modelF_state_dict': self.modelF,
                        'lr': self.scheduler._last_lr
                    }, "model_checkpoints/"
                        + self.args.dataset
                        + "_best_dev_f1_model_"
                        + self.args.modalities
                        + ".pt")

                log.info("Save the best model.")
            log.info("[Test set] [f1 {:.4f}]".format(test_f1))
            dev_f1s.append(dev_f1)
            test_f1s.append(test_f1)
            train_losses.append(train_loss)
        
        # 保存训练历史（用于可视化）
        self.save_training_history()
        
        if self.args.tuning:
            self.args.experiment.log_metric("best_dev_f1", best_dev_f1, epoch=epoch)
            self.args.experiment.log_metric("best_test_f1", best_test_f1, epoch=epoch)

            return best_dev_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s

        return best_dev_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0

        self.model.train()
        self.modelF.train()

        self.trainset.shuffle()
        current_unimodal_weight = self._compute_unimodal_weight(epoch)
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            self.modelF.zero_grad()
            data = self.trainset[idx]
            encoderL = data['encoder_loss']
            for k, v in data.items():
                if not k == "utterance_texts":
                    data[k] = v.to(self.args.device)
            
            # 使用基础优化方案：可配置的损失权重
            encoder_loss_weight = self.loss_weight_config.encoder_loss_weight
            nll = self.model.get_loss(data, True) + encoder_loss_weight * encoderL.to(self.args.device)
            
            # 添加单模态损失（ULGM模块，如果存在）
            if 'unimodal_loss' in data and current_unimodal_weight > 0:
                unimodal_loss = data['unimodal_loss'].to(self.args.device)
                # 确保单模态损失是标量
                if isinstance(unimodal_loss, torch.Tensor):
                    if unimodal_loss.dim() > 0:
                        unimodal_loss = unimodal_loss.mean()
                nll = nll + current_unimodal_weight * unimodal_loss
            
            epoch_loss += nll.item()
            nll.backward()
            
            # 添加梯度裁剪，防止梯度爆炸导致训练不稳定
            max_grad_norm = getattr(self.args, 'max_grad_norm', 1.0)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.modelF.parameters(), max_norm=max_grad_norm)
            
            self.opt1.step()

        # 计算训练集F1分数
        train_f1 = self.evaluate_train_set()
        
        end_time = time.time()
        log.info("")
        log.info(
            "[Epoch %d] [Loss: %f] [Train F1: %f] [Time: %f]"
            % (epoch, epoch_loss, train_f1, end_time - start_time)
        )
        return epoch_loss, train_f1
    
    def _compute_unimodal_weight(self, epoch):
        """
        根据当前epoch计算ULGM损失权重，先延迟、再线性升至目标权重。
        """
        if self.unimodal_target_weight <= 0:
            return 0.0
        if epoch <= self.unimodal_delay_epochs:
            return self.unimodal_init_weight
        progress_epoch = epoch - self.unimodal_delay_epochs
        progress = min(1.0, progress_epoch / self.unimodal_warmup_epochs)
        return self.unimodal_init_weight + (self.unimodal_target_weight - self.unimodal_init_weight) * progress
    
    def evaluate_train_set(self):
        """评估训练集，计算F1分数"""
        self.model.eval()
        self.modelF.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in range(len(self.trainset)):
                data = self.trainset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    if not k == "utterance_texts":
                        data[k] = v.to(self.args.device)
                y_hat = self.model(data, False)
                preds.append(y_hat.detach().to("cpu"))
            
            if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
                golds = torch.cat(golds, dim=0).numpy()
                preds = torch.cat(preds, dim=0).numpy()
                f1 = metrics.f1_score(golds, preds, average="weighted")
            else:
                golds = torch.cat(golds, dim=-1).numpy()
                preds = torch.cat(preds, dim=-1).numpy()
                f1 = metrics.f1_score(golds, preds, average="weighted")
        
        self.model.train()
        self.modelF.train()
        return f1
    
    def save_training_history(self):
        """保存训练历史到文件"""
        import os
        import json
        
        # 创建保存目录
        save_dir = "training_history"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存为JSON文件
        history_file = os.path.join(
            save_dir,
            f"{self.args.dataset}_{self.args.modalities}_history.json"
        )
        
        # 转换numpy类型为Python原生类型
        history_to_save = {
            'epochs': self.training_history['epochs'],
            'train_losses': [float(x) for x in self.training_history['train_losses']],
            'train_f1s': [float(x) for x in self.training_history['train_f1s']],
            'dev_f1s': [float(x) for x in self.training_history['dev_f1s']],
            'test_f1s': [float(x) for x in self.training_history['test_f1s']],
            'class_f1s': {
                k: [float(x) for x in v] 
                for k, v in self.training_history['class_f1s'].items()
            }
        }
        
        with open(history_file, 'w') as f:
            json.dump(history_to_save, f, indent=2)
        
        log.info(f"Training history saved to {history_file}")
        return history_file

    def evaluate(self, test=False):
        dev_loss = 0
        dataset = self.testset if test else self.devset
        self.model.eval()
        self.modelF.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    if not k == "utterance_texts":
                        data[k] = v.to(self.args.device)
                y_hat = self.model(data,False)
                preds.append(y_hat.detach().to("cpu"))
                nll = self.model.get_loss(data,False)
                dev_loss += nll.item()

            if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
                golds = torch.cat(golds, dim=0).numpy()
                preds = torch.cat(preds, dim=0).numpy()
                f1 = metrics.f1_score(golds, preds, average="weighted")
                acc = metrics.accuracy_score(golds, preds)
                if self.args.tuning:
                    self.args.experiment.log_metric("dev_acc", acc)
            else:
                golds = torch.cat(golds, dim=-1).numpy()
                preds = torch.cat(preds, dim=-1).numpy()
                f1 = metrics.f1_score(golds, preds, average="weighted")

            # 计算各类别F1分数（用于可视化）
            class_f1s = {}
            if test:
                print(
                    metrics.classification_report(
                        golds, preds, target_names=self.label_to_idx.keys(), digits=4
                    )
                )

                if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
                    happy = metrics.f1_score(
                        golds[:, 0], preds[:, 0], average="weighted"
                    )
                    sad = metrics.f1_score(golds[:, 1], preds[:, 1], average="weighted")
                    anger = metrics.f1_score(
                        golds[:, 2], preds[:, 2], average="weighted"
                    )
                    surprise = metrics.f1_score(
                        golds[:, 3], preds[:, 3], average="weighted"
                    )
                    disgust = metrics.f1_score(
                        golds[:, 4], preds[:, 4], average="weighted"
                    )
                    fear = metrics.f1_score(
                        golds[:, 5], preds[:, 5], average="weighted"
                    )

                    f1 = {
                        "happy": happy,
                        "sad": sad,
                        "anger": anger,
                        "surprise": surprise,
                        "disgust": disgust,
                        "fear": fear,
                    }
                    class_f1s = f1
                elif self.args.dataset in ["iemocap_4", "iemocap"]:
                    # 计算每个类别的F1分数
                    label_names = list(self.label_to_idx.keys())
                    for i, label_name in enumerate(label_names):
                        # 对于多分类，计算每个类别的F1
                        class_f1 = metrics.f1_score(
                            golds == i, preds == i, average="binary", zero_division=0
                        )
                        class_f1s[label_name] = class_f1
            else:
                # 对于dev集，也计算各类别F1（如果是IEMOCAP_4）
                if self.args.dataset in ["iemocap_4", "iemocap"]:
                    label_names = list(self.label_to_idx.keys())
                    for i, label_name in enumerate(label_names):
                        class_f1 = metrics.f1_score(
                            golds == i, preds == i, average="binary", zero_division=0
                        )
                        class_f1s[label_name] = class_f1
        
        return f1, dev_loss, class_f1s
