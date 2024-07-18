import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR



class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe



def multi_scale_loss(pred_flows, gt_flow):
    total_loss = 0
    weights = [1.0, 0.5, 0.3, 0.1]  # 各スケールの重み、上から最終的なフロー解像度、1/2スケール、1/4スケール、1/8スケール
    
    for i, (key,pred_flow) in enumerate(pred_flows.items()):
        weight = weights[i]
        if i > 0:
            # スケールに応じてground truthをダウンサンプリング
            scale_factor = 1 / (2 ** i)
            scaled_gt = F.interpolate(gt_flow, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        else:
            scaled_gt = gt_flow
        
        # 予測フローのサイズをground truthに合わせる
        pred_flow = F.interpolate(pred_flow, size=scaled_gt.shape[2:], mode='bilinear', align_corners=False)
        
        # EPE損失の計算
        loss = compute_epe_error(pred_flow, scaled_gt)
        total_loss += weight * loss
    
    return total_loss


def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        '''
    
    # ------------------
    #    Dataloader
    # ------------------
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    train_data = DataLoader(train_set,
                                 batch_size=args.data_loader.train.batch_size,
                                 shuffle=args.data_loader.train.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    test_data = DataLoader(test_set,
                                 batch_size=args.data_loader.test.batch_size,
                                 shuffle=args.data_loader.test.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)

    '''
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない
    
    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    '''
    # ------------------
    #       Model
    # ------------------
    model = EVFlowNet(args.train).to(device)

    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    

    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_data), epochs=args.train.epochs)

    # ------------------
    #   Start training
    # ------------------
    model.train()
    for epoch in range(args.train.epochs):
        total_loss = 0
        print("on epoch: {}".format(epoch+1))
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device) # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]

            # モデルの出力が複数のフロー予測を返すように変更
            flow, flow_pred_dict = model(event_image)  # flowとflow_pred_dictを返す        
            # 多スケール損失の計算
            loss: torch.Tensor = multi_scale_loss(flow_pred_dict, ground_truth_flow)

            print(f"batch {i} loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
        

        # Create the directory if it doesn't exist
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

        # エポックごとのlossをcsvファイルに保存
        with open("loss.csv", "a") as f:
            f.write(f"{epoch+1},{total_loss / len(train_data)}\n")


        # 2エポックごとにモデルを保存
        if (epoch + 1) % 1 == 0:        
            model_path = "checkpoints/model_batch16_changeRate_dropout_epoch{}.pth".format(epoch+1)
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    

    # ------------------
    #   Start predicting
    # ------------------
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            batch_flow,_ = model(event_image) # [1, 2, 480, 640]  # flowのみを使用
            flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
        print("test done")
    # ------------------
    #  save submission
    # ------------------
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()
