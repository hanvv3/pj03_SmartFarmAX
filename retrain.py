"""retrain.py — Optuna Best-Trial 재학습 스크립트

Optuna DB에서 best trial 파라미터를 읽어 full training을 실행하고
best_model.pt 를 저장합니다.
Jupyter 노트북에서 직접 학습하면 CUDA OOM이 발생할 수 있으므로
이 스크립트를 터미널에서 실행하세요.

사용법:
    python retrain.py
    python retrain.py --epochs 10
    python retrain.py --backbone resnet50
    python retrain.py --backbone convnext_small --epochs 5 --gpu 1
    python retrain.py --batch-size 2 --lr 1e-4 --wd 5e-4

저장 경로:
    runs/strawberry_detection_optuna/best_model.pt
"""

import argparse
import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from rich.table import Table
from rich import print as rprint

import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import boxes as box_ops

# ─── 경로 상수 ───────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent.resolve()
OUTPUT_DIR      = PROJECT_ROOT / 'runs' / 'lettuce_detection_optuna'
TORCH_CACHE_DIR = OUTPUT_DIR / 'torch_cache'
STUDY_DB_PATH   = OUTPUT_DIR / 'optuna_study.db'
BEST_MODEL_PATH = OUTPUT_DIR / 'best_model.pt'
EXTRACT_ROOT    = PROJECT_ROOT / 'data' / '_extracted_071'

STUDY_NAME = 'lettuce_detection_optuna'

CLASS_NAME_BY_DISEASE = {
    0: 'normal',
    9: 'sclerotinia_rot',
    10: 'downy_mildew'
}

IMAGE_MIN_SIZE = 640
IMAGE_MAX_SIZE = 640

# ── 재다운로드 방지: torch hub 캐시를 프로젝트 내로 고정 ──
os.environ['TORCH_HOME'] = str(TORCH_CACHE_DIR.resolve())

warnings.filterwarnings('ignore')

console = Console()
RICH_REFRESH_PER_SECOND = 8


# ─────────────────────────────────────────────────────────
# 데이터셋
# ─────────────────────────────────────────────────────────

@dataclass
class Record:
    image_path: Path
    label_path: Path
    disease_id: int
    boxes: list
    source_name: str


def build_image_index(image_dir: Path) -> dict:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    index: dict[str, Path] = {}
    for p in image_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            index.setdefault(p.name, p)
    return index


def parse_record(label_path: Path, image_index: dict, source_name: str):
    with label_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    description = data.get('description', {})
    annotations = data.get('annotations', {})
    points      = annotations.get('points', []) or []
    disease_id  = int(annotations.get('disease', 0))
    image_name  = description.get('image')
    if not image_name:
        return None
    image_path = image_index.get(image_name)
    if image_path is None:
        return None
    boxes = []
    for pt in points:
        xtl, ytl, xbr, ybr = float(pt['xtl']), float(pt['ytl']), float(pt['xbr']), float(pt['ybr'])
        if xbr > xtl and ybr > ytl:
            boxes.append([xtl, ytl, xbr, ybr])
    return Record(image_path=image_path, label_path=label_path,
                  disease_id=disease_id, boxes=boxes, source_name=source_name)


def collect_records(sources: list) -> list:
    records, missing = [], []
    for source in sources:
        image_dir   = Path(source['image_dir'])
        label_dir   = Path(source['label_dir'])
        source_name = source['name']
        image_index = build_image_index(image_dir)
        json_files  = sorted(label_dir.rglob('*.json')) if label_dir.exists() else []
        for lp in json_files:
            rec = parse_record(lp, image_index, source_name)
            (missing if rec is None else records).append(rec if rec else str(lp))
        usable = sum(1 for r in records if isinstance(r, Record) and r.source_name == source_name)
        print(f'  {source_name}: images={len(image_index)}, labels={len(json_files)}, usable={usable}')
    if missing:
        print(f'  missing references: {len(missing)}')
    return [r for r in records if isinstance(r, Record)]


def build_class_mapping(records: list, class_name_by_disease: dict):
    disease_ids      = sorted({r.disease_id for r in records})
    disease_to_label = {did: idx + 1 for idx, did in enumerate(disease_ids)}
    label_to_name    = {
        label: class_name_by_disease.get(did, f'disease_{did}')
        for did, label in disease_to_label.items()
    }
    return disease_to_label, label_to_name


class JsonDetectionDataset(Dataset):
    def __init__(self, records: list, disease_to_label: dict):
        self.records          = records
        self.disease_to_label = disease_to_label

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        image = Image.open(rec.image_path).convert('RGB')
        w, h  = image.size
        boxes = torch.tensor(rec.boxes, dtype=torch.float32) if rec.boxes \
                else torch.zeros((0, 4), dtype=torch.float32)
        if len(boxes):
            boxes = box_ops.clip_boxes_to_image(boxes, (h, w))
        label_value = self.disease_to_label[rec.disease_id]
        labels  = torch.full((len(boxes),), label_value, dtype=torch.int64) if len(boxes) \
                  else torch.zeros((0,), dtype=torch.int64)
        area    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) \
                  else torch.zeros((0,), dtype=torch.float32)
        target  = {
            'boxes': boxes, 'labels': labels,
            'image_id': torch.tensor([idx], dtype=torch.int64),
            'area': area,
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64),
        }
        return TF.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


# ─────────────────────────────────────────────────────────
# 모델 빌드
# ─────────────────────────────────────────────────────────

def build_resnet50_fasterrcnn(num_classes: int) -> FasterRCNN:
    try:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights='DEFAULT', trainable_backbone_layers=3,
            min_size=IMAGE_MIN_SIZE, max_size=IMAGE_MAX_SIZE,
        )
    except Exception as e:
        print(f'  pretrained 로드 실패 ({e}), 랜덤 초기화')
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None,
            min_size=IMAGE_MIN_SIZE, max_size=IMAGE_MAX_SIZE,
        )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_convnext_fasterrcnn(num_classes: int, variant: str = 'small') -> FasterRCNN:
    from torchvision.models import convnext_tiny, convnext_small
    from torchvision.models.detection.backbone_utils import BackboneWithFPN

    builders = {
        'tiny':  (lambda: convnext_tiny(weights='DEFAULT').features,
                  lambda: convnext_tiny(weights=None).features),
        'small': (lambda: convnext_small(weights='DEFAULT').features,
                  lambda: convnext_small(weights=None).features),
    }
    if variant not in builders:
        raise ValueError(f'지원하지 않는 variant: {variant}')

    pretrained_fn, fallback_fn = builders[variant]
    try:
        backbone_body = pretrained_fn()
    except Exception as e:
        print(f'  pretrained 로드 실패 ({e}), 랜덤 초기화')
        backbone_body = fallback_fn()

    # Freeze early stages for faster fine-tuning
    # 0:Stem 1:Stage1 2:DS 3:Stage2 4:DS 5:Stage3(27×blocks) 6:DS 7:Stage4
    freeze_stages = [0, 1, 2, 3, 4, 5]
    for stage_idx in freeze_stages:
        for param in backbone_body[stage_idx].parameters():
            param.requires_grad = False

    backbone = BackboneWithFPN(
        backbone=backbone_body,
        return_layers={'3': '0', '5': '1', '7': '2'},
        in_channels_list=[192, 384, 768],
        out_channels=256,
    )
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),   # FPN 출력 4개 ('0','1','2','pool')와 일치
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', 'pool'],
        output_size=7, sampling_ratio=2,
    )
    return FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=IMAGE_MIN_SIZE,
        max_size=IMAGE_MAX_SIZE,
    )


def build_model(backbone_name: str, num_classes: int) -> FasterRCNN:
    if backbone_name == 'resnet50':
        return build_resnet50_fasterrcnn(num_classes)
    if backbone_name in ('convnext_tiny', 'convnext_small'):
        return build_convnext_fasterrcnn(num_classes, variant=backbone_name.split('_')[1])
    raise ValueError(f'지원하지 않는 backbone: {backbone_name}')


# ─────────────────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    running = 0.0
    n = len(loader)

    progress = Progress(
        SpinnerColumn(),
        TextColumn(f"[bold cyan]Train[/bold cyan] epoch [cyan]{epoch:02d}/{total_epochs:02d}[/cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("loss=[bold]{task.fields[loss]}[/bold]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=RICH_REFRESH_PER_SECOND,
        transient=True,
    )
    with progress:
        task = progress.add_task("train", total=n, loss="-.----")
        for step, (images, targets) in enumerate(loader, 1):
            images  = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at epoch={epoch}, step={step}: {loss_dict}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
            running += loss_value
            progress.update(task, advance=1, loss=f"{loss_value:.4f}")

    return running / max(n, 1)


@torch.no_grad()
def eval_val_loss(model, loader, device):
    model.train()   # FasterRCNN: loss 계산은 train mode 필요
    total = 0.0
    component_sums: dict[str, float] = {}

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]Valid [/bold magenta]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("loss=[bold]{task.fields[loss]}[/bold]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=RICH_REFRESH_PER_SECOND,
        transient=True,
    )
    with progress:
        task = progress.add_task("valid", total=len(loader), loss="-.----")
        for images, targets in loader:
            images  = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss_value = float(sum(loss_dict.values()).item())
            total += loss_value
            for k, v in loss_dict.items():
                component_sums[k] = component_sums.get(k, 0.0) + float(v.item())
            progress.update(task, advance=1, loss=f"{loss_value:.4f}")

    n = max(len(loader), 1)
    avg_components = {k: v / n for k, v in component_sums.items()}
    return total / n, avg_components


def run_training(model, train_loader, val_loader, optimizer, scheduler,
                 device, epochs, writer=None):
    best_val_loss = float('inf')
    best_state    = None
    history       = []
    global_step   = 0

    # epoch 진행 상황을 한 줄 요약 테이블로 출력
    summary = Table(
        "Epoch", "Train Loss", "Val Loss", "LR", "Best",
        title="[bold]Training Summary[/bold]",
        show_lines=False,
        style="dim",
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, epochs
        )
        val_loss, val_components = eval_val_loss(model, val_loader, device)

        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': val_loss, 'lr': lr})

        if writer is not None:
            writer.add_scalar('epoch/train_loss', train_loss, epoch)
            writer.add_scalar('epoch/val_loss',   val_loss,   epoch)
            writer.add_scalar('epoch/lr',         lr,         epoch)
            for k, v in val_components.items():
                writer.add_scalar(f'val_components/{k}', v, epoch)
            if is_best:
                writer.add_scalar('best/val_loss', best_val_loss, epoch)

        best_mark = "[bold yellow]★ best[/bold yellow]" if is_best else ""
        summary.add_row(
            f"{epoch:02d}/{epochs:02d}",
            f"{train_loss:.4f}",
            f"[bold]{val_loss:.4f}[/bold]" if is_best else f"{val_loss:.4f}",
            f"{lr:.3e}",
            best_mark,
        )
        console.print(summary)

    return best_state, best_val_loss


# ─────────────────────────────────────────────────────────
# Optuna DB에서 best params 읽기
# ─────────────────────────────────────────────────────────

def load_best_optuna_params() -> dict | None:
    if not STUDY_DB_PATH.exists():
        return None
    import optuna
    storage_url = f'sqlite:///{STUDY_DB_PATH.resolve().as_posix()}'
    try:
        study     = optuna.load_study(study_name=STUDY_NAME, storage=storage_url)
        completed = [t for t in study.trials if str(t.state) == 'TrialState.COMPLETE']
        if not completed:
            return None
        best = min(completed, key=lambda t: t.value)
        print(f'Optuna best → trial#{best.number}  val_loss={best.value:.4f}  params={best.params}')
        return best.params
    except Exception as e:
        print(f'Optuna DB 읽기 실패: {e}')
        return None


# ─────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Best Optuna params로 재학습 후 best_model.pt 저장')
    parser.add_argument('--epochs',    type=int,   default=None,
                        help='학습 epoch 수 (미지정 시 Optuna SEARCH_EPOCHS=3 사용)')
    parser.add_argument('--backbone',  type=str,   default=None,
                        choices=['resnet50', 'convnext_tiny', 'convnext_small'],
                        help='backbone 강제 지정 (미지정 시 Optuna best 사용)')
    parser.add_argument('--batch-size', type=int,  default=None,
                        help='batch size 강제 지정 (미지정 시 Optuna best 사용)')
    parser.add_argument('--lr',         type=float, default=None,
                        help='learning rate 강제 지정 (미지정 시 Optuna best 사용)')
    parser.add_argument('--wd',         type=float, default=None,
                        help='weight decay 강제 지정 (미지정 시 Optuna best 사용)')
    parser.add_argument('--gpu',        type=int,   default=0,
                        help='사용할 CUDA GPU 번호 (기본: 0)')
    parser.add_argument('--num-gpus',   type=int,   default=1,
                        help='사용할 GPU 수 (DataParallel, 기본: 1)')
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='TensorBoard 로깅 비활성화')
    _default_workers = min(8, (os.cpu_count() or 1) // 2)
    parser.add_argument('--num-workers', type=int, default=_default_workers,
                        help=f'DataLoader worker 수 (기본: {_default_workers}, cpu_count={os.cpu_count()})')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 재현성 ──
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── GPU ──
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    else:
        device = torch.device('cpu')
    console.print(f'[bold]device:[/bold] {device}')
    console.print(f'[bold]torch_home:[/bold] {os.environ["TORCH_HOME"]}')

    # ── Optuna best params 로드 ──
    console.rule('[bold cyan][1] Optuna DB 확인[/bold cyan]')
    optuna_params = load_best_optuna_params()

    # CLI 우선, 없으면 Optuna best, 없으면 기본값
    backbone   = args.backbone   or (optuna_params or {}).get('backbone',    'convnext_small')
    batch_size = args.batch_size or (optuna_params or {}).get('batch_size',  2)
    lr         = args.lr         or (optuna_params or {}).get('learning_rate', 1e-4)
    wd         = args.wd         or (optuna_params or {}).get('weight_decay',  1e-4)
    epochs     = args.epochs     or 5

    param_table = Table(title='[bold]학습 파라미터[/bold]', show_header=False, style='cyan')
    param_table.add_column('key',   style='bold')
    param_table.add_column('value')
    param_table.add_row('backbone',   backbone)
    param_table.add_row('batch_size', str(batch_size))
    param_table.add_row('lr',         f'{lr:.4e}')
    param_table.add_row('wd',         f'{wd:.4e}')
    param_table.add_row('epochs',     str(epochs))
    param_table.add_row('gpu',        str(args.gpu))
    param_table.add_row('num_workers', str(args.num_workers))
    console.print(param_table)

    # ── 데이터 ──
    console.rule('[bold cyan][2] 데이터 로드[/bold cyan]')
    train_sources = [{'name': 'train_all',
                      'image_dir': EXTRACT_ROOT / 'train' / 'images',
                      'label_dir': EXTRACT_ROOT / 'train' / 'labels'}]
    val_sources   = [{'name': 'val_all',
                      'image_dir': EXTRACT_ROOT / 'val' / 'images',
                      'label_dir': EXTRACT_ROOT / 'val' / 'labels'}]

    train_records = collect_records(train_sources)
    val_records   = collect_records(val_sources)
    all_records   = train_records + val_records

    disease_to_label, label_to_name = build_class_mapping(all_records, CLASS_NAME_BY_DISEASE)
    num_classes = len(label_to_name) + 1

    console.print(f'  train=[bold]{len(train_records)}[/bold]  val=[bold]{len(val_records)}[/bold]  '
                  f'num_classes=[bold]{num_classes}[/bold]  label_to_name={label_to_name}')

    train_dataset = JsonDetectionDataset(train_records, disease_to_label)
    val_dataset   = JsonDetectionDataset(val_records,   disease_to_label)

    # .py 스크립트: num_workers>0 안전 (fork 방식, Jupyter OOM/BrokenPipe 없음)
    # prefetch_factor=2: GPU 연산 중 CPU가 다음 배치를 미리 준비 (파이프라이닝)
    num_workers = args.num_workers
    console.print(f'  num_workers = [bold]{num_workers}[/bold]  '
                  f'(cpu_count={os.cpu_count()}, prefetch_factor=2)')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0,
                              prefetch_factor=2 if num_workers > 0 else None,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0,
                              prefetch_factor=2 if num_workers > 0 else None,
                              collate_fn=collate_fn)

    # ── 모델 ──
    console.rule(f'[bold cyan][3] 모델 빌드: {backbone}[/bold cyan]')
    model = build_model(backbone, num_classes=num_classes)

    # DataParallel (--num-gpus > 1)
    gpu_count = min(args.num_gpus, torch.cuda.device_count())
    if gpu_count > 1:
        device_ids = list(range(gpu_count))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        console.print(f'  [green]DataParallel[/green]: {device_ids}')

    model = model.to(device)

    # ── Optimizer / Scheduler ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── TensorBoard ──
    writer = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = OUTPUT_DIR / 'retrain_tb'
            writer = SummaryWriter(log_dir=str(tb_dir))
            console.print(f'  TensorBoard: [dim]{tb_dir}[/dim]')
        except ImportError:
            console.print('  [yellow]TensorBoard 없음 (건너뜀)[/yellow]')

    # ── 학습 ──
    console.rule(f'[bold cyan][4] 학습 시작 ({epochs} epochs)[/bold cyan]')
    best_state, best_val_loss = run_training(
        model, train_loader, val_loader, optimizer, scheduler, device, epochs,
        writer=writer,
    )

    # ── 저장 ──
    console.rule('[bold cyan][5] 저장[/bold cyan]')
    console.print(f'  [bold]best_model.pt[/bold]: {BEST_MODEL_PATH}')
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # DataParallel이면 내부 모델 state_dict 저장
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model_to_save.state_dict().items()}

    ckpt = {
        'model_state_dict': best_state,
        'best_params': {
            'backbone':      backbone,
            'batch_size':    batch_size,
            'learning_rate': lr,
            'weight_decay':  wd,
        },
        'best_val_loss':    best_val_loss,
        'class_names':      label_to_name,
        'disease_to_label': disease_to_label,
        'num_classes':      num_classes,
        'epochs':           epochs,
    }
    torch.save(ckpt, BEST_MODEL_PATH)
    console.print(f'  best_val_loss = [bold green]{best_val_loss:.4f}[/bold green]')
    console.print('  [bold green]저장 완료![/bold green]')

    if writer is not None:
        writer.close()

    console.rule('[bold green]완료[/bold green]')
    console.print('test.ipynb 셀 5를 다시 실행하면 [bold]best_model.pt[/bold]가 자동으로 로드됩니다.')


if __name__ == '__main__':
    main()
