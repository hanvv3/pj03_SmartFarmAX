import copy
import json
import math
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import optuna
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import boxes as box_ops

from datetime import datetime
from contextlib import nullcontext

from torch.utils.tensorboard import SummaryWriter

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

print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('optuna:', optuna.__version__)

USE_TENSORBOARD = True
TB_ROOT = Path("runs/fasterrcnn_optuna")

USE_RICH_PROGRESS = True
RICH_REFRESH_PER_SECOND = 8
console = Console()

SEED = 42
AVAILABLE_GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
MAX_TRAIN_GPUS = 1
GPU_COUNT = min(AVAILABLE_GPU_COUNT, MAX_TRAIN_GPUS)
DEVICE = torch.device('cuda:0' if GPU_COUNT > 0 else 'cpu')
DATA_PARALLEL_DEVICE_IDS = list(range(GPU_COUNT)) if GPU_COUNT > 1 else []
USE_DATA_PARALLEL = len(DATA_PARALLEL_DEVICE_IDS) > 1

# Jupyter 노트북에서 DataLoader 멀티프로세싱 방식 비교:
#
#   NUM_WORKERS=0 (메인 프로세스 직접 로딩) ← 현재 설정
#     - 별도 worker 프로세스 없이 메인 프로세스에서 직접 데이터를 로드
#     - multiprocessing 문제(fork/spawn/forkserver) 완전 회피
#     - FasterRCNN + 640×640 처럼 GPU 연산이 압도적 병목인 경우 성능 차이 미미
#       (GPU가 연산하는 동안 CPU는 다음 배치를 미리 준비하기엔 이미 빠름)
#   NUM_WORKERS>0 이슈:
#     - fork   : CUDA context 복사 → Segfault / 커널 충돌
#     - forkserver: Jupyter에서 forkserver 데몬 시작 실패 → BrokenPipeError
#     - spawn  : Jupyter __main__을 새 프로세스에서 재import 불가
#               → AttributeError: Can't get attribute 'JsonDetectionDataset'
NUM_WORKERS = 0
MULTIPROCESSING_CONTEXT = None  # NUM_WORKERS=0 이면 불필요

# CUDA pinned memory: Host → GPU 메모리 복사 속도 향상 (cudaMallocHost 사용)
PIN_MEMORY = torch.cuda.is_available()

# 비동기 Host→Device 전송 활성화 (pin_memory=True와 함께 사용)
NON_BLOCKING = torch.cuda.is_available()

if torch.cuda.is_available():
    # 입력 해상도가 고정된 경우 cuDNN이 최적 알고리즘을 캐싱 → 반복 실행 속도 향상
    torch.backends.cudnn.benchmark = True

    # RTX 4090 (Ada Lovelace) Tensor Core: TF32로 행렬곱 및 컨볼루션을 가속
    # FP32 대비 속도 ~3×, 정밀도 손실은 미미 (학습에 적합)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # FP32 행렬곱 정밀도 완화: 'high' = TF32 텐서코어 활용 극대화
    torch.set_float32_matmul_precision('high')

# =========================
# 1) AIHub 압축 데이터 경로 설정
# =========================
AIHUB_ROOT = Path('data/071.시설 작물 질병 진단/01.데이터')
TRAIN_RAW_ROOT = AIHUB_ROOT / '1.Training'
VAL_RAW_ROOT = AIHUB_ROOT / '2.Validation'

EXTRACT_ROOT = Path('data/_extracted_071')
AUTO_EXTRACT_ZIP = True

def make_run_dir(prefix: str, trial_number: int | None = None):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if trial_number is None:
        run_name = f"{prefix}_{timestamp}"
    else:
        run_name = f"{prefix}_trial{trial_number:03d}_{timestamp}"
    run_dir = TB_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def format_hparams(backbone, batch_size, learning_rate, weight_decay, epochs):
    return {
        "backbone": backbone,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "max_train_gpus": MAX_TRAIN_GPUS,
        "image_min_size": IMAGE_MIN_SIZE,
        "image_max_size": IMAGE_MAX_SIZE,
    }

def _zip_stem(zip_path: Path):
    return zip_path.name[:-4] if zip_path.name.lower().endswith('.zip') else zip_path.stem


def _extract_zip_once(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    done_flag = out_dir / '.done'
    if done_flag.exists():
        return False

    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = [member for member in zf.infolist() if not member.is_dir()]
        extracted = False

        for member in members:
            target_path = out_dir / member.filename
            if target_path.exists():
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target_path.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            extracted = True

    done_flag.write_text('ok', encoding='utf-8')
    return extracted


def extract_split_archives(split_raw_root: Path, split_name: str):
    label_zip_root = split_raw_root / '라벨링데이터' / '04.딸기'
    image_zip_root = split_raw_root / '원천데이터' / '04.딸기'

    if not label_zip_root.exists() or not image_zip_root.exists():
        raise FileNotFoundError(f'압축 폴더를 찾을 수 없습니다: {split_raw_root}')

    split_extract_root = EXTRACT_ROOT / split_name
    image_out_root = split_extract_root / 'images'
    label_out_root = split_extract_root / 'labels'

    image_zip_files = sorted(image_zip_root.glob('*.zip'))
    label_zip_files = sorted(label_zip_root.glob('*.zip'))

    extracted_count = 0

    for zip_path in image_zip_files:
        extracted_count += int(_extract_zip_once(zip_path, image_out_root / _zip_stem(zip_path)))

    for zip_path in label_zip_files:
        extracted_count += int(_extract_zip_once(zip_path, label_out_root / _zip_stem(zip_path)))

    print(f'[{split_name}] image_zips={len(image_zip_files)} label_zips={len(label_zip_files)} newly_extracted={extracted_count}')

    return {
        'name': f'{split_name}_all',
        'image_dir': image_out_root,
        'label_dir': label_out_root,
    }


def build_sources(auto_extract: bool = True):
    if auto_extract:
        train_source = extract_split_archives(TRAIN_RAW_ROOT, 'train')
        val_source = extract_split_archives(VAL_RAW_ROOT, 'val')
    else:
        train_source = {
            'name': 'train_all',
            'image_dir': EXTRACT_ROOT / 'train' / 'images',
            'label_dir': EXTRACT_ROOT / 'train' / 'labels',
        }
        val_source = {
            'name': 'val_all',
            'image_dir': EXTRACT_ROOT / 'val' / 'images',
            'label_dir': EXTRACT_ROOT / 'val' / 'labels',
        }

    return [train_source], [val_source]


TRAIN_SOURCES, VAL_SOURCES = build_sources(auto_extract=AUTO_EXTRACT_ZIP)

# disease id -> display name. 비워두면 disease_숫자 형태로 자동 생성됩니다.
CLASS_NAME_BY_DISEASE = {
    0: 'normal',
    7: 'gray_mold',
    8: 'powdery_mildew'
}

# =========================
# 2) Optuna / 학습 설정
# =========================
# 탐색할 백본 목록
# - resnet50       : 안정적인 기준선, FPN 내장 pretrained 가중치 제공
# - convnext_small : RTX 4090에서 ResNet50보다 TF32 가속 효율이 높고 정확도도 우수
SEARCH_BACKBONES = ['resnet50', 'convnext_small']
SEARCH_TRIALS = 12
SEARCH_EPOCHS = 3
FINAL_EPOCHS = 15

# ─── 배치 크기 선택 가이드 (FasterRCNN + 640×640, RTX 4090 24 GB 기준) ───
# FasterRCNN은 RPN proposals + RoI 풀링 때문에 일반 분류 모델보다 VRAM 소모가 큽니다.
# FP32 기준 이미지 1장당 약 2~3 GB (backbone feature map + gradient 포함)
#
#   global_batch │ GPU당 이미지 수 │ GPU당 예상 VRAM │ 안전 여부
#   ─────────────┼─────────────────┼─────────────────┼──────────
#        2       │        1        │    ~3 GB        │ ✓ 안전
#        4       │        2        │    ~5 GB        │ ✓ 안전
#        8       │        4        │   ~10 GB        │ ✓ 안전
#       16       │        8        │   ~20 GB        │ △ 아슬
#       32       │       16        │   ~40 GB        │ ✗ OOM → 커널 충돌


BATCH_SIZE_OPTIONS = [2, 4]

IMAGE_MIN_SIZE = 640
IMAGE_MAX_SIZE = 640
USE_PRETRAINED_WEIGHTS = True

STUDY_NAME = 'strawberry_detection_optuna'
OUTPUT_DIR = Path('runs') / STUDY_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STUDY_DB_PATH = OUTPUT_DIR / 'optuna_study.db'
TORCH_CACHE_DIR = OUTPUT_DIR / 'torch_cache'
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('TORCH_HOME', str(TORCH_CACHE_DIR.resolve()))

print('device:', DEVICE)
print('available_gpu_count:', AVAILABLE_GPU_COUNT)
print('active_gpu_count:', GPU_COUNT)
if AVAILABLE_GPU_COUNT > MAX_TRAIN_GPUS:
    print(f'gpu usage capped to first {MAX_TRAIN_GPUS} devices.')
if AVAILABLE_GPU_COUNT:
    for gpu_idx in range(AVAILABLE_GPU_COUNT):
        print(f'gpu[{gpu_idx}]:', torch.cuda.get_device_name(gpu_idx))
print('data_parallel:', USE_DATA_PARALLEL, DATA_PARALLEL_DEVICE_IDS if USE_DATA_PARALLEL else 'disabled')
print('num_workers:', NUM_WORKERS)
print('tf32_enabled:', torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False)
print('torch_home:', os.environ['TORCH_HOME'])
print('train_sources:', TRAIN_SOURCES)
print('val_sources:', VAL_SOURCES)
print('output:', OUTPUT_DIR.resolve())

# =========================
# 3) JSON 파서 / 데이터셋
# =========================
@dataclass
class Record:
    image_path: Path
    label_path: Path
    disease_id: int
    boxes: list
    source_name: str
    is_augmented: bool
    original_name: str | None


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def list_json_files(label_dir: Path):
    if not label_dir.exists():
        return []
    return sorted(label_dir.rglob('*.json'))


def build_image_index(image_dir: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_index = {}
    duplicate_count = 0

    for image_path in image_dir.rglob('*'):
        if image_path.is_file() and image_path.suffix.lower() in exts:
            key = image_path.name
            if key in image_index:
                duplicate_count += 1
                continue
            image_index[key] = image_path

    if duplicate_count:
        print(f'warning: duplicated image names ignored={duplicate_count}')

    return image_index


def parse_record(label_path: Path, image_index: dict, source_name: str):
    data = load_json(label_path)
    description = data.get('description', {})
    annotations = data.get('annotations', {})
    points = annotations.get('points', []) or []
    disease_id = int(annotations.get('disease', 0))
    image_name = description.get('image')

    if not image_name:
        return None

    image_path = image_index.get(image_name)
    if image_path is None:
        return None

    boxes = []
    for point in points:
        xtl = float(point['xtl'])
        ytl = float(point['ytl'])
        xbr = float(point['xbr'])
        ybr = float(point['ybr'])
        if xbr > xtl and ybr > ytl:
            boxes.append([xtl, ytl, xbr, ybr])

    return Record(
        image_path=image_path,
        label_path=label_path,
        disease_id=disease_id,
        boxes=boxes,
        source_name=source_name,
        is_augmented='augmented' in data,
        original_name=description.get('original'),
    )


def collect_records(sources):
    records = []
    missing_images = []

    for source in sources:
        image_dir = Path(source['image_dir'])
        label_dir = Path(source['label_dir'])
        source_name = source['name']

        image_index = build_image_index(image_dir)
        json_files = list_json_files(label_dir)

        for label_path in json_files:
            record = parse_record(label_path, image_index=image_index, source_name=source_name)
            if record is None:
                missing_images.append(str(label_path))
                continue
            records.append(record)

        source_usable = sum(1 for record in records if record.source_name == source_name)
        print(f'{source_name}: images={len(image_index)}, labels={len(json_files)}, usable={source_usable}')

    if missing_images:
        print(f'missing image references: {len(missing_images)}')
        print('first 5 missing examples:')
        for item in missing_images[:5]:
            print(' -', item)

    return records


def build_class_mapping(records, class_name_by_disease):
    disease_ids = sorted({record.disease_id for record in records})
    disease_to_label = {disease_id: idx + 1 for idx, disease_id in enumerate(disease_ids)}
    label_to_name = {
        label: class_name_by_disease.get(disease_id, f'disease_{disease_id}')
        for disease_id, label in disease_to_label.items()
    }
    return disease_to_label, label_to_name


class JsonDetectionDataset(Dataset):
    def __init__(self, records, disease_to_label):
        self.records = records
        self.disease_to_label = disease_to_label

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record.image_path).convert('RGB')
        width, height = image.size

        boxes = torch.tensor(record.boxes, dtype=torch.float32) if record.boxes else torch.zeros((0, 4), dtype=torch.float32)
        if len(boxes):
            boxes = box_ops.clip_boxes_to_image(boxes, (height, width))

        label_value = self.disease_to_label[record.disease_id]
        labels = torch.full((len(boxes),), label_value, dtype=torch.int64) if len(boxes) else torch.zeros((0,), dtype=torch.int64)

        image_tensor = torchvision.transforms.functional.to_tensor(image)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else torch.zeros((0,), dtype=torch.float32)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx], dtype=torch.int64),
            'area': area,
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def prepare_datasets():
    train_records = collect_records(TRAIN_SOURCES)
    val_records = collect_records(VAL_SOURCES)

    if not train_records:
        raise ValueError('No train records found. TRAIN_SOURCES 경로를 확인하세요.')
    if not val_records:
        raise ValueError('No validation records found. VAL_SOURCES 경로를 확인하세요.')

    disease_to_label, label_to_name = build_class_mapping(train_records + val_records, CLASS_NAME_BY_DISEASE)
    train_dataset = JsonDetectionDataset(train_records, disease_to_label)
    val_dataset = JsonDetectionDataset(val_records, disease_to_label)

    print('train samples:', len(train_dataset))
    print('val samples:', len(val_dataset))
    print('classes:', {label: name for label, name in sorted(label_to_name.items())})

    return train_dataset, val_dataset, disease_to_label, label_to_name

# =========================
# 4) 모델 / 학습 함수
# =========================
class DetectionDataParallel(nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None):
        super().__init__(module, device_ids=device_ids, output_device=output_device)
        self.last_chunk_sizes = []

    # torchvision detection 모델은 list[Tensor], list[dict] 입력을 쓰므로
    # 기본 DataParallel scatter 대신 샘플 단위로 GPU에 분배합니다.
    def scatter(self, inputs, kwargs, device_ids):
        if not inputs:
            return (), ()
        if len(inputs) != 2:
            raise ValueError('DetectionDataParallel expects (images, targets).')

        images, targets = inputs
        images = list(images)
        targets = list(targets) if targets is not None else None

        if targets is not None and len(images) != len(targets):
            raise ValueError('images and targets must have the same batch size.')
        if not images:
            return (), ()

        num_chunks = min(len(device_ids), len(images))
        chunk_size = math.ceil(len(images) / num_chunks)
        scattered_inputs = []
        self.last_chunk_sizes = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(len(images), start + chunk_size)
            if start >= end:
                continue

            device = torch.device(f'cuda:{device_ids[chunk_idx]}')
            image_chunk = [image.to(device, non_blocking=NON_BLOCKING) for image in images[start:end]]
            self.last_chunk_sizes.append(len(image_chunk))

            if targets is None:
                scattered_inputs.append((image_chunk,))
                continue

            target_chunk = [
                {key: value.to(device, non_blocking=NON_BLOCKING) for key, value in target.items()}
                for target in targets[start:end]
            ]
            scattered_inputs.append((image_chunk, target_chunk))

        scattered_kwargs = tuple({} for _ in scattered_inputs)
        return tuple(scattered_inputs), scattered_kwargs


def build_with_weight_fallback(model_name, pretrained_builder, fallback_builder):
    if not USE_PRETRAINED_WEIGHTS:
        return fallback_builder()

    try:
        return pretrained_builder()
    except Exception as exc:
        print(f'warning: failed to load pretrained weights for {model_name}: {exc}')
        print('falling back to randomly initialized weights.')
        return fallback_builder()


def build_resnet50_fasterrcnn(num_classes: int):
    model = build_with_weight_fallback(
        model_name='fasterrcnn_resnet50_fpn',
        pretrained_builder=lambda: torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights='DEFAULT',
            trainable_backbone_layers=3,
            min_size=IMAGE_MIN_SIZE,
            max_size=IMAGE_MAX_SIZE,
        ),
        fallback_builder=lambda: torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=IMAGE_MIN_SIZE,
            max_size=IMAGE_MAX_SIZE,
        ),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_convnext_fasterrcnn(num_classes: int, variant: str = 'small'):
    """ConvNeXt 백본을 FPN + Faster R-CNN 헤드에 연결합니다.

    variant: 'tiny' | 'small'

    FPN feature map 구성:
        BackboneWithFPN은 기본적으로 LastLevelMaxPool extra_blocks를 추가합니다.
        return_layers 3개 → FPN 출력 4개: '0', '1', '2', 'pool'

        따라서 AnchorGenerator의 sizes/aspect_ratios도 반드시 4개여야 합니다.
        (5개로 설정하면 AssertionError: match between feature maps and sizes 발생)

        ConvNeXt features 인덱스 구조 (tiny/small 공통):
          0: Stem       1: Stage1   2: Stage2(C2)
          3: Downsample 4: Stage3(C3)
          5: Downsample 6: Stage4(C4)
        return_layers = {'2': '0', '4': '1', '6': '2'}
        in_channels = [192, 384, 768]
    """
    from torchvision.models import convnext_tiny, convnext_small
    from torchvision.models.detection.backbone_utils import BackboneWithFPN

    builders = {
        'tiny': (
            lambda: convnext_tiny(weights='DEFAULT').features,
            lambda: convnext_tiny(weights=None).features,
        ),
        'small': (
            lambda: convnext_small(weights='DEFAULT').features,
            lambda: convnext_small(weights=None).features,
        ),
    }
    if variant not in builders:
        raise ValueError(f'Unsupported convnext variant: {variant}. Choose from {list(builders)}')

    pretrained_builder, fallback_builder = builders[variant]
    backbone_body = build_with_weight_fallback(
        model_name=f'convnext_{variant}',
        pretrained_builder=pretrained_builder,
        fallback_builder=fallback_builder,
    )

    # BackboneWithFPN: return_layers 3개 + LastLevelMaxPool → 출력 4개 ('0','1','2','pool')
    backbone = BackboneWithFPN(
        backbone=backbone_body,
        return_layers={'2': '0', '4': '1', '6': '2'},
        in_channels_list=[192, 384, 768],
        out_channels=256,
    )

    # sizes 개수 = feature map 개수(4)와 반드시 일치해야 합니다.
    # '0'→32, '1'→64, '2'→128, 'pool'→256
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', 'pool'],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=IMAGE_MIN_SIZE,
        max_size=IMAGE_MAX_SIZE,
    )
    return model


def build_model(backbone_name: str, num_classes: int):
    if backbone_name == 'resnet50':
        return build_resnet50_fasterrcnn(num_classes)
    if backbone_name in ('convnext_tiny', 'convnext_small'):
        variant = backbone_name.split('_')[1]   # 'tiny' or 'small'
        return build_convnext_fasterrcnn(num_classes, variant=variant)
    raise ValueError(f'Unsupported backbone: {backbone_name}')


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def model_state_dict_cpu(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in unwrap_model(model).state_dict().items()
    }


def validate_batch_size(batch_size: int):
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    if USE_DATA_PARALLEL and batch_size < len(DATA_PARALLEL_DEVICE_IDS):
        raise ValueError(
            f'batch_size={batch_size} must be >= active_gpu_count={len(DATA_PARALLEL_DEVICE_IDS)} '
            'when DataParallel is enabled.'
        )


def describe_batch_split(batch_size: int):
    validate_batch_size(batch_size)
    if not USE_DATA_PARALLEL:
        return [batch_size]

    num_chunks = min(len(DATA_PARALLEL_DEVICE_IDS), batch_size)
    chunk_size = math.ceil(batch_size / num_chunks)
    return [
        min(chunk_size, batch_size - chunk_idx * chunk_size)
        for chunk_idx in range(num_chunks)
        if batch_size - chunk_idx * chunk_size > 0
    ]


def build_training_model(backbone_name: str, num_classes: int, device: torch.device):
    model = build_model(backbone_name, num_classes=num_classes).to(device)
    if USE_DATA_PARALLEL:
        print(f'using DataParallel on GPUs: {DATA_PARALLEL_DEVICE_IDS}')
        model = DetectionDataParallel(model, device_ids=DATA_PARALLEL_DEVICE_IDS)
    return model


def make_loaders(train_dataset, val_dataset, batch_size):
    validate_batch_size(batch_size)
    print(f'batch_size={batch_size}, per_step_split={describe_batch_split(batch_size)}')

    use_persistent = NUM_WORKERS > 0
    prefetch = 2 if NUM_WORKERS > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
        multiprocessing_context=MULTIPROCESSING_CONTEXT,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
        multiprocessing_context=MULTIPROCESSING_CONTEXT,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def move_batch_to_device(images, targets, device):
    images = [image.to(device, non_blocking=NON_BLOCKING) for image in images]
    targets = [{k: v.to(device, non_blocking=NON_BLOCKING) for k, v in target.items()} for target in targets]
    return images, targets


def reduce_loss_dict(loss_dict, model=None):
    reduced = {}
    chunk_sizes = getattr(model, 'last_chunk_sizes', []) if isinstance(model, DetectionDataParallel) else []

    for loss_name, loss_value in loss_dict.items():
        if isinstance(loss_value, torch.Tensor) and loss_value.ndim > 0:
            if chunk_sizes and loss_value.numel() == len(chunk_sizes):
                weights = loss_value.new_tensor(chunk_sizes, dtype=loss_value.dtype)
                reduced[loss_name] = (loss_value * weights / weights.sum()).sum()
            else:
                reduced[loss_name] = loss_value.mean()
        else:
            reduced[loss_name] = loss_value

    return reduced


def forward_loss_dict(model, images, targets, device):
    if isinstance(model, DetectionDataParallel):
        loss_dict = model(list(images), list(targets))
    else:
        images, targets = move_batch_to_device(images, targets, device)
        loss_dict = model(images, targets)
    return reduce_loss_dict(loss_dict, model=model)


def train_one_epoch(model, data_loader, optimizer, device, epoch, writer=None, global_step_start=0):
    model.train()
    running_loss = 0.0
    last_global_step = global_step_start

    progress = None
    task_id = None

    if USE_RICH_PROGRESS:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Train[/bold cyan] epoch {task.fields[epoch]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss={task.fields[loss]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=RICH_REFRESH_PER_SECOND,
            transient=True,
        )
        progress.start()
        task_id = progress.add_task(
            "train",
            total=len(data_loader),
            epoch=f"{epoch:02d}",
            loss="-.----",
        )

    try:
        for step, (images, targets) in enumerate(data_loader, start=1):
            loss_dict = forward_loss_dict(model, images, targets, device)
            loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch}, step={step}: {loss_dict}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            running_loss += loss_value
            global_step = global_step_start + step
            last_global_step = global_step

            if writer is not None:
                writer.add_scalar("train/step_loss", loss_value, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

                for loss_name, loss_tensor in loss_dict.items():
                    if isinstance(loss_tensor, torch.Tensor):
                        writer.add_scalar(
                            f"train_step_components/{loss_name}",
                            float(loss_tensor.detach().item()),
                            global_step,
                        )

            if progress is not None:
                progress.update(task_id, advance=1, loss=f"{loss_value:.4f}")
            elif step % 10 == 0 or step == len(data_loader):
                print(f"[epoch {epoch:02d}] step {step:04d}/{len(data_loader):04d} loss={loss_value:.4f}")

        avg_loss = running_loss / max(len(data_loader), 1)
        return avg_loss, last_global_step

    finally:
        if progress is not None:
            progress.stop()


def evaluate_val_loss(model, data_loader, device):
    model.train()  # torchvision detection 모델은 loss 계산 시 train mode 필요
    total_loss = 0.0
    component_sums = {}

    progress = None
    task_id = None

    if USE_RICH_PROGRESS:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]Valid[/bold magenta]"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss={task.fields[loss]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=RICH_REFRESH_PER_SECOND,
            transient=True,
        )
        progress.start()
        task_id = progress.add_task("valid", total=len(data_loader), loss="-.----")

    try:
        with torch.no_grad():
            for step, (images, targets) in enumerate(data_loader, start=1):
                loss_dict = forward_loss_dict(model, images, targets, device)
                loss = sum(loss_dict.values())
                loss_value = float(loss.item())
                total_loss += loss_value

                for loss_name, loss_tensor in loss_dict.items():
                    component_sums.setdefault(loss_name, 0.0)
                    component_sums[loss_name] += float(loss_tensor.detach().item())

                if progress is not None:
                    progress.update(task_id, advance=1, loss=f"{loss_value:.4f}")

        avg_total = total_loss / max(len(data_loader), 1)
        avg_components = {
            loss_name: loss_sum / max(len(data_loader), 1)
            for loss_name, loss_sum in component_sums.items()
        }
        return avg_total, avg_components

    finally:
        if progress is not None:
            progress.stop()


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    trial=None,
    writer=None,
    hparams=None,
):
    history = []
    best_state = None
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, epochs + 1):
        train_loss, global_step = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer,
            global_step_start=global_step,
        )

        val_loss, val_components = evaluate_val_loss(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                **{f"val_{k}": v for k, v in val_components.items()},
            }
        )

        print(
            f"[epoch {epoch:02d}] "
            f"train={train_loss:.4f} "
            f"val={val_loss:.4f} "
            f"lr={current_lr:.6e}"
        )

        if writer is not None:
            writer.add_scalar("epoch/train_loss", train_loss, epoch)
            writer.add_scalar("epoch/val_loss", val_loss, epoch)
            writer.add_scalar("epoch/lr", current_lr, epoch)

            for loss_name, loss_value in val_components.items():
                writer.add_scalar(f"val_components/{loss_name}", loss_value, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model_state_dict_cpu(model)

            if writer is not None:
                writer.add_scalar("best/val_loss", best_val_loss, epoch)

        if trial is not None:
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    if writer is not None and hparams is not None:
        metric_dict = {"hparam/best_val_loss": best_val_loss}
        writer.add_hparams(hparams, metric_dict)

    return history, best_state, best_val_loss


def smoke_test_training_step(backbone_name, dataset, num_classes, batch_size, device):
    sample_count = min(len(dataset), batch_size)
    if sample_count == 0:
        raise ValueError('dataset is empty, so smoke test cannot run.')

    model = build_training_model(backbone_name, num_classes=num_classes, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    batch = [dataset[idx] for idx in range(sample_count)]
    images, targets = collate_fn(batch)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_dict = forward_loss_dict(model, images, targets, device)
    loss = sum(loss_dict.values())

    if not torch.isfinite(loss):
        raise RuntimeError(f'smoke test produced non-finite loss: {loss_dict}')

    loss.backward()
    optimizer.step()

    result = {
        'loss': float(loss.item()),
        'loss_components': {key: float(value.item()) for key, value in loss_dict.items()},
        'batch_size': sample_count,
        'gpu_count': GPU_COUNT,
        'used_data_parallel': USE_DATA_PARALLEL,
    }

    del model
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

# =========================
# 5) Optuna 탐색 실행
# =========================
set_seed(SEED)
train_dataset, val_dataset, disease_to_label, label_to_name = prepare_datasets()
num_classes = len(label_to_name) + 1

if min(BATCH_SIZE_OPTIONS) < max(GPU_COUNT, 1):
    raise ValueError(
        f'BATCH_SIZE_OPTIONS={BATCH_SIZE_OPTIONS} must be >= active_gpu_count={max(GPU_COUNT, 1)} '
        'for the configured training setup.'
    )

smoke_test_batch_size = max(min(BATCH_SIZE_OPTIONS), max(GPU_COUNT, 1))
smoke_test_result = smoke_test_training_step(
    backbone_name=SEARCH_BACKBONES[0],
    dataset=train_dataset,
    num_classes=num_classes,
    batch_size=smoke_test_batch_size,
    device=DEVICE,
)
print('smoke test passed:', smoke_test_result)


def objective(trial):
    backbone = trial.suggest_categorical("backbone", SEARCH_BACKBONES)
    batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_OPTIONS)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model = build_training_model(backbone, num_classes=num_classes, device=DEVICE)
    train_loader, val_loader = make_loaders(train_dataset, val_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SEARCH_EPOCHS)

    writer = None
    run_dir = None

    hparams = format_hparams(
        backbone=backbone,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=SEARCH_EPOCHS,
    )

    try:
        if USE_TENSORBOARD:
            run_dir = make_run_dir(prefix="optuna", trial_number=trial.number)
            writer = SummaryWriter(log_dir=str(run_dir))
            writer.add_text("meta/device", str(DEVICE))
            writer.add_text("meta/run_dir", str(run_dir))
            writer.add_text("meta/per_step_split", str(describe_batch_split(batch_size)))

        _, _, best_val_loss = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            epochs=SEARCH_EPOCHS,
            trial=trial,
            writer=writer,
            hparams=hparams,
        )

        if writer is not None:
            writer.flush()

        return best_val_loss

    finally:
        if writer is not None:
            writer.close()

        del model
        del optimizer
        del scheduler
        del train_loader
        del val_loader

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _create_study_safe(study_name, storage_url, pruner):
    """Optuna study를 생성합니다.

    기존 DB에 저장된 하이퍼파라미터 분포(예: BATCH_SIZE_OPTIONS, SEARCH_BACKBONES)가
    현재 설정과 달라지면 `CategoricalDistribution does not support dynamic value space`
    ValueError가 발생합니다. 이 경우 기존 study를 삭제하고 새로 생성합니다.

    발생 원인:
        load_if_exists=True 상태에서 BATCH_SIZE_OPTIONS 또는 SEARCH_BACKBONES 값을
        바꾸면 DB에 저장된 분포와 충돌 → Optuna가 거부.
    해결:
        DB에서 해당 study를 삭제하고 처음부터 재탐색.
        (완료된 trial이 있어도 파라미터 공간이 달라졌으므로 재활용 불가)
    """
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction='minimize',
            load_if_exists=True,
            pruner=pruner,
        )
        # 기존 trial이 있고 파라미터 공간이 바뀌었으면 여기서 검증
        if study.trials:
            first_trial = study.trials[0]
            stored_backbones = first_trial.distributions.get('backbone')
            stored_batch = first_trial.distributions.get('batch_size')
            current_backbones = set(SEARCH_BACKBONES)
            current_batch = set(BATCH_SIZE_OPTIONS)

            mismatch = False
            if stored_backbones and set(stored_backbones.choices) != current_backbones:
                print(f'[study] backbone 공간 변경 감지: {stored_backbones.choices} → {sorted(current_backbones)}')
                mismatch = True
            if stored_batch and set(stored_batch.choices) != current_batch:
                print(f'[study] batch_size 공간 변경 감지: {stored_batch.choices} → {sorted(current_batch)}')
                mismatch = True

            if mismatch:
                print('[study] 파라미터 공간 불일치 → 기존 study 삭제 후 재생성합니다.')
                optuna.delete_study(study_name=study_name, storage=storage_url)
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_url,
                    direction='minimize',
                    load_if_exists=False,
                    pruner=pruner,
                )

        return study

    except Exception as exc:
        # DB 파일 자체가 손상됐거나 다른 이유로 생성 실패 시 DB 삭제 후 재시도
        print(f'[study] study 로드 실패 ({exc}) → DB 초기화 후 재생성합니다.')
        if STUDY_DB_PATH.exists():
            STUDY_DB_PATH.unlink()
        return optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction='minimize',
            load_if_exists=False,
            pruner=pruner,
        )


storage_url = f"sqlite:///{STUDY_DB_PATH.resolve().as_posix()}"
study = _create_study_safe(
    study_name=STUDY_NAME,
    storage_url=storage_url,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
)

study.optimize(objective, n_trials=SEARCH_TRIALS, show_progress_bar=True)

best_tb_dir = make_run_dir(prefix="study_summary")
best_writer = SummaryWriter(log_dir=str(best_tb_dir))

best_writer.add_text("study/best_params", str(study.best_trial.params))
best_writer.add_scalar("study/best_value", float(study.best_value), 0)

for key, value in study.best_trial.params.items():
    if isinstance(value, (int, float)):
        best_writer.add_scalar(f"study_best_params/{key}", value, 0)
    else:
        best_writer.add_text(f"study_best_params/{key}", str(value))

best_writer.flush()
best_writer.close()

print('best value:', study.best_value)
print('best params:', study.best_trial.params)
study.best_trial.params

# =========================
# 6) 베스트 파라미터로 최종 학습 + 저장 + 샘플 추론
# =========================
import matplotlib.pyplot as plt

best_params = study.best_trial.params
print('using params:', best_params)

final_model = build_training_model(best_params['backbone'], num_classes=num_classes, device=DEVICE)
final_train_loader, final_val_loader = make_loaders(
    train_dataset,
    val_dataset,
    batch_size=best_params['batch_size'],
)

final_optimizer = torch.optim.AdamW(
    final_model.parameters(),
    lr=best_params['learning_rate'],
    weight_decay=best_params['weight_decay'],
)
final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=FINAL_EPOCHS)

history, best_state, best_val_loss = run_training(
    model=final_model,
    train_loader=final_train_loader,
    val_loader=final_val_loader,
    optimizer=final_optimizer,
    scheduler=final_scheduler,
    device=DEVICE,
    epochs=FINAL_EPOCHS,
)

if best_state is not None:
    unwrap_model(final_model).load_state_dict(best_state)

checkpoint = {
    'model_state_dict': model_state_dict_cpu(final_model),
    'best_params': best_params,
    'best_val_loss': best_val_loss,
    'class_names': label_to_name,
    'disease_to_label': disease_to_label,
    'device': str(DEVICE),
    'available_gpu_count': AVAILABLE_GPU_COUNT,
    'gpu_count': GPU_COUNT,
    'max_train_gpus': MAX_TRAIN_GPUS,
    'data_parallel_device_ids': DATA_PARALLEL_DEVICE_IDS,
    'used_data_parallel': USE_DATA_PARALLEL,
}

best_model_path = OUTPUT_DIR / 'best_model.pt'
last_model_path = OUTPUT_DIR / 'last_model.pt'
torch.save(checkpoint, best_model_path)
torch.save({**checkpoint, 'history': history}, last_model_path)

print('best validation loss:', best_val_loss)
print('saved best model to:', best_model_path.resolve())
print('saved last model to:', last_model_path.resolve())


@torch.inference_mode()
def predict_one(model, image_path: Path, score_threshold=0.3):
    base_model = unwrap_model(model)
    base_model.eval()
    image = Image.open(image_path).convert('RGB')
    tensor = torchvision.transforms.functional.to_tensor(image).to(DEVICE)
    output = base_model([tensor])[0]

    scores = output['scores'].detach().cpu()
    boxes = output['boxes'].detach().cpu()
    labels = output['labels'].detach().cpu()
    keep = scores >= score_threshold

    return image, boxes[keep], labels[keep], scores[keep]


def show_prediction(image, boxes, labels, scores, label_to_name):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        x1, y1, x2, y2 = box
        class_name = label_to_name.get(label, f'class_{label}')
        draw.rectangle([x1, y1, x2, y2], outline='red', width=4)
        draw.text((x1, max(0, y1 - 20)), f'{class_name}: {score:.2f}', fill='red')

    plt.figure(figsize=(8, 8))
    plt.imshow(canvas)
    plt.axis('off')
    plt.show()


sample_record = val_dataset.records[0]
image, boxes, labels, scores = predict_one(final_model, sample_record.image_path, score_threshold=0.25)
print('sample image:', sample_record.image_path)
print('detections:', len(boxes))
show_prediction(image, boxes, labels, scores, label_to_name)
