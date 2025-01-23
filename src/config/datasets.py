from dataclasses import dataclass, field
from typing import List


@dataclass
class Dataset:
    name: str
    modality: str
    val_file: str
    train_file: str
    classes: List[str] = field(default_factory=list)


datasets = {
    "kits23": Dataset(
        name="00_kits23",
        modality="CT",
        classes=[
            "angiomyolipoma",
            "chromophobe_rcc",
            "clear_cell_rcc",
            "oncocytoma",
            "papillary_rcc",
        ],
        val_file="splits/00_kits23/val.jsonl",
        train_file="splits/00_kits23/train.jsonl",
    ),
    "coronahack": Dataset(
        name="01_coronahack",
        modality="X-ray",
        classes=["Normal", "PneumoniaBacteria", "PneumoniaVirus"],
        val_file="splits/01_coronahack/val.jsonl",
        train_file="splits/01_coronahack/train.jsonl",
    ),
    "brain_tumor_classification": Dataset(
        name="03_brain_tumor_classification",
        modality="MRI",
        classes=["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"],
        val_file="splits/03_brain_tumor_classification/val.jsonl",
        train_file="splits/03_brain_tumor_classification/val.jsonl",
    ),
}
