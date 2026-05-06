#!/usr/bin/env python3
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional


class MaskLoader:
    def __init__(self, instances_json_path: str) -> None:
        self.instances_json_path = instances_json_path
        with open(instances_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        annotations = payload.get("annotations", [])
        self.ann_by_id: Dict[Any, Dict[str, Any]] = {}
        for ann in annotations:
            if isinstance(ann, dict) and "id" in ann:
                self.ann_by_id[ann["id"]] = ann
        self._cache: Dict[Any, Dict[str, Any]] = {}

    def get_annotation(self, annotation_id: Any) -> Optional[Dict[str, Any]]:
        if annotation_id in self._cache:
            return self._cache[annotation_id]
        ann = self.ann_by_id.get(annotation_id)
        if ann is not None:
            self._cache[annotation_id] = ann
        return ann

    def get_bbox(self, annotation_id: Any) -> Any:
        ann = self.get_annotation(annotation_id)
        if ann is None:
            return None
        return ann.get("bbox")

    def get_segmentation(self, annotation_id: Any) -> Any:
        ann = self.get_annotation(annotation_id)
        if ann is None:
            return None
        return ann.get("segmentation")

    def get_mask_ref(self, annotation_id: Any) -> Any:
        ann = self.get_annotation(annotation_id)
        if ann is None:
            return None
        return {"annotation_id": annotation_id, "annotation": ann}


class EnhancedRRSISDAdapter:
    def __init__(
        self,
        enhanced_json_path: str,
        expression_mode: str = "enhanced",
        view_mode: str = "flatten",
        failed_policy: str = "use_raw",
        instances_json_path: Optional[str] = None,
    ) -> None:
        if expression_mode not in {"raw", "enhanced", "compressed"}:
            raise ValueError("expression_mode must be raw|enhanced|compressed")
        if view_mode not in {"flatten", "grouped", "virtual_refs"}:
            raise ValueError("view_mode must be flatten|grouped|virtual_refs")
        if failed_policy not in {"use_raw", "skip", "raise_error"}:
            raise ValueError("failed_policy must be use_raw|skip|raise_error")
        self.expression_mode = expression_mode
        self.view_mode = view_mode
        self.failed_policy = failed_policy
        self.mask_loader = MaskLoader(instances_json_path) if instances_json_path else None
        with open(enhanced_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.records = payload.get("records", [])
        if not self.records:
            refs = payload.get("refs", [])
            for ref in refs:
                for expr in ref.get("expressions", []):
                    self.records.append(expr)
        self._flatten = self._build_flatten_items()
        self._grouped = self._build_grouped_items()

    def _select_text(self, rec: Dict[str, Any]) -> str:
        status = rec.get("status")
        if status not in {"success", "unchanged"}:
            if self.failed_policy == "skip":
                return ""
            if self.failed_policy == "raise_error":
                raise ValueError(f"unusable status={status}, expr_id={rec.get('expr_id')}")
            return rec.get("raw", "")
        return rec.get(self.expression_mode, rec.get("raw", ""))

    def _build_flatten_items(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec in self.records:
            text = self._select_text(rec)
            if text == "":
                continue
            ann_id = rec.get("ann_id")
            mask = rec.get("segmentation_ref")
            if self.mask_loader and ann_id is not None:
                mask = self.mask_loader.get_mask_ref(ann_id)
            out.append(
                {
                    "image": rec.get("file_name"),
                    "mask": mask,
                    "text_expression": text,
                    "sample_id": f"image_{rec.get('image_id')}",
                    "ref_id": rec.get("ref_id"),
                    "ann_id": ann_id,
                    "image_id": rec.get("image_id"),
                    "expr_id": rec.get("expr_id"),
                    "split": rec.get("split"),
                    "category_id": rec.get("category_id"),
                    "category_name": rec.get("category_name"),
                    "bbox": rec.get("bbox"),
                    "segmentation_ref": rec.get("segmentation_ref"),
                    "slots": rec.get("slots", {}),
                    "meta": rec,
                }
            )
        return out

    def _build_grouped_items(self) -> List[Dict[str, Any]]:
        by_ref: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for item in self._flatten:
            by_ref[item.get("ref_id")].append(item)
        grouped = []
        for ref_id, rows in by_ref.items():
            first = rows[0]
            grouped.append(
                {
                    "ref_id": ref_id,
                    "ann_id": first.get("ann_id"),
                    "image_id": first.get("image_id"),
                    "file_name": first.get("image"),
                    "split": first.get("split"),
                    "category_id": first.get("category_id"),
                    "category_name": first.get("category_name"),
                    "bbox": first.get("bbox"),
                    "segmentation_ref": first.get("segmentation_ref"),
                    "expressions": rows,
                }
            )
        return grouped

    def get_legacy_refs(self) -> List[Dict[str, Any]]:
        refs: Dict[Any, Dict[str, Any]] = {}
        for rec in self.records:
            text = self._select_text(rec)
            if text == "":
                continue
            ref_id = rec.get("ref_id")
            if ref_id not in refs:
                refs[ref_id] = {
                    "ref_id": ref_id,
                    "ann_id": rec.get("ann_id"),
                    "image_id": rec.get("image_id"),
                    "file_name": rec.get("file_name"),
                    "category_id": rec.get("category_id"),
                    "split": rec.get("split"),
                    "sent_ids": [],
                    "sentences": [],
                }
            sent_id = rec.get("sent_id")
            refs[ref_id]["sent_ids"].append(sent_id)
            refs[ref_id]["sentences"].append(
                {
                    "sent": text,
                    "raw": rec.get("raw", ""),
                    "sent_id": sent_id,
                    "tokens": [],
                    "expr_id": rec.get("expr_id"),
                }
            )
        return list(refs.values())

    def __len__(self) -> int:
        if self.view_mode == "flatten":
            return len(self._flatten)
        if self.view_mode == "grouped":
            return len(self._grouped)
        return len(self.get_legacy_refs())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.view_mode == "flatten":
            return self._flatten[idx]
        if self.view_mode == "grouped":
            return self._grouped[idx]
        return self.get_legacy_refs()[idx]
