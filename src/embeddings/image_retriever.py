from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


class VisualRetrieverUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class VisualRetrieverInfo:
    model_name: str
    device: str
    local_files_only: bool


class ColQwen2ImageRetriever:
    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0-hf",
        device: str | None = None,
        local_files_only: bool = True,
    ):
        try:
            import torch
            from transformers import ColQwen2ForRetrieval, ColQwen2Processor
        except Exception as exc:
            raise VisualRetrieverUnavailable(f"ColQwen2 dependencies are unavailable: {exc}") from exc

        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_files_only = local_files_only
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        try:
            self.processor = ColQwen2Processor.from_pretrained(model_name, local_files_only=local_files_only)
            self.model = ColQwen2ForRetrieval.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                torch_dtype=dtype,
            )
            self.model.eval()
            self.model.to(self.device)
        except Exception as exc:
            raise VisualRetrieverUnavailable(
                f"Could not load visual retriever '{model_name}'. "
                "Cache the model locally or allow network access first. "
                f"Original error: {exc}"
            ) from exc

    @property
    def info(self) -> VisualRetrieverInfo:
        return VisualRetrieverInfo(
            model_name=self.model_name,
            device=self.device,
            local_files_only=self.local_files_only,
        )

    def encode_query(self, query: str) -> np.ndarray:
        batch = self._prepare_query_batch([query])
        embeddings = self._run_model(batch)
        return embeddings[0]

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        return self.encode_images([image_path])[0]

    def encode_images(self, image_paths: Iterable[str | Path], batch_size: int = 1) -> list[np.ndarray]:
        paths = list(image_paths)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        embeddings: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            batch = self._prepare_image_batch(images)
            embeddings.extend(self._run_model(batch))
            self.clear_device_cache()
        return embeddings

    def _prepare_query_batch(self, queries: list[str]):
        if hasattr(self.processor, "process_queries"):
            batch = self.processor.process_queries(queries)
        else:
            batch = self.processor(text=queries, return_tensors="pt", padding=True, truncation=True)
        return self._move_batch(batch)

    def _prepare_image_batch(self, images: list[Image.Image]):
        if hasattr(self.processor, "process_images"):
            batch = self.processor.process_images(images)
        else:
            batch = self.processor(images=images, return_tensors="pt")
        return self._move_batch(batch)

    def _move_batch(self, batch):
        if hasattr(batch, "to"):
            return batch.to(self.device)
        return {key: value.to(self.device) for key, value in batch.items()}

    def _run_model(self, batch) -> list[np.ndarray]:
        with self._torch.inference_mode():
            outputs = self.model(**batch)
        tensor = self._extract_embedding_tensor(outputs)
        tensor = tensor.detach().to("cpu").float().numpy()
        del outputs
        del batch
        return [item.astype("float32") for item in tensor]

    def _extract_embedding_tensor(self, outputs):
        if isinstance(outputs, self._torch.Tensor):
            return outputs

        for name in ("embeddings", "last_hidden_state", "query_embeddings", "image_embeddings"):
            value = getattr(outputs, name, None)
            if value is not None:
                return value

        if hasattr(outputs, "to_tuple"):
            for item in outputs.to_tuple():
                if isinstance(item, self._torch.Tensor):
                    return item
        raise VisualRetrieverUnavailable("Could not extract embeddings from ColQwen2 outputs.")

    def clear_device_cache(self) -> None:
        if self.device.startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
