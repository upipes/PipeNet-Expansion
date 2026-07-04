import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from LLM_desc_gen.prompt_suite import PhysicsAwarePromptSuite
from LLM_desc_gen.semantic_prior_generator import SemanticPriorGenerator
from utility.utils import mkdirp, read_lines, save_json


EMBED_DIMS = {
    "clip": 512,
    "sbert": 768,
    "llama": 4096,
    "qwen": 3584,
}


def _safe_read(path):
    lines = read_lines(path)
    return [] if lines is None else [line for line in lines if line and line.strip()]


def _slug(text):
    return text.replace("/", "_").replace("\\", "_")


class StructuredSemanticBuilder:
    def __init__(self, rootpath, llm_name="gpt4o", backend="sbert"):
        self.rootpath = Path(rootpath)
        self.llm_name = llm_name
        self.backend = backend
        self.dim = EMBED_DIMS[backend]
        self.prompt_suite = PhysicsAwarePromptSuite(rootpath)
        self.prior_generator = SemanticPriorGenerator(rootpath, llm_name=llm_name)

    def _domain_text(self, dataset):
        domain_lines = _safe_read(self.rootpath / "prompt" / "aux_info" / f"domain_{dataset}.txt")
        return " ".join(domain_lines)

    def _classnames(self, dataset):
        data_names = _safe_read(self.rootpath / "data" / dataset / "idx2name.txt")
        if data_names:
            return data_names
        aux_names = _safe_read(self.rootpath / "prompt" / "aux_info" / f"classnames_{dataset}.txt")
        return aux_names

    def _class_descriptions(self, dataset):
        desc_lines = _safe_read(self.rootpath / "prompt" / "aux_info" / f"classes_description_{dataset}.txt")
        if not desc_lines:
            desc_lines, _ = self.prior_generator.ensure_priors(dataset)
        descriptions = {}
        for line in desc_lines:
            if ":" not in line:
                continue
            name, desc = line.split(":", 1)
            descriptions[name.strip().lower()] = desc.strip()
        return descriptions

    def _views(self, dataset):
        view_lines = _safe_read(self.rootpath / "prompt" / "aux_info" / f"views_{dataset}.txt")
        if not view_lines:
            _, view_lines = self.prior_generator.ensure_priors(dataset)
        views = []
        for line in view_lines:
            if ". " in line:
                views.append(line.split(". ", 1)[1].strip())
            else:
                views.append(line.strip())
        return views

    def _parse_response_text(self, text):
        classnames = []
        section_map = self._parse_class_section_text(text, classnames)
        if section_map:
            return section_map
        results = {}
        for raw_line in text.strip().splitlines():
            line = raw_line.strip().replace("**", "")
            if not line:
                continue
            match = re.match(r"^\d+\.\s*([^:]+):\s*(.+)$", line)
            if match:
                results[match.group(1).strip().lower()] = match.group(2).strip()
        return results

    def _looks_like_class_map(self, payload, classnames):
        if not isinstance(payload, dict) or not payload:
            return False
        normalized = {name.lower() for name in classnames}
        overlap = sum(1 for key in payload if key.lower() in normalized)
        return overlap > 0

    def _parse_class_section_text(self, text, classnames):
        normalized_names = {name.lower(): name for name in classnames}
        section_map = {}
        current_name = None
        current_lines = []
        header_pattern = re.compile(r"^\s*(?:\d+[\.\)]\s*)?(?:[-*]\s*)?(?:\*\*|__)?([^:\n*]+?)(?:\*\*|__)?\s*:\s*(.*)$")

        def flush_section():
            nonlocal current_name, current_lines
            if current_name is None:
                return
            merged = " ".join(line.strip() for line in current_lines if line.strip())
            merged = re.sub(r"\s+", " ", merged).strip()
            section_map[current_name] = merged
            current_name = None
            current_lines = []

        for raw_line in text.splitlines():
            stripped = raw_line.strip().replace("**", "").replace("__", "")
            if not stripped:
                continue
            match = header_pattern.match(stripped)
            candidate = match.group(1).strip().lower() if match else None
            if candidate and (not normalized_names or candidate in normalized_names):
                flush_section()
                current_name = candidate
                tail = match.group(2).strip() if match else ""
                current_lines = [tail] if tail else []
                continue
            if current_name is not None:
                current_lines.append(stripped)

        flush_section()
        return section_map

    def _parse_role_query_file(self, dataset, role, view_name):
        classnames = self._classnames(dataset)
        structured_path = (
            self.rootpath
            / "LLM_query"
            / "physics_aware"
            / "roles"
            / dataset
            / role
            / self.llm_name
            / "structured.json"
        )
        if structured_path.exists():
            payload = json.loads(structured_path.read_text(encoding="utf-8"))
            parsed = payload.get("parsed", {}).get(view_name, {})
            if self._looks_like_class_map(parsed, classnames):
                return parsed

        path = self.rootpath / "LLM_query" / "multi_view" / dataset / f"{self.llm_name}_view" / f"{view_name}.txt"
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return {}
        return self._parse_class_section_text(text, classnames) or self._parse_response_text(text)

    def _parse_transfer_query_file(self, source_dataset, target_dataset, view_name):
        classnames = self._classnames(target_dataset)
        structured_path = (
            self.rootpath
            / "LLM_query"
            / "physics_aware"
            / "transfers"
            / f"{source_dataset}_to_{target_dataset}"
            / self.llm_name
            / "structured.json"
        )
        if structured_path.exists():
            payload = json.loads(structured_path.read_text(encoding="utf-8"))
            parsed = payload.get("parsed", {}).get(view_name, {})
            if self._looks_like_class_map(parsed, classnames):
                return parsed

        raw_path = (
            self.rootpath
            / "LLM_query"
            / "physics_aware"
            / "transfers"
            / f"{source_dataset}_to_{target_dataset}"
            / self.llm_name
            / f"{view_name}.txt"
        )
        if raw_path.exists():
            text = raw_path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return self._parse_class_section_text(text, classnames) or self._parse_response_text(text)
        return {}

    def _compose_global_text(self, dataset, class_name, class_desc, role):
        domain = self._domain_text(dataset)
        return (
            f"Dataset domain: {domain}. "
            f"Role: {role}. "
            f"Target class: {class_name}. "
            f"Global physical interpretation: {class_desc}. "
            f"This text summarizes stable GPR evidence, likely waveform response, "
            f"structural context, and discriminative cues for cross-domain recognition."
        )

    def _compose_view_text(self, dataset, class_name, class_desc, view_name, query_desc, role):
        domain = self._domain_text(dataset)
        if query_desc:
            return (
                f"Dataset domain: {domain}. "
                f"Role: {role}. "
                f"Class: {class_name}. "
                f"View: {view_name}. "
                f"Prompt-derived description: {query_desc}. "
                f"Base class prior: {class_desc}."
            )
        return (
            f"Dataset domain: {domain}. "
            f"Role: {role}. "
            f"Class: {class_name}. "
            f"View: {view_name}. "
            f"Interpret the class from this physical perspective using GPR cues. "
            f"Base class prior: {class_desc}."
        )

    def _compose_intervention_text(
        self,
        source_dataset,
        target_dataset,
        class_name,
        source_global,
        target_global,
        view_name,
        source_view,
        target_view,
        transfer_text="",
    ):
        source_domain = self._domain_text(source_dataset)
        target_domain = self._domain_text(target_dataset)
        text = (
            f"Causal transfer for class {class_name}. "
            f"Source domain: {source_domain}. "
            f"Target domain: {target_domain}. "
            f"Stable source semantics: {source_global}. "
            f"Stable target semantics: {target_global}. "
            f"View focus: {view_name}. "
            f"Source-view expression: {source_view}. "
            f"Target-view expression: {target_view}. "
        )
        if transfer_text:
            text += f" Prompt-derived cross-domain intervention: {transfer_text}. "
        text += "Describe mechanism-preserving evidence and domain-induced changes separately."
        return text

    def _compose_target_counterfactual_text(
        self,
        target_dataset,
        class_name,
        target_global,
        counterfactual_text,
    ):
        target_domain = self._domain_text(target_dataset)
        return (
            f"Target-internal counterfactual for class {class_name}. "
            f"Target dataset: {target_dataset}. "
            f"Target domain: {target_domain}. "
            f"Stable target semantics: {target_global}. "
            f"Counterfactual description: {counterfactual_text}. "
            f"Describe which cues preserve the class and which substitutions change the decision boundary."
        )

    def _hash_embed(self, text):
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"[a-zA-Z0-9_\-\+\[\]]+", text.lower())
        if not tokens:
            return vector
        for ngram_size in (1, 2, 3):
            for idx in range(len(tokens) - ngram_size + 1):
                ngram = " ".join(tokens[idx : idx + ngram_size])
                digest = hashlib.sha256(ngram.encode("utf-8")).digest()
                for offset in range(0, len(digest), 4):
                    chunk = digest[offset : offset + 4]
                    pos = int.from_bytes(chunk, "little") % self.dim
                    sign = 1.0 if chunk[0] % 2 == 0 else -1.0
                    vector[pos] += sign / float(ngram_size)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def _embed_texts(self, texts):
        return np.stack([self._hash_embed(text) for text in texts], axis=0)

    def _parse_target_internal_query_file(self, target_dataset):
        classnames = self._classnames(target_dataset)
        structured_path = (
            self.rootpath
            / "LLM_query"
            / "physics_aware"
            / "target_internal"
            / target_dataset
            / self.llm_name
            / "structured.json"
        )
        if structured_path.exists():
            payload = json.loads(structured_path.read_text(encoding="utf-8"))
            parsed = payload.get("parsed", {})
            if self._looks_like_class_map(parsed, classnames):
                return parsed

        raw_path = (
            self.rootpath
            / "LLM_query"
            / "physics_aware"
            / "target_internal"
            / target_dataset
            / self.llm_name
            / "target_internal_counterfactual.txt"
        )
        if raw_path.exists():
            text = raw_path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return self._parse_class_section_text(text, classnames) or self._parse_response_text(text)

        path = (
            self.rootpath
            / "LLM_query"
            / "target_internal"
            / target_dataset
            / f"{self.llm_name}_view"
            / "target_internal_counterfactual.txt"
        )
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return {}
        return self._parse_class_section_text(text, classnames) or self._parse_response_text(text)

    def build_role_semantics(self, dataset, role):
        classnames = self._classnames(dataset)
        descriptions = self._class_descriptions(dataset)
        views = self._views(dataset)
        self.prompt_suite.build_dataset_prompts(
            dataset=dataset,
            domain=self._domain_text(dataset),
            classes=classnames,
            class_descriptions=descriptions,
            views=views,
            role=role,
        )
        query_bank = {view: self._parse_role_query_file(dataset, role, view) for view in ["global"] + views}

        texts = []
        records = []
        for class_name in classnames:
            class_key = class_name.lower()
            class_desc = descriptions.get(class_key, f"{class_name} in {dataset} GPR imagery.")
            global_text = self._compose_global_text(dataset, class_name, class_desc, role)
            class_texts = [global_text]
            class_records = {"class_name": class_name, "global": global_text, "views": {}}

            for view_name in views:
                query_desc = query_bank.get(view_name, {}).get(class_key, "")
                view_text = self._compose_view_text(dataset, class_name, class_desc, view_name, query_desc, role)
                class_texts.append(view_text)
                class_records["views"][view_name] = view_text

            texts.extend(class_texts)
            records.append(class_records)

        embedding = self._embed_texts(texts).reshape(len(classnames), len(views) + 1, self.dim)
        return {
            "dataset": dataset,
            "role": role,
            "backend": self.backend,
            "llm_name": self.llm_name,
            "classnames": classnames,
            "views": ["global"] + views,
            "embedding": embedding,
            "records": records,
        }

    def build_intervention_semantics(self, source_dataset, target_dataset):
        source_sem = self.build_role_semantics(source_dataset, role="source")
        target_sem = self.build_role_semantics(target_dataset, role="target")
        views = source_sem["views"][1:]
        self.prompt_suite.build_transfer_prompts(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            source_domain=self._domain_text(source_dataset),
            target_domain=self._domain_text(target_dataset),
            classes=target_sem["classnames"],
            class_descriptions=self._class_descriptions(target_dataset),
            views=views,
        )
        transfer_query_bank = {
            view: self._parse_transfer_query_file(source_dataset, target_dataset, view) for view in views
        }

        texts = []
        records = []
        for src_record, tgt_record in zip(source_sem["records"], target_sem["records"]):
            class_name = tgt_record["class_name"]
            class_texts = []
            class_record = {"class_name": class_name, "views": {}}
            for view_name in views:
                text = self._compose_intervention_text(
                    source_dataset,
                    target_dataset,
                    class_name,
                    src_record["global"],
                    tgt_record["global"],
                    view_name,
                    src_record["views"][view_name],
                    tgt_record["views"][view_name],
                    transfer_query_bank.get(view_name, {}).get(class_name.lower(), ""),
                )
                class_texts.append(text)
                class_record["views"][view_name] = text
            texts.extend(class_texts)
            records.append(class_record)

        embedding = self._embed_texts(texts).reshape(len(records), len(views), self.dim)
        return {
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "backend": self.backend,
            "llm_name": self.llm_name,
            "views": views,
            "embedding": embedding,
            "records": records,
        }

    def build_target_internal_semantics(self, target_dataset):
        target_sem = self.build_role_semantics(target_dataset, role="target")
        self.prompt_suite.build_target_internal_prompts(
            target_dataset=target_dataset,
            target_domain=self._domain_text(target_dataset),
            classes=target_sem["classnames"],
            class_descriptions=self._class_descriptions(target_dataset),
        )
        query_bank = self._parse_target_internal_query_file(target_dataset)

        texts = []
        records = []
        for tgt_record in target_sem["records"]:
            class_name = tgt_record["class_name"]
            class_key = class_name.lower()
            cf_text = query_bank.get(class_key, "")
            text = self._compose_target_counterfactual_text(
                target_dataset,
                class_name,
                tgt_record["global"],
                cf_text if cf_text else tgt_record["global"],
            )
            texts.append(text)
            class_record = {"class_name": class_name, "counterfactual": text}
            records.append(class_record)

        embedding = self._embed_texts(texts)
        return {
            "target_dataset": target_dataset,
            "backend": self.backend,
            "llm_name": self.llm_name,
            "embedding": embedding,
            "records": records,
        }

    def save_role_semantics(self, dataset, role, paired_dataset):
        payload = self.build_role_semantics(dataset, role=role)
        out_dir = self.rootpath / "embeddings" / self.backend
        mkdirp(str(out_dir))
        stem = f"{dataset}_as_{role}_for_{paired_dataset}_{self.llm_name}_{self.backend}"
        np.save(out_dir / f"{stem}.npy", payload["embedding"].astype(np.float32))
        meta = {
            "dataset": payload["dataset"],
            "role": payload["role"],
            "paired_dataset": paired_dataset,
            "backend": payload["backend"],
            "llm_name": payload["llm_name"],
            "classnames": payload["classnames"],
            "views": payload["views"],
            "records": payload["records"],
        }
        save_json(str(out_dir / f"{stem}.json"), meta)
        return payload

    def save_intervention_semantics(self, source_dataset, target_dataset):
        payload = self.build_intervention_semantics(source_dataset, target_dataset)
        out_dir = self.rootpath / "embeddings" / self.backend
        mkdirp(str(out_dir))
        file_stem = f"{source_dataset}_to_{target_dataset}_{self.llm_name}_{self.backend}_intervention"
        np.save(out_dir / f"{file_stem}.npy", payload["embedding"].astype(np.float32))
        meta = {
            "source_dataset": payload["source_dataset"],
            "target_dataset": payload["target_dataset"],
            "backend": payload["backend"],
            "llm_name": payload["llm_name"],
            "views": payload["views"],
            "records": payload["records"],
        }
        save_json(str(out_dir / f"{file_stem}.json"), meta)
        return payload

    def save_target_internal_semantics(self, target_dataset):
        payload = self.build_target_internal_semantics(target_dataset)
        out_dir = self.rootpath / "embeddings" / self.backend
        mkdirp(str(out_dir))
        file_stem = f"{target_dataset}_{self.llm_name}_{self.backend}_target_internal"
        np.save(out_dir / f"{file_stem}.npy", payload["embedding"].astype(np.float32))
        meta = {
            "target_dataset": payload["target_dataset"],
            "backend": payload["backend"],
            "llm_name": payload["llm_name"],
            "records": payload["records"],
        }
        save_json(str(out_dir / f"{file_stem}.json"), meta)
        return payload

def rebuild_semantics(rootpath, source_dataset, target_dataset, llm_name="gpt4o", backend="sbert"):
    builder = StructuredSemanticBuilder(rootpath=rootpath, llm_name=llm_name, backend=backend)
    builder.save_role_semantics(source_dataset, role="source", paired_dataset=target_dataset)
    builder.save_role_semantics(target_dataset, role="target", paired_dataset=source_dataset)
    builder.save_intervention_semantics(source_dataset, target_dataset)
    builder.save_target_internal_semantics(target_dataset)
