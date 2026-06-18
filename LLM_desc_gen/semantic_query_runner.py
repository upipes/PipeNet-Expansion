import json
import re
from pathlib import Path

from LLM_desc_gen.LLM import chatLLM
from LLM_desc_gen.prompt_suite import PhysicsAwarePromptSuite
from LLM_desc_gen.semantic_prior_generator import SemanticPriorGenerator
from utility.utils import load_json, mkdirp, read_lines, save_json


MODEL_ALIAS = {
    "gpt4o": "gpt-4o-2024-08-06",
    "gpt4omini": "gpt-4o-mini-2024-07-18",
    "gemini2.5": "gemini-2.5-flash",
    "llama70b": "llama3.1-70b-instruct",
    "qwen_plus": "qwen-plus",
}


def _safe_read(path):
    lines = read_lines(str(path))
    return [] if lines is None else [line for line in lines if line and line.strip()]


def _safe_view_name(view):
    if " - " in view and not view.startswith("["):
        return f"[{view.replace(' - ', ']-[')}]"
    return view


class SemanticQueryRunner:
    def __init__(self, rootpath, llm_name="gpt4o", temperature=0.7, delay_seconds=0):
        self.rootpath = Path(rootpath)
        self.llm_name = llm_name
        self.temperature = temperature
        self.delay_seconds = delay_seconds
        self.prompt_suite = PhysicsAwarePromptSuite(rootpath)
        self.prior_generator = SemanticPriorGenerator(rootpath, llm_name=llm_name)
        self.model_name = MODEL_ALIAS.get(llm_name, llm_name)

    def _domain_text(self, dataset):
        return " ".join(_safe_read(self.rootpath / "prompt" / "aux_info" / f"domain_{dataset}.txt"))

    def _classnames(self, dataset):
        data_names = _safe_read(self.rootpath / "data" / dataset / "idx2name.txt")
        if data_names:
            return data_names
        return _safe_read(self.rootpath / "prompt" / "aux_info" / f"classnames_{dataset}.txt")

    def _class_descriptions(self, dataset):
        desc_lines = _safe_read(self.rootpath / "prompt" / "aux_info" / f"classes_description_{dataset}.txt")
        descriptions = {}
        for line in desc_lines:
            if ":" not in line:
                continue
            name, desc = line.split(":", 1)
            descriptions[name.strip().lower()] = desc.strip()
        return descriptions

    def _views(self, dataset):
        view_lines = _safe_read(self.rootpath / "prompt" / "aux_info" / f"views_{dataset}.txt")
        views = []
        for line in view_lines:
            views.append(line.split(". ", 1)[1].strip() if ". " in line else line.strip())
        return views

    def _parse_classwise_response(self, text, classnames):
        text = text.strip()
        if not text:
            return {}

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return {str(key).strip().lower(): str(value).strip() for key, value in parsed.items()}
        except json.JSONDecodeError:
            pass

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
            line = raw_line.rstrip()
            stripped = line.strip().replace("**", "").replace("__", "")
            if not stripped:
                continue
            match = header_pattern.match(stripped)
            candidate = match.group(1).strip().lower() if match else None
            if candidate in normalized_names:
                flush_section()
                current_name = candidate
                tail = match.group(2).strip() if match else ""
                current_lines = [tail] if tail else []
                continue
            if current_name is not None:
                current_lines.append(stripped)

        flush_section()
        if section_map:
            return section_map

        mapping = {}
        lines = [line.strip().replace("**", "") for line in text.splitlines() if line.strip()]
        for line in lines:
            for marker in ("- ", "* "):
                if line.startswith(marker):
                    line = line[len(marker) :].strip()
                    break
            if ". " in line:
                prefix, suffix = line.split(". ", 1)
                if prefix.isdigit():
                    line = suffix.strip()
            if ":" in line:
                name, desc = line.split(":", 1)
                mapping[name.strip().lower()] = desc.strip()

        if mapping:
            return mapping

        fallback = {}
        if len(lines) == len(classnames):
            for class_name, desc in zip(classnames, lines):
                fallback[class_name.lower()] = desc
        return fallback

    def _query(self, system_prompt, user_prompt, save_path, overwrite=False):
        save_path = Path(save_path)
        if save_path.exists() and not overwrite:
            return save_path.read_text(encoding="utf-8", errors="ignore")

        mkdirp(str(save_path.parent))
        model = chatLLM(system_prompt, temperature=self.temperature)
        answer = model.chat(
            user_prompt,
            model=self.model_name,
            save_path=str(save_path),
            delay_seconds=self.delay_seconds,
        )
        if isinstance(answer, str):
            save_path.write_text(answer, encoding="utf-8")
            return answer
        return ""

    def _load_or_query_text(self, system_prompt, user_prompt, save_path, overwrite=False):
        save_path = Path(save_path)
        if save_path.exists() and not overwrite:
            return save_path.read_text(encoding="utf-8", errors="ignore")
        return self._query(system_prompt, user_prompt, save_path, overwrite=overwrite)

    def _mirror_legacy_role_outputs(self, dataset, role, payload):
        if role == "source":
            legacy_dir = self.rootpath / "LLM_query" / "multi_view" / dataset / f"{self.llm_name}_view"
            mkdirp(str(legacy_dir))
            for name, text in payload.items():
                legacy_name = "global.txt" if name == "global" else f"{name}.txt"
                (legacy_dir / legacy_name).write_text(text, encoding="utf-8")

    def _load_or_query_role(self, dataset, role, overwrite=False):
        classnames = self._classnames(dataset)
        views = self._views(dataset)
        self.prior_generator.ensure_priors(dataset, force=False)
        self.prompt_suite.build_dataset_prompts(
            dataset=dataset,
            domain=self._domain_text(dataset),
            classes=classnames,
            class_descriptions=self._class_descriptions(dataset),
            views=views,
            role=role,
        )

        prompt_dir = self.rootpath / "prompt" / dataset / "physics_aware"
        system_prompt = (prompt_dir / f"system_{role}.txt").read_text(encoding="utf-8")
        output_dir = self.rootpath / "LLM_query" / "physics_aware" / "roles" / dataset / role / self.llm_name
        structured_path = output_dir / "structured.json"

        payload = {}
        parsed = {}
        raw_specs = [("global", prompt_dir / f"{role}_global.txt")]
        raw_specs.extend((view, prompt_dir / f"{role}_view_{_safe_view_name(view)}.txt") for view in views)

        for logical_name, prompt_path in raw_specs:
            user_prompt = prompt_path.read_text(encoding="utf-8")
            answer = self._load_or_query_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                save_path=output_dir / f"{logical_name}.txt",
                overwrite=overwrite,
            )
            payload[logical_name] = answer
            parsed[logical_name] = self._parse_classwise_response(answer, classnames)

        result = {
            "dataset": dataset,
            "role": role,
            "llm_name": self.llm_name,
            "classnames": classnames,
            "views": ["global"] + views,
            "raw_files": {name: f"{name}.txt" for name in payload},
            "parsed": parsed,
        }
        save_json(str(structured_path), result)
        self._mirror_legacy_role_outputs(dataset, role, payload)
        return result

    def _load_or_query_transfer(self, source_dataset, target_dataset, overwrite=False):
        classnames = self._classnames(target_dataset)
        views = self._views(target_dataset)
        self.prompt_suite.build_transfer_prompts(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            source_domain=self._domain_text(source_dataset),
            target_domain=self._domain_text(target_dataset),
            classes=classnames,
            class_descriptions=self._class_descriptions(target_dataset),
            views=views,
        )

        prompt_dir = self.rootpath / "prompt" / f"{source_dataset}_to_{target_dataset}" / "physics_aware"
        system_prompt = (prompt_dir / "system_transfer.txt").read_text(encoding="utf-8")
        output_dir = (
            self.rootpath
            / "LLM_query"
            / "physics_aware"
            / "transfers"
            / f"{source_dataset}_to_{target_dataset}"
            / self.llm_name
        )
        structured_path = output_dir / "structured.json"

        parsed = {}
        for view in views:
            prompt_path = prompt_dir / f"transfer_view_{_safe_view_name(view)}.txt"
            user_prompt = prompt_path.read_text(encoding="utf-8")
            answer = self._load_or_query_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                save_path=output_dir / f"{view}.txt",
                overwrite=overwrite,
            )
            parsed[view] = self._parse_classwise_response(answer, classnames)

        result = {
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "llm_name": self.llm_name,
            "classnames": classnames,
            "views": views,
            "parsed": parsed,
        }
        save_json(str(structured_path), result)
        return result

    def _load_or_query_target_internal(self, target_dataset, overwrite=False):
        classnames = self._classnames(target_dataset)
        self.prompt_suite.build_target_internal_prompts(
            target_dataset=target_dataset,
            target_domain=self._domain_text(target_dataset),
            classes=classnames,
            class_descriptions=self._class_descriptions(target_dataset),
        )

        prompt_dir = self.rootpath / "prompt" / f"{target_dataset}_internal" / "physics_aware"
        system_prompt = (prompt_dir / "system_target_internal.txt").read_text(encoding="utf-8")
        user_prompt = (prompt_dir / "target_internal_counterfactual.txt").read_text(encoding="utf-8")
        output_dir = self.rootpath / "LLM_query" / "physics_aware" / "target_internal" / target_dataset / self.llm_name
        structured_path = output_dir / "structured.json"

        answer = self._load_or_query_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            save_path=output_dir / "target_internal_counterfactual.txt",
            overwrite=overwrite,
        )
        parsed = self._parse_classwise_response(answer, classnames)

        result = {
            "target_dataset": target_dataset,
            "llm_name": self.llm_name,
            "classnames": classnames,
            "parsed": parsed,
        }
        save_json(str(structured_path), result)

        legacy_dir = self.rootpath / "LLM_query" / "target_internal" / target_dataset / f"{self.llm_name}_view"
        mkdirp(str(legacy_dir))
        (legacy_dir / "target_internal_counterfactual.txt").write_text(answer, encoding="utf-8")
        return result

    def run_task(self, source_dataset, target_dataset, overwrite=False):
        source_payload = self._load_or_query_role(source_dataset, role="source", overwrite=overwrite)
        target_payload = self._load_or_query_role(target_dataset, role="target", overwrite=overwrite)
        transfer_payload = self._load_or_query_transfer(source_dataset, target_dataset, overwrite=overwrite)
        target_internal_payload = self._load_or_query_target_internal(target_dataset, overwrite=overwrite)

        task_dir = (
            self.rootpath
            / "LLM_query"
            / "physics_aware"
            / "tasks"
            / f"{source_dataset}_to_{target_dataset}"
            / self.llm_name
        )
        mkdirp(str(task_dir))
        summary = {
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "llm_name": self.llm_name,
            "source_role": source_payload,
            "target_role": target_payload,
            "transfer": transfer_payload,
            "target_internal": target_internal_payload,
        }
        save_json(str(task_dir / "semantic_query_summary.json"), summary)
        return summary
