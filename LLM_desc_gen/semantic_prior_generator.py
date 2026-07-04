import re
from pathlib import Path

from LLM_desc_gen.LLM import chatLLM
from utility.utils import mkdirp, read_lines, save_lines


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


class SemanticPriorGenerator:
    def __init__(self, rootpath, llm_name="gpt4o"):
        self.rootpath = Path(rootpath)
        self.llm_name = llm_name
        self.model_name = MODEL_ALIAS.get(llm_name, llm_name)

    def _domain_text(self, dataset):
        lines = _safe_read(self.rootpath / "prompt" / "aux_info" / f"domain_{dataset}.txt")
        return " ".join(lines)

    def _classnames(self, dataset):
        names = _safe_read(self.rootpath / "prompt" / "aux_info" / f"classnames_{dataset}.txt")
        return names

    def _render(self, template_name, **kwargs):
        template_path = self.rootpath / "prompt" / "semantic_bootstrap" / template_name
        text = template_path.read_text(encoding="utf-8")
        return text.format(**kwargs)

    def _query(self, system_prompt, user_prompt, save_path):
        mkdirp(str(Path(save_path).parent))
        llm = chatLLM(system_prompt)
        return llm.chat(user_prompt, model=self.model_name, save_path=str(save_path))

    def _parse_descriptions(self, text, classnames):
        desc_map = {}
        for line in text.splitlines():
            clean = line.strip().replace("**", "")
            match = re.match(r"^\d+\.\s*([^:]+):\s*(.+)$", clean)
            if match:
                desc_map[match.group(1).strip().lower()] = match.group(2).strip()

        outputs = []
        for name in classnames:
            desc = desc_map.get(name.lower(), f"{name} in GPR imagery.")
            outputs.append(f"{name}: {desc}")
        return outputs

    def _parse_views(self, text):
        views = []
        for line in text.splitlines():
            clean = line.strip().replace("**", "")
            match = re.match(r"^\d+\.\s*(.+)$", clean)
            if match:
                content = match.group(1).strip()
                if content:
                    views.append(content)
        dedup = []
        seen = set()
        for view in views:
            key = view.lower()
            if key not in seen:
                seen.add(key)
                dedup.append(view)
        return dedup

    def generate_class_descriptions(self, dataset, force=False):
        out_path = self.rootpath / "prompt" / "aux_info" / f"classes_description_{dataset}.txt"
        if out_path.exists() and not force and _safe_read(out_path):
            return _safe_read(out_path)

        classnames = self._classnames(dataset)
        domain = self._domain_text(dataset)
        numbered_classes = "\n".join([f"{idx + 1}. {name}" for idx, name in enumerate(classnames)])
        system_prompt = self._render("system_bootstrap.txt", domain=domain)
        user_prompt = self._render(
            "class_description_bootstrap.txt",
            dataset=dataset,
            domain=domain,
            numbered_classes=numbered_classes,
        )
        raw_path = self.rootpath / "LLM_query" / "semantic_bootstrap" / dataset / self.llm_name / "class_descriptions.txt"
        answer = self._query(system_prompt, user_prompt, raw_path)
        parsed = self._parse_descriptions(answer, classnames)
        save_lines(str(out_path), parsed)
        return parsed

    def generate_views(self, dataset, force=False):
        out_path = self.rootpath / "prompt" / "aux_info" / f"views_{dataset}.txt"
        if out_path.exists() and not force and _safe_read(out_path):
            return _safe_read(out_path)

        classnames = self._classnames(dataset)
        domain = self._domain_text(dataset)
        numbered_classes = "\n".join([f"{idx + 1}. {name}" for idx, name in enumerate(classnames)])
        system_prompt = self._render("system_bootstrap.txt", domain=domain)
        user_prompt = self._render(
            "view_bootstrap.txt",
            dataset=dataset,
            domain=domain,
            numbered_classes=numbered_classes
        )
        raw_path = self.rootpath / "LLM_query" / "semantic_bootstrap" / dataset / self.llm_name / "views.txt"
        answer = self._query(system_prompt, user_prompt, raw_path)
        parsed = self._parse_views(answer)
        numbered = [f"{idx + 1}. {view}" for idx, view in enumerate(parsed)]
        save_lines(str(out_path), numbered)
        return numbered

    def ensure_priors(self, dataset, force=False):
        class_desc = self.generate_class_descriptions(dataset, force=force)
        views = self.generate_views(dataset, force=force)
        return class_desc, views