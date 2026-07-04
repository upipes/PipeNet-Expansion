from pathlib import Path

from utility.utils import mkdirp, read_lines
from LLM_desc_gen.semantic_prior_generator import SemanticPriorGenerator


class PhysicsAwarePromptSuite:
    def __init__(self, rootpath):
        self.rootpath = Path(rootpath)
        self.template_dir = self.rootpath / "prompt" / "physics_aware"
        self.prior_generator = SemanticPriorGenerator(rootpath)

    def _read_template(self, name):
        path = self.template_dir / name
        lines = read_lines(str(path))
        if lines is None:
            raise FileNotFoundError(f"Missing template: {path}")
        return "\n".join(lines)

    def render(self, name, **kwargs):
        template = self._read_template(name)
        return template.format(**kwargs)

    def build_dataset_prompts(self, dataset, domain, classes, class_descriptions, views, role):
        if not class_descriptions or not views:
            generated_desc, generated_views = self.prior_generator.ensure_priors(dataset)
            if not class_descriptions:
                class_descriptions = {}
                for line in generated_desc:
                    if ":" in line:
                        name, desc = line.split(":", 1)
                        class_descriptions[name.strip().lower()] = desc.strip()
            if not views:
                views = [line.split(". ", 1)[1].strip() if ". " in line else line.strip() for line in generated_views]

        out_dir = self.rootpath / "prompt" / dataset / "physics_aware"
        mkdirp(str(out_dir))

        numbered_classes = "\n".join([f"{idx + 1}. {name}" for idx, name in enumerate(classes)])
        class_block = "\n".join(
            [f"- {name}: {class_descriptions.get(name.lower(), '')}" for name in classes]
        )

        system_prompt = self.render("system_source.txt", domain=domain)
        (out_dir / f"system_{role}.txt").write_text(system_prompt, encoding="utf-8")

        if role == "source":
            global_name = "source_global.txt"
            global_template = "source_global.txt"
            view_template = "source_view.txt"
            view_prefix = "source_view_"
        else:
            global_name = "target_global.txt"
            global_template = "target_global.txt"
            view_template = "target_view.txt"
            view_prefix = "target_view_"

        role_global = self.render(
            global_template,
            domain=domain,
            dataset=dataset,
            numbered_classes=numbered_classes,
            class_block=class_block,
        )
        (out_dir / global_name).write_text(role_global, encoding="utf-8")

        for view in views:
            safe_view = f"[{view.replace(' - ', ']-[')}]" if " - " in view and not view.startswith("[") else view
            role_view = self.render(
                view_template,
                domain=domain,
                dataset=dataset,
                view=view,
                numbered_classes=numbered_classes,
                class_block=class_block,
            )
            (out_dir / f"{view_prefix}{safe_view}.txt").write_text(role_view, encoding="utf-8")

    def build_transfer_prompts(
        self,
        source_dataset,
        target_dataset,
        source_domain,
        target_domain,
        classes,
        class_descriptions,
        views,
    ):
        out_dir = self.rootpath / "prompt" / f"{source_dataset}_to_{target_dataset}" / "physics_aware"
        mkdirp(str(out_dir))

        numbered_classes = "\n".join([f"{idx + 1}. {name}" for idx, name in enumerate(classes)])
        class_block = "\n".join(
            [f"- {name}: {class_descriptions.get(name.lower(), '')}" for name in classes]
        )

        system_prompt = self.render(
            "system_transfer.txt",
            source_domain=source_domain,
            target_domain=target_domain,
        )
        (out_dir / "system_transfer.txt").write_text(system_prompt, encoding="utf-8")

        for view in views:
            safe_view = f"[{view.replace(' - ', ']-[')}]" if " - " in view and not view.startswith("[") else view
            transfer_view = self.render(
                "intervention_transfer.txt",
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                source_domain=source_domain,
                target_domain=target_domain,
                view=view,
                numbered_classes=numbered_classes,
                class_block=class_block,
            )
            (out_dir / f"transfer_view_{safe_view}.txt").write_text(transfer_view, encoding="utf-8")

    def build_target_internal_prompts(
        self,
        target_dataset,
        target_domain,
        classes,
        class_descriptions,
    ):
        out_dir = self.rootpath / "prompt" / f"{target_dataset}_internal" / "physics_aware"
        mkdirp(str(out_dir))

        numbered_classes = "\n".join([f"{idx + 1}. {name}" for idx, name in enumerate(classes)])
        class_block = "\n".join(
            [f"- {name}: {class_descriptions.get(name.lower(), '')}" for name in classes]
        )

        system_prompt = self.render(
            "system_target_internal.txt",
            target_domain=target_domain,
            target_dataset=target_dataset,
        )
        (out_dir / "system_target_internal.txt").write_text(system_prompt, encoding="utf-8")
        prompt_text = self.render(
            "target_internal_counterfactual.txt",
            target_dataset=target_dataset,
            target_domain=target_domain,
            numbered_classes=numbered_classes,
            class_block=class_block,
        )
        (out_dir / "target_internal_counterfactual.txt").write_text(prompt_text, encoding="utf-8")
