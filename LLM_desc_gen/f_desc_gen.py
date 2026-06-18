from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
import clip
from tqdm import tqdm
from rich import print
import scipy.io as sio
from sentence_transformers import SentenceTransformer
from .LLM import chatLLM
from utility.utils import *
import torch
import re
import csv
from transformers import modeling_utils

class GenDesc:
    def __init__(self, dataset="SD", batch_size=10, model_name="gpt4o") -> None:
        self.dataset = dataset
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset2domain = {"SD": "GPR Subsurface Diagnosis, which uses GPR to collect images from concrete and asphalt roads", 
                               "Road": "GPR B-scan images collected from three different road surfaces: concrete, asphalt, and unpaved"}
        self.domain = self.dataset2domain[self.dataset]
        self.batch_size = batch_size
        self.class_set, self.view_set = self.load_class_view_set(self.dataset)
        self.view_set_plus = ['global'] + self.view_set

        if len(self.class_set) > self.batch_size:
            self.batch_num = (len(self.class_set) + self.batch_size - 1) // self.batch_size
            self.last_batch_size = len(self.class_set) % self.batch_size
            if self.last_batch_size == 0:
                self.last_batch_size = self.batch_size
        else:
            self.batch_num = 1
            self.last_batch_size = len(self.class_set)
            
    def load_class_view_set(self, dataset):
        class_set = read_lines(f"prompt/aux_info/classnames_{dataset}.txt")
        class_set = [classname.replace('_', ' ').replace("+", " ") for classname in class_set]
        view_set = read_lines(f"prompt/aux_info/views_{dataset}.txt")
        view_set = ['. '.join(view.split('. ')[1:]) for view in view_set]
        return class_set, view_set
            
    def gen_main_prompt(self):
        all_class_set = [self.class_set[i:i+self.batch_size]
                        for i in range(0, len(self.class_set), self.batch_size)]
        all_class_set_str = [f"{i+1}. {classname}" for i, classname in enumerate(self.class_set)]
        all_class_set_n = '\n'.join(all_class_set_str)
        numbered_class_set = [[f"{i+1}. {classname}" for i,
                            classname in enumerate(sub_class_set)] for sub_class_set in all_class_set]
        numbered_class_set_str = ['\n'.join(sub_class_set)
                                for sub_class_set in numbered_class_set]

        main_prompt_sys = '\n'.join(read_lines("prompt/main_prompt_sys.txt"))
        main_prompt_sys = main_prompt_sys.replace("[domain]", self.domain)

        mkdirp(f"prompt/{self.dataset}/view_generate")
        save_lines(f"prompt/{self.dataset}/view_generate/main_prompt_sys.txt", [main_prompt_sys])
        
        self.prompt_system = '\n'.join(read_lines(f"prompt/{self.dataset}/view_generate/main_prompt_sys.txt"))
        mkdirp(f"LLM_query/multi_view/{self.dataset}/{self.model_name}_view")
        
        for k, sub_class_set in enumerate(tqdm(numbered_class_set)):
            main_prompt_user_global = '\n'.join(
                read_lines("prompt/main_prompt_user_global.txt"))
            main_prompt_user_global = main_prompt_user_global.replace(
                "[domains]", f"{self.domain}s")
            main_prompt_user_global = main_prompt_user_global.replace(
                "[numbered_class_set]", numbered_class_set_str[k])
            main_prompt_user_global = main_prompt_user_global.replace(
                "[class 1]", all_class_set[k][0])
            main_prompt_user_global = main_prompt_user_global.replace(
                "[class 2]", all_class_set[k][1])
            save_lines(f"prompt/{self.dataset}/view_generate/main_prompt_global.txt", [main_prompt_user_global])

            print(
                f"Finished generating main_prompt_global for {self.dataset} and saved in prompt/{self.dataset}/view_generate.")

            for view in tqdm(self.view_set):
                main_prompt_user = '\n'.join(
                    read_lines("prompt/main_prompt_user_view.txt"))
                main_prompt_user = main_prompt_user.replace(
                    "[domains]", f"{self.domain}s")
                main_prompt_user = main_prompt_user.replace(
                    "[numbered_class_set]", numbered_class_set_str[k])
                main_prompt_user = main_prompt_user.replace(
                    "[view]", view.replace(']-[', ' - '))
                main_prompt_user = main_prompt_user.replace(
                    "[class 1]", all_class_set[k][0])
                main_prompt_user = main_prompt_user.replace(
                    "[class 2]", all_class_set[k][1])
                
                save_lines(
                    f"prompt/{self.dataset}/view_generate/main_prompt_{view}.txt", [main_prompt_user])

        print(
            f"Finished generating main_prompt_view for {self.dataset} and saved in prompt/{self.dataset}/view_generate.")

    def load_prompt(self):
        self.prompt_system = '\n'.join(read_lines(f"prompt/{self.dataset}/view_generate/main_prompt_sys.txt"))
        self.prompt_user_view = {}
        if self.batch_num > 1:
            for k in range(self.batch_num):
                self.prompt_user_view[f'global_{k:02d}'] = '\n'.join(
                    read_lines(f"prompt/{self.dataset}/view_generate/main_prompt_global_{k:02d}.txt"))
            for k in range(self.batch_num):
                for view in self.view_set:
                    self.prompt_user_view[f'{view}_{k:02d}'] = '\n'.join(
                        read_lines(f"prompt/{self.dataset}/view_generate/main_prompt_{view}_{k:02d}.txt"))
        else:
            self.prompt_user_view['global'] = '\n'.join(
                read_lines(f"prompt/{self.dataset}/view_generate/main_prompt_global.txt"))
            for view in self.view_set:
                self.prompt_user_view[view] = '\n'.join(read_lines(
                    f"prompt/{self.dataset}/view_generate/main_prompt_{view}.txt"))

    def init_local_LLM(self):
        modelname2model = {
            "llama-local": "meta-llama/Llama-3.1-8B-Instruct",
            "qwen-local": "Qwen/Qwen2.5-7B-Instruct"
        }
        model = modelname2model[self.model_name]
        model_path = f"/data/model/{model}"
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

    def gen_multiview_desc(self, query_view='all'):
        self.llm_chat = chatLLM(self.prompt_system)
        modelname2model = {
            "gpt4o": "gpt-4o-2024-08-06",
            "gpt4omini": "gpt-4o-mini-2024-07-18",
            "gemini2.5": "gemini-2.5-flash",
        }
        model = modelname2model[self.model_name]
        mkdirp(f"LLM_query/multi_view/{self.dataset}/{self.model_name}_view")
        if query_view != 'all':
            print(f"Requesting {query_view}...")
            self.llm_chat.chat(
                self.prompt_user_view[query_view],
                model=model,
                save_path=f"LLM_query/multi_view/{self.dataset}/{self.model_name}_view/{query_view}.txt")
        else:
            for view_name, view_prompt in self.prompt_user_view.items():
                if not fileExist(f"LLM_query/multi_view/{self.dataset}/{self.model_name}_view/{view_name}.txt"):
                    print(f"Requesting {view_name}...")
                    self.llm_chat.chat(
                        view_prompt,
                        model=model,
                        save_path=f"LLM_query/multi_view/{self.dataset}/{self.model_name}_view/{view_name}.txt")

    def gen_cf_desc(self):
        g_merge_desc = []
        for k in range(self.batch_num):
            g_view_desc = '\n'.join(read_lines(f"LLM_query/multi_view/{self.dataset}/{self.model_name}_view/global_{k:02d}.txt")).replace('**','')
            matches = re.findall(r"^\d+\.\s+(.*)", g_view_desc, re.MULTILINE)
            if matches:
                g_merge_desc.extend(matches)
            else:
                file_path = 'error.txt'
                error_str = f"Error! LLM_query/multi_view/{self.dataset}/{self.model_name}_view/global.txt"
                with open(file_path, 'w', encoding='utf-8') as file_object:
                        file_object.write(error_str)
                print(f"Error! LLM_query/multi_view/{self.dataset}/{self.model_name}_view/global.txt")
            save_lines(f"LLM_query/multi_view/{self.dataset}/{self.model_name}_view/global.txt", g_merge_desc)

        self.prompt_system = '\n'.join(read_lines(
            f"prompt_view/{self.dataset}/view_generate/main_prompt_sys.txt"))
        self.llm_chat = chatLLM(self.prompt_system)
        modelname2model = {
            "gpt4o": "gpt-4o-2024-08-06",
            "gpt4omini": "gpt-4o-mini-2024-07-18",
            "gemini2.5": "gemini-2.5-flash",
        }
        model = modelname2model[self.model_name]
        mkdirp(f"prompt_view/{self.dataset}/cf_class_generate")
        mkdirp(f"prompt_view/{self.dataset}/cf_class_merge")
        mkdirp(f"LLM_query/cf_class_generate/{self.dataset}/{self.model_name}_view")
        mkdirp(f"LLM_query/cf_class_merge/{self.dataset}/{self.model_name}_view")

        self.class_name = []
        self.class_desc = []
        for g_desc in tqdm(g_merge_desc):
            g_desc_parts = g_desc.split(':', 1)
            class_name = g_desc_parts[0].strip().replace('**','')
            class_desc = g_desc_parts[1].strip()
            self.class_name.append(class_name)
            self.class_desc.append(class_desc)
            # multi counterfactuals
            main_prompt_user_global_cf = '\n'.join(
                        read_lines("prompt_view/main_prompt_user_global_cf.txt"))
            main_prompt_user_global_cf = main_prompt_user_global_cf.replace(
                        "[domains]", f"{self.domain}s")
            main_prompt_user_global_cf = main_prompt_user_global_cf.replace(
                        "[categories]", f"{', '.join(self.class_set)}.")
            main_prompt_user_global_cf = main_prompt_user_global_cf.replace(
                        "[Class name]", f"{class_name}")
            main_prompt_user_global_cf = main_prompt_user_global_cf.replace(
                        "[Global description text]", f"{class_desc}")
            save_lines(f"prompt_view/{self.dataset}/cf_class_generate/main_prompt_[{class_name}]_cf.txt", [main_prompt_user_global_cf])
            
            if not fileExist(f"LLM_query/cf_class_generate/{self.dataset}/{self.model_name}_view/[{class_name}]_cf.txt"):
                print(f"Requesting [{class_name}]...")
                self.llm_chat.chat(
                    main_prompt_user_global_cf,
                    model=model,
                    save_path=f"LLM_query/cf_class_generate/{self.dataset}/{self.model_name}_view/[{class_name}]_cf.txt")

            # signal counterfactual
            # main_prompt_user_global_cf_merge = '\n'.join(
            #             read_lines("prompt_view/main_prompt_user_global_cf_merge.txt"))
            # main_prompt_user_global_cf_merge = main_prompt_user_global_cf_merge.replace(
            #             "[domains]", f"{self.domain}s")
            # main_prompt_user_global_cf_merge = main_prompt_user_global_cf_merge.replace(
            #             "[categories]", f"{', '.join(self.class_set)}.")
            # main_prompt_user_global_cf_merge = main_prompt_user_global_cf_merge.replace(
            #             "[Class name]", f"{class_name}")
            # main_prompt_user_global_cf_merge = main_prompt_user_global_cf_merge.replace(
            #             "[Global description text]", f"{class_desc}")
            # save_lines(f"prompt_view/{self.dataset}/cf_class_merge/main_prompt_[{class_name}]_cf.txt", [main_prompt_user_global_cf_merge])
        
            # if not fileExist(f"LLM_query/cf_class_merge/{self.dataset}/{self.model_name}_view/[{class_name}]_cf.txt"):
            #     print(f"Requesting [{class_name}]...")
            #     self.llm_chat.chat(
            #         main_prompt_user_global_cf_merge,
            #         model=model,
            #         save_path=f"LLM_query/cf_class_merge/{self.dataset}/{self.model_name}_view/[{class_name}]_cf.txt")

    def gen_local_desc(self, query_view='all'):
        mkdirp(f"LLM/{self.dataset}/{self.model_name}_view")
        prompt_user_view = self.prompt_user_view[query_view] if query_view != 'all' else self.prompt_user_view
        for view_name, view_prompt in prompt_user_view.items():
            print(f"Requesting {view_name}...")
            text = self.tokenizer.apply_chat_template(
                view_prompt, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer(
                [text], return_tensors="pt").to(self.device)
            
            generated_ids = self.llm.generate(model_inputs.input_ids, max_new_tokens=5000)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            save_path = f"LLM_query/multi_view/{self.dataset}/{self.model_name}_view/{view_name}.txt"
            save_lines(save_path, [answer])

    def merge_cf_desc(self):

        mkdirp(f"LLM_query/data_collation/{self.dataset}/cf_class_collation")
        self.class_desc_cf = []
        pattern = re.compile(r"^(?:\d+\.\s*)?(CF_.*?):(.*)$")
        for name in self.class_name:
            line_list = read_lines(f"LLM_query/cf_class_generate/{self.dataset}/{self.model_name}_view/[{name}]_cf.txt")
            desc_cf_list = []
            for line in line_list:
                if line == '':
                    continue
                else:
                    line = line.replace("**", "").replace("- ","")
                    match = re.match(pattern, line)
                    if match:
                        class_name = match.group(1).strip()
                        description = match.group(2).strip()
                        desc_cf_list.append(description)
            # if len(desc_cf_list) != 10:
            #     print(name)
            self.class_desc_cf.append(desc_cf_list)
        
        self.line_list = []
        for name in self.class_name:
            line = read_lines(f"LLM_query/cf_class_merge/{self.dataset}/{self.model_name}_view/[{name}]_cf.txt")
            self.line_list.append(line[0])
        save_lines(f"LLM_query/data_collation/{self.dataset}/cf_class_collation/{self.model_name}_cf_global.txt", ['\n'.join(self.line_list)])
        

    def merge_true_desc(self):

        self.class_view_desc = [['' for _ in range(len(self.view_set_plus))] for _ in range(len(self.class_set))]
        for j, view in enumerate(self.view_set_plus):
            view_desc = read_lines(f"LLM_query/multi_view/{self.dataset}/{self.model_name}_view/{view}.txt")
            view_desc = [desc for desc in view_desc if desc.strip() != ""]
            view_desc = [desc.split(': ', 1)[-1] for desc in view_desc]
            for i, desc in enumerate(self.class_set):
                # print(view)
                self.class_view_desc[i][j] = view_desc[i]

    def text_embedding(self, emb_modelname):
        
        mkdirp(f"embeddings/{emb_modelname}")
        if emb_modelname == "clip":
            emb_model, preprocess = clip.load("RN101", device=self.device)
        elif emb_modelname == "sbert":
            emb_model = SentenceTransformer(
                f'/data/model/sentence-transformers/all-mpnet-base-v2', device=self.device)
        elif emb_modelname == "qwen":
            tokenizer = AutoTokenizer.from_pretrained(
                f"/data/model/Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
                modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]
            emb_model = AutoModel.from_pretrained(
                f"/data/model/Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto", device_map="auto", trust_remote_code=True)
        elif emb_modelname == "llama":
            tokenizer = AutoTokenizer.from_pretrained(
                f"/data/model/meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
                modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]
            emb_model = AutoModel.from_pretrained(
                f"/data/model/meta-llama/Llama-3.1-8B-Instruct", torch_dtype="auto", device_map="auto", trust_remote_code=True)
        else:
            raise ValueError(f"Unknown embedding model: {emb_modelname}")
        
        # base multi views
        text_features = []
        for i, class_ in enumerate(tqdm(self.class_set)):
            desc = self.class_view_desc[i]
            if emb_modelname == "clip":
                tokenized_desc = clip.tokenize(
                    desc, truncate=True).to(self.device)
                with torch.no_grad():
                    desc_embedding = emb_model.encode_text(
                        tokenized_desc).cpu().numpy()
            elif emb_modelname == "sbert":
                desc_embedding = emb_model.encode(
                    desc, show_progress_bar=False)
            else:
                model_inputs = tokenizer(desc, return_tensors="pt",
                                        padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = emb_model(model_inputs.input_ids,
                                        output_hidden_states=True, return_dict=True)
                    hidden_states = outputs.hidden_states[-1].to(
                        dtype=torch.float32)
                    desc_embedding = hidden_states.mean(
                        dim=1).squeeze().cpu().numpy()
            text_features.append(desc_embedding)
        text_features = np.array(text_features)
        if emb_modelname == "clip":
            save_path = f"embeddings/clip/{self.dataset}_{self.model_name}_clip.npy"
        elif emb_modelname == "sbert":
            save_path = f"embeddings/sbert/{self.dataset}_{self.model_name}_sbert.npy"
        elif emb_modelname == "qwen":
            save_path = f"embeddings/qwen/{self.dataset}_{self.model_name}_qwen.npy"
        elif emb_modelname == "llama":
            save_path = f"embeddings/llama/{self.dataset}_{self.model_name}_llama.npy"
        np.save(save_path, text_features)
        print(f"Embedding saved at {save_path}")

if __name__ == "__main__":
    dataset_set = ["Road","SD"]
    model_set = ["gpt4o"]
    embedding_model_set = ["llama","qwen","llama","clip"]
    for dataset in dataset_set:
        for modelname in model_set:
            gen_desc = GenDesc(dataset, model_name=modelname)
            gen_desc.gen_main_prompt()
            gen_desc.load_prompt()
            # gen_desc.init_local_LLM()
            gen_desc.gen_multiview_desc()
            # gen_desc.gen_cf_desc()
            # gen_desc.gen_local_desc()
            # gen_desc.merge_cf_desc()
            gen_desc.merge_true_desc()
            for embedding_model in embedding_model_set:
                gen_desc.text_embedding(embedding_model)