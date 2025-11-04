import torch
from .model import Model
from .utils import sample_token, get_last_attn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttentionModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.name = config["model_info"]["name"]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        model_id = config["model_info"]["model_id"]

        # Thêm cờ để kiểm soát việc sử dụng system prompt
        self.use_system_prompt = config["model_info"].get("use_system_prompt", True)  # Mặc định là True

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",
        ).eval()

        self.top_k = 50
        self.top_p = None

        if config["params"]["important_heads"] == "all":
            attn_size = self.get_map_dim()
            self.important_heads = [[i, j] for i in range(attn_size[0]) for j in range(attn_size[1])]
        else:
            self.important_heads = config["params"]["important_heads"]

    def get_map_dim(self):
        _, _, attention_maps, _, _, _ = self.inference("print hi", "")
        attention_map = attention_maps[0]
        return len(attention_map), attention_map[0].shape[1]

    def inference(self, instruction, data, max_output_tokens=None):
        # Logic tạo template chat linh hoạt
        if self.use_system_prompt:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": "Data: " + data}
            ]
        else:
            full_content = instruction + "\n" + "Data: " + data
            messages = [
                {"role": "user", "content": full_content},
            ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        instruction_len = len(self.tokenizer.encode(instruction))
        data_len = len(self.tokenizer.encode("Data: " + data))  # Thêm "Data: " để nhất quán

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_tokens = self.tokenizer.convert_ids_to_tokens(model_inputs['input_ids'][0])

        # Hợp nhất logic xác định data_range
        if "qwen" in self.name:
            data_range = ((3, 3 + instruction_len), (-5 - data_len, -5))
        elif "phi3" in self.name:
            data_range = ((1, 1 + instruction_len), (-2 - data_len, -2))
        elif "llama3-8b" in self.name:
            data_range = ((5, 5 + instruction_len), (-5 - data_len, -5))
        elif "mistral-7b" in self.name:
            data_range = ((3, 3 + instruction_len), (-1 - data_len, -1))
        elif "granite3-8b" in self.name:
            data_range = ((3, 3 + instruction_len), (-5 - data_len, -5))
        elif "gemma2_9b-attn" in self.name:
            # Logic này từ attn_model_nsys.py
            data_range = ((5, 5 + instruction_len), (-4 - data_len, -5))
        else:
            # Cung cấp một giá trị mặc định hoặc báo lỗi để dễ dàng gỡ lỗi
            print(f"Warning: data_range not implemented for model '{self.name}'. Using fallback.")
            # Fallback (cần kiểm tra cẩn thận)
            data_range = ((1, 1 + instruction_len), (-1 - data_len, -1))

        # --- Phần còn lại của hàm inference giữ nguyên ---
        generated_tokens = []
        generated_probs = []
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        attention_maps = []

        n_tokens = max_output_tokens if max_output_tokens is not None else self.max_output_tokens

        with torch.no_grad():
            for i in range(n_tokens):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True
                )
                logits = output.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token_id = sample_token(logits[0], top_k=self.top_k, top_p=self.top_p, temperature=1.0)[0]
                generated_probs.append(probs[0, next_token_id.item()].item())
                generated_tokens.append(next_token_id.item())

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0).unsqueeze(0)), dim=-1)
                attention_mask = torch.cat((attention_mask, torch.tensor([[1]], device=input_ids.device)), dim=-1)
                attention_map = [attention.detach().cpu().half() for attention in output['attentions']]
                attention_map = [torch.nan_to_num(attention, nan=0.0) for attention in attention_map]
                attention_map = get_last_attn(attention_map)
                attention_maps.append(attention_map)

        output_tokens = [self.tokenizer.decode(token, skip_special_tokens=True) for token in generated_tokens]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text, output_tokens, attention_maps, input_tokens, data_range, generated_probs