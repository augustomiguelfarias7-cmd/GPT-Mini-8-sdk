# gpt_mini8_local_sdk.py
# GPT-Mini 8 SDK LOCAL — servidor Flask + cliente + código do modelo + tokenizer
# Autor: Augusto (esqueleto pronto para integração)
# ---------------------------------------------------
# Como usar (resumo):
# 1) Ajuste CONFIG abaixo (DEVICE, PORT etc).
# 2) Rode "python gpt_mini8_local_sdk.py --start-server" para iniciar o servidor local.
# 3) Em outro terminal (ou no código), crie GPTMini8Client() e chame generate_text / generate_cot / generate_image.
# ---------------------------------------------------

import os
import io
import time
import json
import base64
import threading
import argparse
import logging
from typing import Optional, List, Dict, Any

# ---- dependências externas ----
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tokenizers import Tokenizer as HFTokenizer, models, trainers, pre_tokenizers
    from flask import Flask, request, jsonify, send_file
    import requests
    from PIL import Image
    import numpy as np
    import torchvision.transforms as T
    HAS_ALL = True
except Exception as e:
    # Dependências podem faltar — o servidor/cliente instrui como instalar
    HAS_ALL = False
    _IMPORT_ERROR = e

# logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gpt-mini8-sdk")

# ---------------------------
# CONFIG (edite conforme necessário)
# ---------------------------
CONFIG = {
    "DEVICE": "cuda" if (HAS_ALL and torch.cuda.is_available()) else "cpu",
    "SERVER_HOST": "127.0.0.1",
    "SERVER_PORT": 7860,
    "TOKENIZER_PATH": "tokenizer_gpt_mini8.json",
    "CHECKPOINT_DIR": "checkpoints",
    # alteração: DEFAULT_MAX_TOKENS agora é 1_000_000
    "DEFAULT_MAX_TOKENS": 1_000_000,
    "DEFAULT_COT_STEPS": 3,
    "DEFAULT_COT_TOKENS_PER_STEP": 32,
    # limite prudente para evitar estouro acidental (pode ajustar)
    "MAX_ALLOWED_TOKENS": 1_000_000,
}
# ---------------------------

# mensagens de dependência
if not HAS_ALL:
    log.error("Faltam bibliotecas. Instale: torch tokenizers flask requests pillow torchvision datasets numpy")
    log.error(f"Erro detectado: {_IMPORT_ERROR}")

# ---------------------------
# Tokenizer (BPE) — RealTokenizer
# ---------------------------
class RealTokenizer:
    """
    Wrapper simples em torno de tokenizers.Tokenizer (BPE).
    Se o arquivo não existir, treina um tokenizer básico com SAMPLE_TEXTS e salva.
    """
    SAMPLE_TEXTS = [
        "Olá mundo",
        "GPT-Mini 8 é um modelo multimodal.",
        "Treinar tokenizer localmente permite uso offline.",
        "Exemplo: Aprendizado de máquina, redes neurais e transformação.",
        "Transformers, atenção, embeddings e inferência.",
        "O cachorro correu pelo parque.",
        "A Inteligência Artificial está mudando o mundo.",
        "Como instalar dependências: pip install -r requirements.txt",
        "JSON, HTTP, Flask, servidor local, API REST.",
        "Imagem, áudio, vídeo e texto são modalidades multimodais.",
        "Teste rápido de tokenização e decodificação.",
        "Robô futurista voando sobre a cidade à noite.",
        "Desenvolvedor: Augusto Miguel de Farias.",
        "Tokenization é importante para modelos de linguagem.",
        "Passo a passo: encode, decode, gerar texto.",
        "Exemplo em Português e English mixed tokens.",
        "Matemática básica: 2 + 2 = 4.",
        "Prova que sqrt(2) é irracional - esboço.",
        "Prompt: descreva um pôr do sol em Porto Alegre.",
        "Fim das amostras."
    ]

    def __init__(self, vocab_size: int = 50000, path: Optional[str] = None, auto_train: bool = True):
        self.vocab_size = vocab_size
        self.path = path or CONFIG["TOKENIZER_PATH"]
        self.tok = None
        self.trainer = None

        if not HAS_ALL:
            # fallback minimal (não operacional) para evitar crashes se libs faltarem
            raise RuntimeError("Dependências faltando (tokenizers). Instale a biblioteca 'tokenizers'.")

        # se arquivo existir, carrega
        if os.path.exists(self.path):
            try:
                self.tok = HFTokenizer.from_file(self.path)
                log.info(f"Tokenizer carregado de {self.path}")
                return
            except Exception as e:
                log.warning(f"Falha ao carregar tokenizer salvo ({self.path}): {e}. Irei recriar.")

        # cria novo tokenizer BPE e trainer
        self.tok = HFTokenizer(models.BPE(unk_token="[UNK]"))
        self.tok.pre_tokenizer = pre_tokenizers.Whitespace()
        self.trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"])

        # se pediu auto_train e não existe arquivo, treina com SAMPLE_TEXTS básicos
        if auto_train and not os.path.exists(self.path):
            try:
                log.info("Treinando tokenizer BPE com amostras internas (auto_train)...")
                self.tok.train_from_iterator(self.SAMPLE_TEXTS, self.trainer)
                self.save(self.path)
                log.info(f"Tokenizer treinado e salvo em {self.path}")
            except Exception as e:
                log.warning(f"Falha ao treinar tokenizer automaticamente: {e}")

    def train_from_iterator(self, iterator):
        """Treina tokenizer BPE a partir de um iterador de textos (usuário)."""
        if not HAS_ALL:
            raise RuntimeError("Dependências faltando (tokenizers).")
        log.info("Treinando tokenizer BPE com iterator fornecido...")
        self.tok.train_from_iterator(iterator, self.trainer)
        self.save(self.path)

    def encode(self, text: str) -> List[int]:
        if not HAS_ALL:
            raise RuntimeError("Dependências faltando (tokenizers).")
        if not text:
            return []
        enc = self.tok.encode(text if isinstance(text, str) else str(text))
        return enc.ids

    def decode(self, ids: List[int]) -> str:
        if not HAS_ALL:
            raise RuntimeError("Dependências faltando (tokenizers).")
        if not ids:
            return ""
        return self.tok.decode(ids)

    def save(self, path: Optional[str] = None):
        if not HAS_ALL:
            raise RuntimeError("Dependências faltando (tokenizers).")
        path = path or self.path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.tok.save(path)
        log.info(f"Tokenizer salvo em {path}")

    @classmethod
    def load(cls, path: str):
        if not HAS_ALL:
            raise RuntimeError("Dependências faltando (tokenizers).")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        inst = cls(1, path, auto_train=False)
        inst.tok = HFTokenizer.from_file(path)
        return inst

# ---------------------------
# Modelo (esqueleto multimodal)
# - projeto para receber pesos reais; serve como referência funcional
# ---------------------------
if HAS_ALL:
    class TransformerBlock(nn.Module):
        def __init__(self, d_model, nhead, dim_ff):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            self.ln1 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model))
            self.ln2 = nn.LayerNorm(d_model)
        def forward(self, x):
            a,_ = self.attn(x,x,x)
            x = self.ln1(x + a)
            x = self.ln2(x + self.ff(x))
            return x

    class VQVAE_Simple(nn.Module):
        def __init__(self, in_channels=3, latent_c=64, codebook_size=4096):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(in_channels, 128, 4, 2, 1), nn.ReLU(),
                nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
                nn.Conv2d(256, latent_c, 3, 1, 1)
            )
            self.codebook = nn.Embedding(codebook_size, latent_c)
            nn.init.normal_(self.codebook.weight, std=0.02)
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(latent_c, 256, 3, 1, 1), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(128, in_channels, 4, 2, 1), nn.Sigmoid()
            )
        def encode(self, x):
            return self.enc(x)
        def quantize(self, z):
            B,C,H,W = z.shape
            flat = z.permute(0,2,3,1).contiguous().view(-1, C)  # (N, C)
            dists = torch.cdist(flat, self.codebook.weight)  # (N, K)
            ids = dists.argmin(dim=1)
            quant = F.embedding(ids, self.codebook.weight).view(B, H, W, C).permute(0,3,1,2).contiguous()
            return quant, ids.view(B,H,W)
        def decode(self, z_q):
            return self.dec(z_q)
        def forward(self, x):
            z = self.encode(x)
            z_q, ids = self.quantize(z)
            recon = self.decode(z_q)
            return recon, z, z_q, ids

    class SimpleUNet(nn.Module):
        def __init__(self, in_ch=64, base=128, cond_dim=384):
            super().__init__()
            self.enc1 = nn.Conv2d(in_ch, base, 3, 1, 1)
            self.enc2 = nn.Conv2d(base, base*2, 3, 2, 1)
            self.mid = nn.Conv2d(base*2, base*2, 3, 1, 1)
            self.dec2 = nn.ConvTranspose2d(base*2, base, 4, 2, 1)
            self.dec1 = nn.Conv2d(base, in_ch, 3, 1, 1)
            self.cond = nn.Linear(cond_dim, base*2)
        def forward(self, x, t, cond_emb=None):
            e1 = F.relu(self.enc1(x))
            e2 = F.relu(self.enc2(e1))
            m = F.relu(self.mid(e2))
            if cond_emb is not None:
                p = self.cond(cond_emb).unsqueeze(-1).unsqueeze(-1)
                m = m + p
            d2 = F.relu(self.dec2(m))
            out = torch.sigmoid(self.dec1(d2 + e1))
            return out

    class GPTMini8Model(nn.Module):
        """
        Esqueleto do modelo multimodal.
        Substitua / carregue pesos reais para ter desempenho de verdade.
        """
        def __init__(self, vocab_size=50_000, d_model=384, depth=6, heads=6, max_seq=512):
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.max_seq = max_seq
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, max_seq, d_model))
            self.blocks = nn.ModuleList([TransformerBlock(d_model, heads, d_model*4) for _ in range(depth)])
            self.ln = nn.LayerNorm(d_model)
            self.text_head = nn.Linear(d_model, vocab_size)
            self.reason_head = nn.Linear(d_model, vocab_size)
            self.code_head = nn.Linear(d_model, vocab_size)
            # multimodal parts
            self.vqvae = VQVAE_Simple(in_channels=3, latent_c=64, codebook_size=4096)
            self.unet = SimpleUNet(in_ch=64, cond_dim=d_model)
            # small audio/video projections
            self.audio_proj = nn.Linear(256, d_model)
            self.video_proj = nn.Linear(16*3*32*32, d_model)

        def encode_tokens(self, token_ids: torch.LongTensor):
            # token_ids: (B, L)
            x = self.token_emb(token_ids) + self.pos_emb[:, :token_ids.size(1), :]
            for b in self.blocks:
                x = b(x)
            x = self.ln(x)
            return x

        def text_logits(self, token_ids: torch.LongTensor):
            x = self.encode_tokens(token_ids)
            return self.text_head(x)

        def reason_logits(self, token_ids: torch.LongTensor):
            x = self.encode_tokens(token_ids)
            return self.reason_head(x)

        # Geração autoregressiva simples (usa sampling por top_k/top_p)
        def generate_text(self, tokenizer, prompt: str, max_new_tokens:int=150, temperature:float=0.7, top_k:int=50, top_p:float=0.95, device:Optional[str]=None):
            device = device or CONFIG["DEVICE"]
            self.eval()
            ids = tokenizer.encode(prompt)
            if len(ids) == 0:
                ids = [0]
            cur = torch.tensor([ids], dtype=torch.long, device=device)
            generated = list(ids)
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    logits = self.text_logits(cur)  # (B,L,V)
                    last = logits[:, -1, :].squeeze(0)  # (V,)
                    # sampling
                    if temperature == 0.0:
                        next_id = int(last.argmax().item())
                    else:
                        logits_scaled = last / max(1e-8, temperature)
                        if top_k > 0:
                            vals, idx = torch.topk(logits_scaled, top_k)
                            minv = vals[-1]
                            logits_scaled = torch.where(logits_scaled < minv, torch.tensor(-1e10, device=logits_scaled.device), logits_scaled)
                        probs = F.softmax(logits_scaled, dim=-1)
                        next_id = int(torch.multinomial(probs, num_samples=1).item())
                    generated.append(next_id)
                    cur = torch.cat([cur, torch.tensor([[next_id]], device=device)], dim=1)
            return tokenizer.decode(generated)

        def generate_reasoning_step(self, tokenizer, input_ids:List[int], max_new:int=32, temperature:float=0.0, device:Optional[str]=None):
            device = device or CONFIG["DEVICE"]
            self.eval()
            cur = torch.tensor([input_ids], dtype=torch.long, device=device)
            gen = list(input_ids)
            with torch.no_grad():
                for _ in range(max_new):
                    logits = self.reason_logits(cur)
                    last = logits[:, -1, :].squeeze(0)
                    if temperature == 0.0:
                        nxt = int(last.argmax().item())
                    else:
                        probs = F.softmax(last/temperature, dim=-1)
                        nxt = int(torch.multinomial(probs, num_samples=1).item())
                    gen.append(nxt)
                    cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
            return gen

        # Imagem: encode prompt -> cond emb -> sample UNet -> decode via VQ-VAE
        def sample_image_from_text(self, tokenizer, prompt: str, steps:int=40, device:Optional[str]=None):
            device = device or CONFIG["DEVICE"]
            self.eval()
            ids = tokenizer.encode(prompt)
            if not ids:
                ids = [0]
            with torch.no_grad():
                token_tensor = torch.tensor([ids], dtype=torch.long, device=device)
                enc = self.encode_tokens(token_tensor)  # (B,L,D)
                cond = enc.mean(dim=1)  # (B,D)
                # start noise latent
                B=1; C=64; H=16; W=16
                x = torch.randn(B,C,H,W, device=device)
                # simple reverse loop (not proper DDPM but demonstrative)
                for i in reversed(range(min(steps,50))):
                    t = torch.tensor([i], device=device)
                    pred = self.unet(x, t, cond)
                    # simple denoising step
                    x = x - 0.1 * pred + 0.01 * torch.randn_like(x)
                # decode with vqvae.decode (expects latent shape)
                try:
                    recon = self.vqvae.decode(x)
                except Exception:
                    # if shapes mismatch, do a naive upsample
                    recon = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)
                    # bring to 3 channels
                    if recon.shape[1] != 3:
                        recon = torch.cat([recon[:, :3, :, :]] * 1, dim=1)
                recon = recon.clamp(0,1)
                return recon  # tensor (B,3,H',W')
# end HAS_ALL

# ---------------------------
# Servidor Flask que expõe o modelo localmente
# ---------------------------
class ModelServer:
    """
    Roda um servidor Flask que expõe endpoints REST para geração.
    O servidor mantém um objeto `model` e `tokenizer`. Você pode carregar pesos via /load_model.
    Use start() para rodar em background thread.
    """
    def __init__(self, host:str = CONFIG["SERVER_HOST"], port:int = CONFIG["SERVER_PORT"], model:Optional[Any]=None, tokenizer:Optional[RealTokenizer]=None, tokenizer_path:Optional[str]=None):
        if not HAS_ALL:
            raise RuntimeError(f"Dependências faltando: {_IMPORT_ERROR}")
        from flask import Flask, request, jsonify
        self.app = Flask("gpt_mini8_local_server")
        self.host = host
        self.port = port
        self.model = model or GPTMini8Model()
        # se tokenizer fornecido, usa; se não, carrega de path ou cria e treina
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            tk_path = tokenizer_path or CONFIG["TOKENIZER_PATH"]
            if os.path.exists(tk_path):
                try:
                    self.tokenizer = RealTokenizer.load(tk_path)
                except Exception as e:
                    log.warning(f"Falha ao carregar tokenizer de {tk_path}: {e}. Criando novo.")
                    self.tokenizer = RealTokenizer(path=tk_path, auto_train=True)
            else:
                # cria e treina com SAMPLE_TEXTS
                self.tokenizer = RealTokenizer(path=tk_path, auto_train=True)
        self._thread = None
        self._server_running = False
        self._lock = threading.Lock()
        self._setup_routes()

    def _setup_routes(self):
        app = self.app

        @app.route("/status", methods=["GET"])
        def status():
            return jsonify({"ok": True, "model_loaded": self.model is not None, "tokenizer": bool(self.tokenizer)})

        @app.route("/tokenizer_info", methods=["GET"])
        def tokenizer_info():
            info = {"tokenizer_path": getattr(self.tokenizer, "path", None)}
            return jsonify(info)

        @app.route("/load_model", methods=["POST"])
        def load_model():
            # payload: {"checkpoint_path": "..."} optional
            data = request.get_json() or {}
            ck = data.get("checkpoint_path")
            if ck and os.path.exists(ck):
                try:
                    state = torch.load(ck, map_location=CONFIG["DEVICE"])
                    if "model_state" in state:
                        self.model.load_state_dict(state["model_state"])
                    else:
                        self.model.load_state_dict(state)
                    return jsonify({"loaded": True, "path": ck})
                except Exception as e:
                    return jsonify({"loaded": False, "error": str(e)}), 500
            return jsonify({"loaded": False, "error": "checkpoint_path não fornecido ou inválido"}), 400

        @app.route("/generate_text", methods=["POST"])
        def generate_text():
            payload = request.get_json() or {}
            prompt = payload.get("prompt", "")
            max_tokens = int(payload.get("max_tokens", CONFIG["DEFAULT_MAX_TOKENS"]))
            # checagem de segurança: não permitir mais que MAX_ALLOWED_TOKENS
            if max_tokens > CONFIG.get("MAX_ALLOWED_TOKENS", CONFIG["DEFAULT_MAX_TOKENS"]):
                return jsonify({"error": f"max_tokens maior que permitido ({CONFIG.get('MAX_ALLOWED_TOKENS')})"}), 400
            temperature = float(payload.get("temperature", 0.7))
            top_k = int(payload.get("top_k", 50))
            top_p = float(payload.get("top_p", 0.95))
            with self._lock:
                out = self.model.generate_text(self.tokenizer, prompt, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p, device=CONFIG["DEVICE"])
            return jsonify({"text": out})

        @app.route("/generate_cot", methods=["POST"])
        def generate_cot():
            payload = request.get_json() or {}
            prompt = payload.get("prompt", "")
            steps = int(payload.get("steps", CONFIG["DEFAULT_COT_STEPS"]))
            tokens_per_step = int(payload.get("tokens_per_step", CONFIG["DEFAULT_COT_TOKENS_PER_STEP"]))
            refine = bool(payload.get("refine", True))
            pensei_por_mais_tempo = bool(payload.get("pensei_por_mais_tempo", False))
            thoughts = []
            final_answer = ""
            # chain-of-thought loop
            with self._lock:
                # prepare base ids
                base_ids = self.tokenizer.encode(prompt)
                current_ids = list(base_ids)
                if pensei_por_mais_tempo:
                    tokens_per_step = int(tokens_per_step * 1.8)
                    steps = max(steps, 3)
                for s in range(steps):
                    gen_ids = self.model.generate_reasoning_step(self.tokenizer, current_ids, max_new=tokens_per_step, temperature=0.0, device=CONFIG["DEVICE"])
                    # decode only the newly generated portion
                    new_part = gen_ids[len(current_ids):]
                    thought_text = self.tokenizer.decode(new_part)
                    thoughts.append(thought_text)
                    # refine?
                    if refine:
                        critique_prompt = f"{prompt}\nPensamento: {thought_text}\nCritique e proponha uma melhora curta:"
                        crit_ids = self.tokenizer.encode(critique_prompt)
                        crit_gen = self.model.generate_reasoning_step(self.tokenizer, crit_ids, max_new=max(8, tokens_per_step//4), temperature=0.0, device=CONFIG["DEVICE"])
                        critique_text = self.tokenizer.decode(crit_gen[len(crit_ids):])
                        # build new context for next step
                        combined = f"{prompt}\nPensamento: {thought_text}\nCrítica: {critique_text}\nRefine:"
                        current_ids = self.tokenizer.encode(combined)
                    else:
                        # append thought to prompt
                        current_ids = current_ids + new_part
                    if pensei_por_mais_tempo:
                        extra = self.model.generate_reasoning_step(self.tokenizer, current_ids, max_new=max(4, tokens_per_step//8), temperature=0.0, device=CONFIG["DEVICE"])
                        ext_text = self.tokenizer.decode(extra[len(current_ids):])
                        thoughts[-1] += " | ext: " + ext_text
                        current_ids = extra
                # final answer
                final_prompt = prompt + "\n" + "\n".join([f"Passo {i+1}: {t}" for i,t in enumerate(thoughts)]) + "\nResposta final (concisa):"
                final_ids = self.tokenizer.encode(final_prompt)
                final_gen = self.model.generate_reasoning_step(self.tokenizer, final_ids, max_new=128, device=CONFIG["DEVICE"])
                final_answer = self.tokenizer.decode(final_gen[len(final_ids):])
            return jsonify({"final_answer": final_answer, "thoughts": thoughts})

        @app.route("/generate_image", methods=["POST"])
        def generate_image():
            payload = request.get_json() or {}
            prompt = payload.get("prompt", "")
            steps = int(payload.get("steps", 40))
            with self._lock:
                img_tensor = self.model.sample_image_from_text(self.tokenizer, prompt, steps=steps, device=CONFIG["DEVICE"])
            # converte tensor para PNG base64
            img = img_tensor[0].detach().cpu().clamp(0,1).numpy()
            # img shape (3,H,W) -> convert to H,W,3
            if img.shape[0] == 3:
                img = np.transpose(img, (1,2,0))
            img = (img * 255).astype("uint8")
            pil = Image.fromarray(img)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return jsonify({"image_b64": b64})

        @app.route("/shutdown", methods=["POST"])
        def shutdown():
            # só permite shutdown local
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                return jsonify({"ok": False, "error": "Not running with the Werkzeug Server"}), 500
            func()
            return jsonify({"ok": True})

    def start(self, background: bool = True):
        """Start Flask server in background thread (use background=False to block)."""
        if self._server_running:
            log.warning("Server já rodando.")
            return
        def run_app():
            self._server_running = True
            log.info(f"Iniciando servidor em http://{self.host}:{self.port} ...")
            # debug False, use_reloader False para rodar em thread única
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False, threaded=True)
            self._server_running = False
            log.info("Servidor finalizado.")
        if background:
            t = threading.Thread(target=run_app, daemon=True)
            t.start()
            self._thread = t
            # espera pequeno para confirmar
            time.sleep(0.5)
        else:
            run_app()

    def stop(self):
        try:
            requests.post(f"http://{self.host}:{self.port}/shutdown")
        except Exception:
            pass

# ---------------------------
# Cliente SDK — conversa com o servidor local via HTTP
# ---------------------------
class GPTMini8Client:
    def __init__(self, server_host: str = CONFIG["SERVER_HOST"], server_port: int = CONFIG["SERVER_PORT"], model:Optional[Any]=None, tokenizer:Optional[RealTokenizer]=None):
        self.base_url = f"http://{server_host}:{server_port}"
        self.model = model
        self.tokenizer = tokenizer
        self._session = requests.Session()

    def status(self) -> Dict[str,Any]:
        r = self._session.get(f"{self.base_url}/status", timeout=3)
        return r.json()

    def tokenizer_info(self) -> Dict[str,Any]:
        try:
            r = self._session.get(f"{self.base_url}/tokenizer_info", timeout=3)
            return r.json()
        except Exception:
            return {}

    def generate_text(self, prompt:str, max_tokens:int=None, temperature:float=0.7, top_k:int=50, top_p:float=0.95, fallback_inprocess:bool=True):
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens or CONFIG["DEFAULT_MAX_TOKENS"],
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        }
        # checagem local também
        if payload["max_tokens"] > CONFIG.get("MAX_ALLOWED_TOKENS", CONFIG["DEFAULT_MAX_TOKENS"]):
            raise ValueError(f"max_tokens maior que permitido ({CONFIG.get('MAX_ALLOWED_TOKENS')})")
        try:
            r = self._session.post(f"{self.base_url}/generate_text", json=payload, timeout=60)
            return r.json().get("text")
        except Exception as e:
            log.warning(f"Request falhou: {e}. Tentando em-processo fallback.")
            if fallback_inprocess and self.model and self.tokenizer:
                return self.model.generate_text(self.tokenizer, prompt, max_new_tokens=max_tokens or CONFIG["DEFAULT_MAX_TOKENS"], temperature=temperature, top_k=top_k, top_p=top_p, device=CONFIG["DEVICE"])
            raise

    def generate_cot(self, prompt:str, steps:int=None, tokens_per_step:int=None, refine:bool=True, pensei_por_mais_tempo:bool=False, fallback_inprocess:bool=True):
        payload = {"prompt": prompt, "steps": steps or CONFIG["DEFAULT_COT_STEPS"], "tokens_per_step": tokens_per_step or CONFIG["DEFAULT_COT_TOKENS_PER_STEP"], "refine": refine, "pensei_por_mais_tempo": pensei_por_mais_tempo}
        try:
            r = self._session.post(f"{self.base_url}/generate_cot", json=payload, timeout=120)
            return r.json()
        except Exception as e:
            log.warning(f"Request falhou: {e}. Tentando fallback in-process.")
            if fallback_inprocess and self.model and self.tokenizer:
                # replicar localmente (usando model.generate_reasoning_step)
                cot = { "final_answer": "", "thoughts": [] }
                # simplified local fallback: use client-side loop
                current_prompt = prompt
                steps = payload["steps"]
                tokens_per_step = payload["tokens_per_step"]
                for s in range(steps):
                    gen = self.model.generate_text(self.tokenizer, current_prompt, max_new_tokens=tokens_per_step, temperature=0.0, top_k=0, top_p=1.0, device=CONFIG["DEVICE"])
                    # extract generated part
                    if gen.startswith(current_prompt):
                        new = gen[len(current_prompt):].strip()
                    else:
                        new = gen
                    cot["thoughts"].append(new)
                    if refine:
                        critique_prompt = f"{current_prompt}\nPensamento: {new}\nCritique e melhore:"
                        critique = self.model.generate_text(self.tokenizer, critique_prompt, max_new_tokens=max(8, tokens_per_step//4), temperature=0.0, top_k=0, top_p=1.0, device=CONFIG["DEVICE"])
                        current_prompt = f"{prompt}\nPensamento: {new}\nCrítica: {critique}\nRefine:"
                    else:
                        current_prompt = current_prompt + "\nPensamento: " + new
                    if pensei_por_mais_tempo:
                        extra = self.model.generate_text(self.tokenizer, current_prompt, max_new_tokens=max(4, tokens_per_step//8), temperature=0.0, top_k=0, top_p=1.0, device=CONFIG["DEVICE"])
                        cot["thoughts"][-1] += " | ext: " + extra
                # final answer
                final_prompt = prompt + "\n" + "\n".join([f"Passo {i+1}: {t}" for i,t in enumerate(cot["thoughts"])]) + "\nResposta final:"
                final = self.model.generate_text(self.tokenizer, final_prompt, max_new_tokens=128, temperature=0.0, top_k=0, top_p=1.0, device=CONFIG["DEVICE"])
                cot["final_answer"] = final
                return cot
            raise

    def generate_image(self, prompt:str, steps:int=40, save_path:Optional[str]=None, fallback_inprocess:bool=True):
        payload = {"prompt": prompt, "steps": steps}
        try:
            r = self._session.post(f"{self.base_url}/generate_image", json=payload, timeout=300)
            data = r.json()
            b64 = data.get("image_b64")
            if not b64:
                raise RuntimeError("Resposta sem image_b64")
            img_bytes = base64.b64decode(b64)
            if save_path:
                with open(save_path, "wb") as f:
                    f.write(img_bytes)
                return save_path
            else:
                return Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            log.warning(f"Request falhou: {e}. Tentando fallback in-process.")
            if fallback_inprocess and self.model and self.tokenizer:
                t = self.model.sample_image_from_text(self.tokenizer, prompt, steps=steps, device=CONFIG["DEVICE"])
                # converte tensor para PIL
                img = t[0].detach().cpu().clamp(0,1).numpy()
                if img.shape[0] == 3:
                    img = np.transpose(img, (1,2,0))
                pil = Image.fromarray((img * 255).astype("uint8"))
                if save_path:
                    pil.save(save_path)
                    return save_path
                return pil
            raise

# ---------------------------
# CLI / utilitários
# ---------------------------
def start_server_interactive(host: str = CONFIG["SERVER_HOST"], port:int = CONFIG["SERVER_PORT"], tokenizer_path:Optional[str]=None, checkpoint_path:Optional[str]=None):
    """Convenience: cria servidor, tenta carregar tokenizer/checkpoint e start."""
    if not HAS_ALL:
        raise RuntimeError("Dependências faltando. Instale as libs necessárias.")
    # carregar tokenizer se existir
    tk = None
    if tokenizer_path and os.path.exists(tokenizer_path):
        tk = RealTokenizer.load(tokenizer_path)
    else:
        tk = RealTokenizer(path=tokenizer_path or CONFIG["TOKENIZER_PATH"], auto_train=True)
    model = GPTMini8Model()
    server = ModelServer(host=host, port=port, model=model, tokenizer=tk)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ck = torch.load(checkpoint_path, map_location=CONFIG["DEVICE"])
            if "model_state" in ck:
                model.load_state_dict(ck["model_state"])
            else:
                model.load_state_dict(ck)
            log.info("Checkpoint carregado.")
        except Exception as e:
            log.warning(f"Falha ao carregar checkpoint: {e}")
    server.start(background=True)
    log.info("Servidor rodando. Use CTRL+C para parar (se rodou no foreground) ou chame server.stop().")
    return server

# ---------------------------
# Exemplo de uso (apenas se executado como script)
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-Mini8 Local SDK — servidor + cliente")
    parser.add_argument("--start-server", action="store_true", help="Inicia servidor local (background).")
    parser.add_argument("--host", type=str, default=CONFIG["SERVER_HOST"])
    parser.add_argument("--port", type=int, default=CONFIG["SERVER_PORT"])
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--demo-client", action="store_true", help="Roda demo do cliente (envia requests ao servidor).")
    args = parser.parse_args()

    if args.start_server:
        srv = start_server_interactive(host=args.host, port=args.port, tokenizer_path=args.tokenizer, checkpoint_path=args.checkpoint)
        log.info("Servidor iniciado em background. Rode outro processo para usar a SDK.")
        # manter processo vivo (foreground) caso queira
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Interrupção recebida. Parando servidor...")
            srv.stop()
            time.sleep(0.5)
    elif args.demo_client:
        # tenta conectar
        client = GPTMini8Client(server_host=args.host, server_port=args.port)
        try:
            st = client.status()
            print("STATUS:", st)
            ti = client.tokenizer_info()
            print("TOKENIZER:", ti)
            t = client.generate_text("Explique em 2 linhas o que é uma árvore de decisão:", max_tokens=60)
            print("TEXTO:", t)
            cot = client.generate_cot("Prove que √2 é irracional:", steps=3, tokens_per_step=24, pensei_por_mais_tempo=True)
            print("COT:", cot)
        except Exception as e:
            print("Falha no demo client:", e)
    else:
        print("Sem argumento. Use --start-server para iniciar servidor ou --demo-client para demo.")
