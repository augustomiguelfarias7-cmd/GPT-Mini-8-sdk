GPT-Mini 8 SDK

Biblioteca multimodal Python para o GPT-Mini 8, com suporte a:

- Geração de texto
- Raciocínio passo-a-passo (Chain-of-Thought / COT)
- Geração de imagens integrada
- Integração via servidor Flask local

---

## Instalação

Instale diretamente do repositório:

```bash
pip install git+https://github.com/augustomiguelfarias7-cmd/gpt_mini8_sdk.git

_____________________________________

Usando o servidor Flask local

Para iniciar o servidor:

python gpt_mini8_local_sdk.py --start-server

O servidor roda por padrão em http://127.0.0.1:7860.


---

Exemplos de uso via API

from gpt_mini8_sdk import GPTMini8Client

# Criar cliente conectando ao servidor local
client = GPTMini8Client(server_host="127.0.0.1", server_port=7860)

# Gerar texto
texto = client.generate_text(
    "Explique rapidamente o que é uma árvore de decisão:",
    max_tokens=100
)
print(texto)

# Gerar raciocínio passo-a-passo (COT)
cot = client.generate_cot(
    "Prove que √2 é irracional",
    steps=3,
    tokens_per_step=32,
    refine=True
)
print(cot)

# Gerar imagem (integrado)
imagem = client.generate_image("Um dinossauro pintando um quadro colorido")
imagem.show()


---

Exemplos de uso direto (in-process)

from gpt_mini8_sdk import GPTMini8Model, RealTokenizer

# Inicializar tokenizer e modelo
tokenizer = RealTokenizer()
model = GPTMini8Model()

# Gerar texto
saida = model.generate_text(tokenizer, "O que é aprendizado de máquina?", max_new_tokens=60)
print(saida)

# Gerar COT diretamente
cot_direto = model.generate_cot(tokenizer, "Resolva 2x + 3 = 7", steps=2, tokens_per_step=32)
print(cot_direto)

# Gerar imagem integrada
imagem_direta = model.generate_image("Um robô pintando um quadro futurista")
imagem_direta.show()


---

Observações

Tudo roda internamente, não é necessário baixar nenhum outro modelo ou peso.

Instalar a biblioteca já deixa o GPT-Mini 8 pronto para uso, com textos, COT e imagens.

A biblioteca funciona offline após a instalação, sem depender de servidores externos.
