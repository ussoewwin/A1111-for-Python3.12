# A1111 における Pony / Illustrious の CLIP 問題 — 完全解説

> **ベースコミット:** `f175fb0984aaf4478d3233d402bf5111e790d120` (v1.15)
>
> 本ドキュメントに記載の「修正前」コードはすべてこのコミット時点の状態であり、「修正後」コードがこのコミットに対する変更内容である。

## 概要

A1111 (Stable Diffusion WebUI) で SDXL 派生モデル（Pony, Illustrious 等）を使用した際、以下の2つの問題が発生していた。

1. **画像がノイズ化する / LoRA が効かない**（CLIP-L 側の問題）
2. **モデルのロード自体が RuntimeError で失敗する**（CLIP-G 側の問題）

本ドキュメントでは、問題の技術的な背景、原因、および修正内容を完全かつ詳細に解説する。

---

## 背景：SDXL のテキストエンコーダ構成

SDXL は2つのテキストエンコーダを並列で使用する：

| エンコーダ | モデル | 出力次元 | 用途 |
|---|---|---|---|
| **CLIP-L** | `openai/clip-vit-large-patch14` (HuggingFace Transformers) | 768 | テキストの埋め込み（その1） |
| **CLIP-G** | `ViT-bigG-14` (open_clip) | 1280 | テキストの埋め込み（その2）＋ pooled embedding |

両者の出力は channel 次元で結合され（768 + 1280 = 2048）、UNet の cross-attention 層に入力される。

### SDXL モデルにおける CLIP の役割

```
プロンプトテキスト
    │
    ├──→ CLIP-L (HuggingFace Transformers) ──→ hidden_states[-2] (768次元)
    │                                              ↓
    ├──→ CLIP-G (open_clip) ──────────────────→ hidden_states[-2] (1280次元) + pooled
    │                                              ↓
    │                                    torch.cat([L, G], dim=-1) → 2048次元
    │                                              ↓
    └──────────────────────────────────────→ UNet cross-attention
```

Pony や Illustrious は、**ComfyUI の CLIP エンコード挙動を前提として学習**されている。したがって、A1111 が ComfyUI と異なる CLIP 挙動をしていると、学習時と推論時で embedding の分布が乖離し、画像が崩れる・LoRA が効かないなどの問題が発生する。

---

## 問題1：CLIP-L 側 — ノイズ化・LoRA 不効

### ファイル：`modules/sd_hijack_clip.py`

### 問題の本質

Pony / Illustrious は ComfyUI の以下の挙動で学習されている：

- `hidden_states[-2]` を使用（最終層の一つ手前の出力）
- `final_layer_norm` を**適用しない**（`layer_norm_hidden_state=False`）

しかし、A1111（修正前）の `FrozenCLIPEmbedderForSDXLWithCustomWords.encode_with_transformers` は：

- `self.wrapped.layer == "last"` の場合、`outputs.last_hidden_state` を使用（全レイヤー通過 + `final_layer_norm` 適用後）
- つまり**全く別の分布空間**の embedding を生成していた

### 修正前のコード

```python
# modules/sd_hijack_clip.py — FrozenCLIPEmbedderForSDXLWithCustomWords

class FrozenCLIPEmbedderForSDXLWithCustomWords(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

    def encode_with_transformers(self, tokens):
        # output_hidden_states は layer == "hidden" の時しか True にならない
        outputs = self.wrapped.transformer(
            input_ids=tokens,
            output_hidden_states=self.wrapped.layer == "hidden"
        )

        if opts.sdxl_clip_l_skip is True:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers]
        elif self.wrapped.layer == "last":
            # ← 問題: last_hidden_state は全レイヤー + final_layer_norm 通過後
            z = outputs.last_hidden_state
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx]

        return z
```

**問題点：**
1. `self.wrapped.layer == "last"` の場合、`output_hidden_states=True` にならないため `hidden_states` にアクセスできない
2. `last_hidden_state` は全 transformer レイヤー通過後 + `final_layer_norm` 適用後の出力
3. Pony / Illustrious が学習に使用した `hidden_states[-2]`（final_layer_norm なし）とは**別物**

### 修正後のコード

```python
# modules/sd_hijack_clip.py — FrozenCLIPEmbedderForSDXLWithCustomWords

class FrozenCLIPEmbedderForSDXLWithCustomWords(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

    def encode_with_transformers(self, tokens):
        # 常に hidden_states を取得（ComfyUI/Forge と同じ挙動）
        outputs = self.wrapped.transformer(
            input_ids=tokens,
            output_hidden_states=True
        )

        # hidden_states[-max(clip_skip, 2)] を使用（最小 clip_skip=2）
        # ComfyUI: hidden_states[-2], Forge: minimal_clip_skip=2
        layer_id = -max(opts.CLIP_stop_at_last_layers, 2)
        z = outputs.hidden_states[layer_id]

        # final_layer_norm を適用しない
        # ComfyUI: layer_norm_hidden_state=False
        # Forge: final_layer_norm=False (for Pony/Illustrious)

        return z
```

### 修正のポイント

| 項目 | 修正前 | 修正後 | ComfyUI / Forge |
|---|---|---|---|
| `output_hidden_states` | `layer == "hidden"` 時のみ | **常に `True`** | 常に `True` |
| 使用する層 | `last_hidden_state` | `hidden_states[-max(clip_skip, 2)]` | `hidden_states[-2]` |
| `final_layer_norm` | 暗黙に適用（`last_hidden_state` に含まれる） | **適用しない** | `layer_norm_hidden_state=False` |

### なぜ `final_layer_norm` が問題なのか

`final_layer_norm` は transformer 最終層の出力に対する Layer Normalization である。Pony / Illustrious の学習時に ComfyUI はこれをスキップしていた（`layer_norm_hidden_state=False`）。

`final_layer_norm` を通すと通さないとでは、embedding の統計的分布（平均・分散）が大きく異なる。LoRA は cross-attention 層で特定の conditioning パターンを増強する仕組みだが、入力される embedding の分布が学習時と異なると、LoRA の重み調整が見当違いの場所に作用する → **LoRA が効かない、または画像が崩れる**。

### `clip_skip` の最小値を 2 にする理由

ComfyUI は SDXL の場合 `layer="hidden"`, `layer_idx=-2` で固定している。つまり常に最終層の一つ手前（penultimate）を使用する。A1111 の `CLIP_stop_at_last_layers` が 1 に設定されていても、`max(clip_skip, 2)` により強制的に 2 以上になる。

### float32 キャストについて

Forge の `ClassicTextProcessingEngine.encode_with_transformers` では、`position_embedding` と `token_embedding` を `float32` にキャストしている：

```python
self.text_encoder.transformer.embeddings.position_embedding = self.text_encoder.transformer.embeddings.position_embedding.to(dtype=torch.float32)
self.text_encoder.transformer.embeddings.token_embedding = self.text_encoder.transformer.embeddings.token_embedding.to(dtype=torch.float32)
```

これは、混合精度（fp16/bf16）環境で embedding の精度が落ちるのを防ぐため。A1111 側でも同様の対応が必要な場合があるが、`FrozenCLIPEmbedderForSDXLWithCustomWords` は HuggingFace Transformers を経由するため、`@autocast` デコレータの挙動に依存する。必要に応じて個別にキャストを追加する。

---

## 問題2：CLIP-G 側 — モデルロード時の RuntimeError

### ファイル：`repositories/generative-models/sgm/modules/encoders/modules.py`

### 問題の本質

`open_clip 3.1.0` で `nn.MultiheadAttention` の `batch_first` パラメータのデフォルトが `True`（NLD形式）に変更された。しかし、SGM (Stability Generative Models) のコードは `batch_first=False`（LND形式）前提で書かれており、無条件で `permute(1, 0, 2)` を実行していた。

### batch_first とは何か

`nn.MultiheadAttention` は入力テンソルの形状解釈を `batch_first` パラメータで切り替える：

| `batch_first` | 期待される形状 | 解釈 |
|---|---|---|
| `False`（旧デフォルト） | `(L, N, D)` | Length, Batch, Dim |
| `True`（新デフォルト 3.1.0+） | `(N, L, D)` | Batch, Length, Dim |

CLIP テキストエンコーダの場合、トークン埋め込み後のテンソルは `(N=1, L=77, D=1280)` という形状（NLD形式）。

### 何が起きたか

SGM の `encode_with_transformer` は以下の処理を行う：

1. `token_embedding(text)` → `(N, L, D)` = `(1, 77, 1280)` — NLD形式
2. `x.permute(1, 0, 2)` → `(L, N, D)` = `(77, 1, 1280)` — LND形式に変換
3. `text_transformer_forward(x, attn_mask=...)` に渡す
4. 各 `ResidualAttentionBlock` が `nn.MultiheadAttention` を呼ぶ

**open_clip 3.1.0 未満（`batch_first=False`）：**
- `(77, 1, 1280)` は `(L=77, N=1, D=1280)` と解釈される
- `tgt_len=77, src_len=77` → attn_mask `(77, 77)` と一致 ✅

**open_clip 3.1.0 以上（`batch_first=True`）：**
- `(77, 1, 1280)` は `(N=77, L=1, D=1280)` と解釈される
- `tgt_len=1, src_len=1` → attn_mask `(77, 77)` と不一致 ❌

```
RuntimeError: The shape of the 2D attn_mask is torch.Size([77, 77]), but should be (1, 1).
```

### 修正前のコード

SGM の `generative-models/sgm/modules/encoders/modules.py` には2つの OpenCLIP エンコーダクラスがある：

#### FrozenOpenCLIPEmbedder2（SDXL の CLIP-G として使用）

```python
# 修正前

def encode_with_transformer(self, text):
    x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model] → (N, L, D)
    x = x + self.model.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND  ← 無条件で permute
    x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
    # ...

def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
    outputs = {}
    for i, r in enumerate(self.model.transformer.resblocks):
        if i == len(self.model.transformer.resblocks) - 1:
            outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD  ← 無条件で permute
        # ...
        x = r(x, attn_mask=attn_mask)
    outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD  ← 無条件で permute
    return outputs
```

#### FrozenOpenCLIPEmbedder（SD2 等で使用）

```python
# 修正前

def encode_with_transformer(self, text):
    x = self.model.token_embedding(text)  # (N, L, D)
    x = x + self.model.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND  ← 無条件で permute
    x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD  ← 無条件で permute
    x = self.model.ln_final(x)
    return x
```

### 修正後のコード

両クラスの `encode_with_transformer` と `text_transformer_forward` で、`batch_first` 属性をチェックして条件付きで permute を行うようにした。

#### FrozenOpenCLIPEmbedder2（修正後）

```python
def encode_with_transformer(self, text):
    x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
    x = x + self.model.positional_embedding
    batch_first = getattr(self.model.transformer, 'batch_first', False)
    if not batch_first:
        x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
    if self.legacy:
        x = x[self.layer]
        x = self.model.ln_final(x)
        return x
    else:
        o = x["last"]
        o = self.model.ln_final(o)
        pooled = self.pool(o, text)
        x["pooled"] = pooled
        return x

def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
    batch_first = getattr(self.model.transformer, 'batch_first', False)
    outputs = {}
    for i, r in enumerate(self.model.transformer.resblocks):
        if i == len(self.model.transformer.resblocks) - 1:
            outputs["penultimate"] = x if batch_first else x.permute(1, 0, 2)  # LND -> NLD
        if (
            self.model.transformer.grad_checkpointing
            and not torch.jit.is_scripting()
        ):
            x = checkpoint(r, x, attn_mask)
        else:
            x = r(x, attn_mask=attn_mask)
    outputs["last"] = x if batch_first else x.permute(1, 0, 2)  # LND -> NLD
    return outputs
```

#### FrozenOpenCLIPEmbedder（修正後）

```python
def encode_with_transformer(self, text):
    x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
    x = x + self.model.positional_embedding
    batch_first = getattr(self.model.transformer, 'batch_first', False)
    if not batch_first:
        x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
    if not batch_first:
        x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.model.ln_final(x)
    return x
```

### `getattr(self.model.transformer, 'batch_first', False)` の意味

`getattr` に第三引数 `False` を指定することで、`batch_first` 属性が存在しない古い open_clip でも `False` が返る。これにより、新旧どちらの open_clip でも正しく動作する：

- **open_clip < 3.1.0**（`batch_first` 属性なし、または `False`）→ permute を実行（従来通り）
- **open_clip >= 3.1.0**（`batch_first=True`）→ permute をスキップ（NLDのまま処理）

---

## 他実装との比較：なぜ Forge と ComfyUI は影響を受けなかったのか

### ComfyUI — open_clip に非依存

ComfyUI は `comfy/sd1_clip.py` + `comfy/clip_model.py` で独自の CLIP transformer 実装を持っている。open_clip パッケージを使用せず、HuggingFace Transformers の `CLIPTextModel` ベースの独自ラッパーで処理。`batch_first` 変更の影響を一切受けない。

CLIP-L 側も `layer_norm_hidden_state=False` で正しく動作するよう設計されている。

### Forge Nunchaku — open_clip のモデル構造のみ利用、実行は別経路

Forge は `backend/text_processing/classic_engine.py` の `ClassicTextProcessingEngine` でテキストエンコードを行う。このエンジンは：

- モデル構造（重みの定義）は open_clip から借用
- しかし transformer の実行は HuggingFace Transformers の `text_encoder.transformer(tokens, output_hidden_states=True)` を直接呼ぶ
- `open_clip.transformer.ResidualAttentionBlock` → `nn.MultiheadAttention` の経路を通らない

```python
# Forge Nunchaku — backend/text_processing/classic_engine.py

def encode_with_transformers(self, tokens):
    outputs = self.text_encoder.transformer(tokens, output_hidden_states=True)
    layer_id = -max(self.clip_skip, self.minimal_clip_skip)
    z = outputs.hidden_states[layer_id]
    if self.final_layer_norm:
        z = self.text_encoder.transformer.final_layer_norm(z)
    # ...
```

- `batch_first` 問題：HF Transformers の API を使うため影響なし
- CLIP-L 層選択：`minimal_clip_skip` と `final_layer_norm` パラメータで ComfyUI に合わせられる

### A1111 — open_clip に完全依存

A1111 は SGM の `generative-models` モジュールを通じて open_clip の内部実装に完全依存していた。SGM が open_clip の `ResidualAttentionBlock` → `nn.MultiheadAttention` の経路をそのまま使用するため、open_clip の内部変更が直撃する。

| | open_clip モデル定義 | open_clip transformer 実行 | batch_first 影響 | CLIP-L 層選択 |
|---|---|---|---|---|
| **ComfyUI** | 使わない | 使わない | なし | 正しい（`layer_norm_hidden_state=False`） |
| **Forge** | 使う | 使わない（HF Transformers 経由） | なし | 正しい（`final_layer_norm` パラメータ） |
| **A1111（修正前）** | 使う | 使う | **直撃** | **誤り**（`last_hidden_state` + `final_layer_norm`） |
| **A1111（修正後）** | 使う | 使う（条件付き permute で対応） | 対応済み | 正しい（`hidden_states[-2]` + norm なし） |

---

## 修正ファイル一覧

| ファイル | 修正内容 |
|---|---|
| `modules/sd_hijack_clip.py` | `FrozenCLIPEmbedderForSDXLWithCustomWords.encode_with_transformers` を書き換え |
| `repositories/generative-models/sgm/modules/encoders/modules.py` | `FrozenOpenCLIPEmbedder2` と `FrozenOpenCLIPEmbedder` の計3メソッドを修正 |

---

## 修正の検証

修正後、以下のモデルで正常動作を確認：

- ✅ **WAI Illustrious SDXL v1.7.0** — 正常な画像生成（ノイズ化なし）
- ✅ **RealVisXL V5.0** — 正常な画像生成
- ✅ モデルロード時の RuntimeError 解消
- ✅ Flash-Attention 2.9.1 + PyTorch 2.12.1+cu132 環境で稼働

---

## 環境情報

| 項目 | 値 |
|---|---|
| A1111 バージョン | v1.15 (commit f175fb0) |
| Python | 3.12.10 |
| PyTorch | 2.12.1+cu132 |
| CUDA | 13.2 |
| Flash-Attention | 2.9.1 |
| open_clip | 3.1.0 |
| リポジトリ | `ussoewwin/A1111-for-Python3.12` (Python 3.12 フォーク) |

---

## 参照リポジトリ

- Forge Nunchaku: `backend/text_processing/classic_engine.py` — `ClassicTextProcessingEngine`
- ComfyUI: `comfy/sdxl_clip.py`, `comfy/sd1_clip.py`, `comfy/clip_model.py`
- A1111: `modules/sd_hijack_clip.py`, `modules/sd_hijack_open_clip.py`
- SGM: `repositories/generative-models/sgm/modules/encoders/modules.py`

---

## まとめ

今回の問題は、2つの異なるレイヤーで発生していた：

1. **CLIP-L の層選択・layer_norm 問題** — A1111 が SGM 経由で `last_hidden_state` + `final_layer_norm` を使っていた一方、Pony / Illustrious は `hidden_states[-2]` + `final_layer_norm` なしで学習されていた。embedding の分布乖離によりノイズ化・LoRA 不効が発生。

2. **CLIP-G の batch_first 問題** — open_clip 3.1.0 で `batch_first` デフォルトが `True` に変更されたが、SGM コードが無条件 permute を行っていたため、テンソル形状の誤認識により RuntimeError が発生。

両者とも根本原因は同じ構造的問題：**A1111 が SGM + open_clip というレガシーな依存関係にあり、依存ライブラリの変更に追従していなかった**。Forge と ComfyUI は独自のテキストエンコード経路を持っていたため影響を受けなかった。

修正はいずれも「Forge / ComfyUI の正しい挙動に合わせる」アプローチで、新旧両方の open_clip で動作するよう `getattr` によるフォールバック付き判定を使用した。
