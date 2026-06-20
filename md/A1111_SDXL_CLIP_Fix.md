# A1111 における Pony / Illustrious の CLIP 問題 — 完全解説

> **ベースコミット:** `f175fb0984aaf4478d3233d402bf5111e790d120` (v1.15)
>
> 本ドキュメントに記載の「修正前」コードはすべてこのコミット時点の状態であり、「修正後」コードがこのコミットに対する変更内容である。

## 概要

A1111 (Stable Diffusion WebUI) で SDXL 派生モデル（Pony, Illustrious 等）を使用した際、以下の問題が発生していた。

1. **画像がノイズ化する / LoRA が効かない**（CLIP-L 側の問題として疑われた）
2. **モデルのロード自体が RuntimeError で失敗する**（CLIP-G 側の問題）★実際に修正が必要だったのはこちら

本ドキュメントでは、問題の技術的な背景、調査過程、および最終的に適用した修正内容を完全かつ詳細に解説する。

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

Pony や Illustrious は、**ComfyUI の CLIP エンコード挙動を前提として学習**されている。したがって、A1111 が ComfyUI と異なる CLIP 挙動をしていると、学習時と推論時で embedding の分布が乖離し、画像が崩れる・LoRA が効かないなどの問題が発生する可能性がある。

---

## 調査1：CLIP-L 側 — 層選択・layer_norm の差異（結果的に修正不要）

### ファイル：`modules/sd_hijack_clip.py`

### 疑われた問題

Pony / Illustrious は ComfyUI の以下の挙動で学習されている：

- `hidden_states[-2]` を使用（最終層の一つ手前の出力）
- `final_layer_norm` を**適用しない**（`layer_norm_hidden_state=False`）

一方、A1111 の `FrozenCLIPEmbedderForSDXLWithCustomWords.encode_with_transformers` は：

```python
# f175fb0 時点のコード（現在も変更なし）

class FrozenCLIPEmbedderForSDXLWithCustomWords(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

    def encode_with_transformers(self, tokens):
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=self.wrapped.layer == "hidden")

        if opts.sdxl_clip_l_skip is True:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers]
        elif self.wrapped.layer == "last":
            z = outputs.last_hidden_state
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx]

        return z
```

コード上の違いは確かに存在する：

| 項目 | A1111 | ComfyUI / Forge |
|---|---|---|
| `output_hidden_states` | `layer == "hidden"` 時のみ | 常に `True` |
| 使用する層 | `last_hidden_state` | `hidden_states[-2]` |
| `final_layer_norm` | 暗黙に適用 | 適用しない |

### 結論：修正不要

CLIP-G 側の `batch_first` 修正のみを適用した状態で、Pony / Illustrious 共にベース生成・LoRA 効果ともに正常に動作することを実証済み。CLIP-L の層選択・`final_layer_norm` の差異は、実用上問題ないことが確認された。

> **教訓:** コード上の差異があっても、実証的に問題がなければ修正は不要。推測で修正を追加すべきではない。

---

## 問題2：CLIP-G 側 — モデルロード時の RuntimeError（実際に修正）

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

## 修正3：不要なワークアラウンドの削除

### ファイル：`modules/sd_hijack_open_clip.py`

### 背景

CLIP-G の `batch_first` 修正前に、`FrozenOpenCLIPEmbedder2WithCustomWords.encode_with_transformers` に暫定的なワークアラウンドが存在していた。これは `attn_mask` を一時的に `None` に設定してエラーを回避するハックだった。

### 修正前

```python
# 修正前 — attn_mask を一時的に無効化するハック

def encode_with_transformers(self, tokens):
    original_attn_mask = None
    if hasattr(self.wrapped.model, 'attn_mask') and self.wrapped.model.attn_mask is not None:
        original_attn_mask = self.wrapped.model.attn_mask
        if original_attn_mask.shape == (77, 77):
            self.wrapped.model.attn_mask = None  # Temporarily disable mask

    try:
        d = self.wrapped.encode_with_transformer(tokens)
        z = d[self.wrapped.layer]
        pooled = d.get("pooled")
        if pooled is not None:
            z.pooled = pooled
        return z
    finally:
        if original_attn_mask is not None:
            self.wrapped.model.attn_mask = original_attn_mask
```

### 修正後

```python
# 修正後 — ハック削除、シンプルに呼び出し

def encode_with_transformers(self, tokens):
    d = self.wrapped.encode_with_transformer(tokens)
    z = d[self.wrapped.layer]

    pooled = d.get("pooled")
    if pooled is not None:
        z.pooled = pooled

    return z
```

`modules.py` 側で `batch_first` を正しく処理したため、`attn_mask` を無効化する必要がなくなった。

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

| | open_clip モデル定義 | open_clip transformer 実行 | batch_first 影響 |
|---|---|---|---|
| **ComfyUI** | 使わない | 使わない（独自実装） | なし |
| **Forge** | 使う（重みロード用） | 使わない（HF Transformers 経由） | なし |
| **A1111（修正前）** | 使う | 使う（SGM 経由） | **直撃** |
| **A1111（修正後）** | 使う | 使う（条件付き permute で対応） | 対応済み |

---

## 修正ファイル一覧

| ファイル | 修正内容 |
|---|---|
| `repositories/generative-models/sgm/modules/encoders/modules.py` | `FrozenOpenCLIPEmbedder2` と `FrozenOpenCLIPEmbedder` の計3メソッドに `batch_first` 条件分岐を追加 |
| `modules/sd_hijack_open_clip.py` | 不要になった `attn_mask` 一時無効化ハックを削除 |

> **注意:** `modules/sd_hijack_clip.py` は調査対象だったが、最終的に変更していない。CLIP-G の `batch_first` 修正だけで Pony / Illustrious 共に正常動作（ベース生成・LoRA 効果ともに）を確認済み。

---

## 修正の検証

修正後、以下のモデルで正常動作を確認：

- ✅ **Pony 系列・Illustrious 系列ベースモデル** — 正常な画像生成（ノイズ化なし）、LoRA 効果も正常
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

## 調査の経緯

### 迷走期間

Pony / Illustrious でノイズ化・LoRA 不効の問題が発生してから、約1年間にわたり原因調査を行った。初期の調査では以下の仮説を立てて試行錯誤したが、いずれも的外れだった：

- **v-prediction 判定の誤り** — Pony / Illustrious は v-prediction を使用しないにもかかわらず、ファイル名ベースの v-prediction 自動検出ハックを実装（後に revert）
- **GroupNorm / LayerNorm の dtype 問題** — bfloat16 の NaN や mixed dtype を疑い、in-place `self.float()` キャストなどを試行（後に revert）
- **VAE ロードの最適化** — VAE 周りの挙動を疑ったが関係なし

これらの試行錯誤は全て UNet 側（ノイズ生成側）の問題と想定していたが、実際はテキストエンコーダ側（CLIP）の問題だった。

### 突破口

Forge と ComfyUI のコードを厳密に比較した結果、以下のアーキテクチャ差異を発見：

1. ComfyUI は open_clip 非依存（独自実装）
2. Forge は open_clip のモデル定義のみ利用し、実行は HuggingFace Transformers 経由
3. A1111 だけが SGM 経由で open_clip の内部実装に完全依存

この構造的差異により、open_clip 3.1.0 の `batch_first` デフォルト変更が A1111 に直撃したことが判明。`batch_first` 条件分岐を追加することで問題を解決した。

### 重要な教訓

- **推測ではなく、実際にコードを読む** — Forge と ComfyUI のコードを比較することが突破口だった
- **コード上の差異 ≠ 必ず修正が必要** — CLIP-L 側には ComfyUI との差異があったが、実証的に問題がなければ修正不要
- **症状から原因を推定しない** — ノイズ化＝v-prediction や dtype 問題と推測したが、実際は CLIP-G の `batch_first` だった

---

## まとめ

今回の問題の本質は **A1111 が SGM + open_clip というレガシーな依存関係にあり、依存ライブラリ（open_clip）のバージョンアップに追従していなかった** こと。

実際に適用した修正は CLIP-G 側の `batch_first` 対応のみ。CLIP-L 側は調査の結果、修正不要と判断した。Forge と ComfyUI は独自のテキストエンコード経路を持っていたため影響を受けなかった。

修正は「新旧どちらの open_clip でも動くように `getattr` によるフォールバック付き判定を使用」するアプローチで、後方互換性を保ちながら問題を解決した。
