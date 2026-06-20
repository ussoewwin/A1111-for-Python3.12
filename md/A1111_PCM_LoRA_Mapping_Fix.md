# A1111 における PCM / Turbo 系 SDXL 高速化 LoRA のマッピング問題 — 完全解説

> **ベースコミット:** `1fc455d` (v2.0, 誤修正revert後)
>
> **修正コミット:** `53be7f5`

---

## 概要

PCM (Phased Consistency Model) や Turbo 系の SDXL 高速化 LoRA（`pcm_sdxl_normalcfg_16step_converted.safetensors` 等）を A1111 でロードすると、`3/2364 unmatched keys` の警告が出て一部のキーが正しく適用されなかった。これらの LoRA は Forge や ComfyUI では正常に動作するが、A1111 では開発終了後にリリースされたため対応していなかった。

本ドキュメントでは、問題の技術的な背景、原因、修正内容を完全かつ詳細に解説する。

---

## 問題の現象

PCM LoRA をロードした際、コンソールに以下の警告が出力される：

```
WARNING:root:[LORA] Loading pcm_sdxl_normalcfg_16step_converted.safetensors for OpenAIWrapper with 3/2364 unmatched keys
```

デバッグ出力で確認したところ、不一致キーは以下の 1 モジュール（3 サブキー）だった：

```
lora_key: lora_unet_up_blocks_0_upsamplers_0_conv.alpha
lora_key: lora_unet_up_blocks_0_upsamplers_0_conv.lora_down.weight
lora_key: lora_unet_up_blocks_0_upsamplers_0_conv.lora_up.weight
  -> compvis_key: diffusion_model_output_blocks_2_1_conv
```

`up_blocks_0` の upsampler（アップサンプラー）だけが A1111 の UNet モジュールにマッピングできていなかった。

---

## 原因：upsampler の type index 誤り

### A1111 の LoRA キーマッピングの仕組み

PCM LoRA は Kohaku-ss 形式（Diffusers 互換）のキー名を使用する：

```
lora_unet_up_blocks_0_upsamplers_0_conv
```

A1111 の UNet は CompVis 形式のモジュール名を使用するため、キー名の変換が必要。この変換は `extensions-builtin/Lora/networks.py` の `convert_diffusers_name_to_compvis()` 関数が担当する。

### バグのあったコード

```python
# 修正前（バグあり）
if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
    return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"
```

`up_blocks_0`（`m[0]=0`）の場合：
- インデックス: `2 + 0 * 3 = 2`
- type: `1`（`m[0] > 0` が偽のため）
- 結果: `diffusion_model_output_blocks_2_1_conv`

### なぜ間違いなのか

A1111 の SDXL UNet 出力ブロックの構造：

| output_blocks インデックス | 内容 | type |
|---|---|---|
| 0 | up_0 attention_0 (10 transformer blocks) | 1 |
| 1 | up_0 attention_1 (10 transformer blocks) | 1 |
| 2 | up_0 attention_2 (10 transformer blocks) | 1 |
| **2** | **up_0 upsampler** | **2** |
| 3 | up_1 attention_0 (2 transformer blocks) | 1 |
| ... | ... | ... |

`output_blocks_2` には **2 つのモジュール** が存在する：

1. `output_blocks_2_1_*`: attention_2（type=1）
2. `output_blocks_2_2_conv`: upsampler（type=2）

バグのコードは upsampler に **type=1** を割り当てていたため、`output_blocks_2_1_conv` となってしまった。しかし、type=1 の position は attention 用であり、upsampler 用の `conv` 属性は存在しない。そのため `network_layer_mapping` で検索失敗 → unmatched。

### なぜ `up_blocks_0` だけ問題なのか

条件式 `2 if m[0]>0 else 1` は：
- `up_blocks_1`（m[0]=1）: type=2 → `output_blocks_5_2_conv` ✅ 正しい
- `up_blocks_0`（m[0]=0）: type=1 → `output_blocks_2_1_conv` ❌ 間違い

SD 1.x/2.x の UNet 構造では up_blocks_0 に upsampler が type=1 の位置にあったことがあるため、この条件分岐が残っていたと考えられる。SDXL では常に type=2 が正しい。

### 参考：Forge / ComfyUI ではなぜ動くのか

- **`unet_to_diffusers` マップ**（`unet_diffusers_map.py`）は `up_blocks.0.upsamplers.0.conv` → `output_blocks.2.2.conv` と正しくマッピングしている
- `_register_diffusers_unet_aliases` はこのマップを使って正しいエイリアスを作成する
- **しかし**、upsampler の `conv_shortcut` や `embed` などの特殊キーに対する `network_layer_mapping` の検索でモジュールが `None` になるケースがあり、_register_diffusers_unet_aliases でのエイリアス登録がスキップされる
- その結果フォールバックとして `convert_diffusers_name_to_compvis` が使われ、バグのある type 判定が適用された

---

## 修正内容

### 修正ファイル

`extensions-builtin/Lora/networks.py`

### 修正前

```python
if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
    return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"
```

### 修正後

```python
if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
    return f"diffusion_model_output_blocks_{2 + m[0] * 3}_2_conv"
```

### 修正の意味

- 条件分岐 `2 if m[0]>0 else 1` を削除し、常に **type=2** を使用
- SDXL UNet では upsampler は常に attention ブロックの後に配置され、type=2 の位置になる
- インデックス計算 `2 + m[0] * 3` は変更なし（正しい）

---

## 修正の検証

### 修正前

```
[LORA DEBUG] Unmatched keys for pcm_sdxl_normalcfg_16step_converted.safetensors:
  -> compvis_key: diffusion_model_output_blocks_2_1_conv  (3 keys)
```

### 修正後

ユーザー確認: **警告消滅**。「直った」とのこと。全 2364 キーが正常にマッピング。

### 影響範囲

- **修正対象**: SDXL 用 Kohaku-ss 形式 LoRA の `up_blocks_0_upsamplers_0_conv` キー
- **後方互換性**: SD 1.x/2.x では `up_blocks_0` に upsampler キーが存在しない、または別の mapping で処理されるため影響なし
- `up_blocks_1` 以降は元々 type=2 が正しかったため変更なし

---

## 対象 LoRA

| LoRA ファイル | キー数 | 修正要否 |
|---|---|---|
| `pcm_sdxl_normalcfg_16step_converted.safetensors` | 2364 | ✅ 修正対象 |
| `pcm_sdxl_smallcfg_16step_converted.safetensors` | 2364 | ✅ 修正対象（同一構造） |
| `pcm_sd15_normalcfg_16step_converted.safetensors` | 834 | ❌ 修正不要（SD1.5） |
| `pcm_sd15_smallcfg_16step_converted.safetensors` | 834 | ❌ 修正不要（SD1.5） |

---

## 環境情報

| 項目 | 値 |
|---|---|
| A1111 バージョン | v2.0 |
| Python | 3.12.10 |
| PyTorch | 2.12.1+cu132 |
| CUDA | 13.2 |
| Flash-Attention | 2.9.1 |
| open_clip | 3.1.0 |
| リポジトリ | `ussoewwin/A1111-for-Python3.12` |

---

## 修正ファイル一覧

| ファイル | 修正内容 |
|---|---|
| `extensions-builtin/Lora/networks.py` | `convert_diffusers_name_to_compvis()`: upsampler type index を常に 2 に修正 |

---

## 調査経緯

### 誤ったアプローチ（revert 済み）

当初、SDXL UNet のブロック内モジュール順序が Diffusers 形式（attentions → resnets）と CompVis 形式（resnets → attentions）で異なると仮定し、resnet インデックスにオフセットを追加する修正を行った。この修正は逆に状況を悪化させ（3 unmatched → 54 unmatched）、即座に revert した。

得られた教訓：A1111 の CompVis UNet はブロック内で resnets と attentions がペアで交互に配置され、両者は同じブロックインデックスを共有し、type (0 vs 1) で区別される。Diffusers の attentions → resnets 順序とは内部表現が根本的に異なる。

### 正しいアプローチ

1. `load_network()` にデバッグログを追加し、実際の不一致キーを特定
2. 不一致キーが `lora_unet_up_blocks_0_upsamplers_0_conv` の 3 サブキーのみであることを確認
3. `compvis_key` が `diffusion_model_output_blocks_2_1_conv`（type=1）になっていることを発見
4. `unet_to_diffusers` の正しいマッピング（type=2）と比較し、`convert_diffusers_name_to_compvis` の type 判定がバグっていることを特定
5. 条件分岐を削除し常に type=2 に修正

---

## まとめ

今回の問題の本質は、`convert_diffusers_name_to_compvis()` の upsampler マッピングにおける **type index の条件分岐バグ** だった。

SDXL UNet では upsampler は常に type=2 の位置にあるが、コードにはレガシーな条件分岐 `2 if m[0]>0 else 1` が残っており、`up_blocks_0` の場合だけ誤って type=1（attention 位置）にマッピングされていた。

修正は 1 行の条件分岐削除のみ。影響範囲は極めて限定的で、後方互換性を完全に保つ。
